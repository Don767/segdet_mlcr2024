from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict

from mmcv.cnn import ConvModule, is_norm
from mmcv.ops import batched_nms

from mmdet.structures.bbox import distance2bbox
from mmdet.registry import TASK_UTILS, MODELS
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.utils import (
    InstanceList,
    OptInstanceList,
    reduce_mean,
    OptMultiConfig,
    OptConfigType,
)
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (
    images_to_levels,
    multi_apply,
    unmap,
)
from mmdet.structures.bbox import get_box_tensor, get_box_wh, scale_boxes


class RTMDetHead(BaseDenseHead):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        train_cfg: OptConfigType,
        test_cfg: OptConfigType,
        init_cfg: OptMultiConfig = dict(type="Normal", layer="Conv2d", std=0.01),
        **kwargs
    ) -> None:
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.use_sigmoid_cls = True
        self.in_channels = in_channels
        self.hidden_channels = feat_channels
        self.activation = dict(type="SiLU", inplace=True)
        self.norm = dict(type="SyncBN")
        self.prior_generator = TASK_UTILS.build(
            dict(type="MlvlPointGenerator", offset=0, strides=[8, 16, 32])
        )
        self.loss_cls = MODELS.build(
            dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0)
        )
        self.loss_bbox = MODELS.build(dict(type="GIoULoss", loss_weight=2.0))
        if "assigner" not in train_cfg:
            raise ValueError("assigner must be in train_cfg")
        self.assigner = TASK_UTILS.build(train_cfg["assigner"])
        if "sampler" not in train_cfg:
            self.sampler = PseudoSampler(context=self)
        else:
            self.sampler = TASK_UTILS.build(
                train_cfg["sampler"], default_args=dict(context=self)
            )
        self.bbox_coder = TASK_UTILS.build(dict(type="DistancePointBBoxCoder"))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.bbox_feat_processors = nn.ModuleList()
        self.bbox_feat_extractors = nn.ModuleList()
        self.cls_feat_processors = nn.ModuleList()
        self.cls_feat_extractors = nn.ModuleList()

        for _ in range(self.prior_generator.num_levels):
            self.bbox_feat_processors.append(self._init_processor())
            self.cls_feat_processors.append(self._init_processor())
            self.bbox_feat_extractors.append(
                nn.Conv2d(self.hidden_channels, 4, kernel_size=1, stride=1, padding=0)
            )
            self.cls_feat_extractors.append(
                nn.Conv2d(
                    self.hidden_channels,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        for n in range(self.prior_generator.num_levels):
            for i in range(2):
                self.bbox_feat_processors[n][i].conv = self.bbox_feat_processors[0][
                    i
                ].conv
                self.cls_feat_processors[n][i].conv = self.cls_feat_processors[0][
                    i
                ].conv

    def _init_processor(self) -> None:
        return nn.Sequential(
            *[
                ConvModule(
                    self.in_channels if x == 0 else self.hidden_channels,
                    self.hidden_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    norm_cfg=self.norm,
                    act_cfg=self.activation,
                )
                for x in range(2)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        cls_out = []
        bbox_out = []
        for i, (x, prior_stride) in enumerate(
            zip(inputs, self.prior_generator.strides)
        ):
            cls_feat = self.cls_feat_processors[i](x)
            cls_feat = self.cls_feat_extractors[i](cls_feat)
            cls_out.append(cls_feat)

            bbox_feat = self.bbox_feat_processors[i](x)
            bbox_feat = self.bbox_feat_extractors[i](bbox_feat)
            bbox_feat = bbox_feat.exp() * prior_stride[0]
            bbox_out.append(bbox_feat)

        return tuple(cls_out), tuple(bbox_out)

    def loss_by_feat_single(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        labels: torch.Tensor,
        label_weights: torch.Tensor,
        bbox_targets: torch.Tensor,
        assign_metrics: torch.Tensor,
        stride: List[int],
    ):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (List[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], "h stride is not equal to w stride!"
        cls_score = (
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes).contiguous()
        )
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.0)

        return loss_cls, loss_bbox, assign_metrics.sum(), pos_bbox_weight.sum()

    def loss_by_feat(
        self,
        cls_scores: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ):
        """Compute losses of the head.

        Args:
            cls_scores (list[torch.Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[torch.Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device
        )
        flatten_cls_scores = torch.cat(
            [
                cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
                for cls_score in cls_scores
            ],
            1,
        )
        decoded_bboxes = []
        for anchor, bbox_pred in zip(anchor_list[0], bbox_preds):
            anchor = anchor.reshape(-1, 4)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            bbox_pred = distance2bbox(anchor, bbox_pred)
            decoded_bboxes.append(bbox_pred)

        flatten_bboxes = torch.cat(decoded_bboxes, 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bboxes,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            _,
        ) = cls_reg_targets

        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            decoded_bboxes,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            self.prior_generator.strides,
        )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def get_targets(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        anchor_list: List[List[torch.Tensor]],
        valid_flag_list: List[List[torch.Tensor]],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        unmap_outputs=True,
    ):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[torch.Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[torch.Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: a tuple containing learning targets.

            - anchors_list (list[list[torch.Tensor]]): Anchors of each level.
            - labels_list (list[torch.Tensor]): Labels of each level.
            - label_weights_list (list[torch.Tensor]): Label weights of each
              level.
            - bbox_targets_list (list[torch.Tensor]): BBox targets of each level.
            - assign_metrics_list (list[torch.Tensor]): alignment metrics of each
              level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_assign_metrics,
            sampling_results_list,
        ) = multi_apply(
            self._get_targets_single,
            cls_scores.detach(),
            bbox_preds.detach(),
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        assign_metrics_list = images_to_levels(all_assign_metrics, num_level_anchors)

        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            assign_metrics_list,
            sampling_results_list,
        )

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        if rescale:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors

        if cfg.get("min_bbox_size", -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            with torch.cuda.amp.autocast(enabled=False):
                det_bboxes, keep_idxs = batched_nms(
                    bboxes.float(),
                    results.scores.float(),
                    results.labels.float(),
                    cfg.nms,
                )
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

        return results

    def _get_targets_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        flat_anchors: torch.Tensor,
        valid_flags: torch.Tensor,
        gt_instances: InstanceData,
        img_meta: dict,
        gt_instances_ignore: Optional[InstanceData] = None,
        unmap_outputs=True,
    ):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors. Defaults to True.

        Returns:
            tuple: N is the number of total anchors in the image.

            - anchors (Tensor): All anchors in the image with shape (N, 4).
            - labels (Tensor): Labels of all anchors in the image with shape
              (N,).
            - label_weights (Tensor): Label weights of all anchor in the
              image with shape (N,).
            - bbox_targets (Tensor): BBox targets of all anchors in the
              image with shape (N, 4).
            - norm_alignment_metrics (Tensor): Normalized alignment metrics
              of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg["allowed_border"],
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        pred_instances = InstanceData(
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :],
            priors=anchors,
        )

        assign_result = self.assigner.assign(
            pred_instances, gt_instances, gt_instances_ignore
        )

        sampling_result = self.sampler.sample(
            assign_result, pred_instances, gt_instances
        )

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (num_valid_anchors,), self.num_classes, dtype=torch.long
        )
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        assign_metrics = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg["pos_weight"] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg["pos_weight"]
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds == gt_inds]
            assign_metrics[gt_class_inds] = assign_result.max_overlaps[gt_class_inds]

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes
            )
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            assign_metrics = unmap(assign_metrics, num_total_anchors, inside_flags)
        return (
            anchors,
            labels,
            label_weights,
            bbox_targets,
            assign_metrics,
            sampling_result,
        )

    def get_anchors(
        self,
        featmap_sizes: List[tuple],
        batch_img_metas: List[dict],
        device: Union[torch.device, str] = "cuda",
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device or str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[torch.Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[torch.Tensor]]): Valid flags of each
              image.
        """
        num_imgs = len(batch_img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device, with_stride=True
        )
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for _, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta["pad_shape"], device
            )
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for cls, bbox in zip(self.cls_feat_extractors, self.bbox_feat_extractors):
            normal_init(cls, std=0.01, bias=bias_cls)
            normal_init(bbox, std=0.01)
