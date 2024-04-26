import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops as ein
from typing import List, Tuple, Union

from .transformer import Transformer
from .absolute_positional_encoding import *
from .boxes import *

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Criterion(nn.Module):
    """ Compute the loss of the model.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.num_classes = num_classes
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits, _ = outputs

        idx = self._get_src_permutation_idx(indices)
        trgt_classes, _ = targets
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(trgt_classes, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits, _ = outputs
        device = pred_logits.device
        trgt_classes, _ = targets
        tgt_lengths = torch.as_tensor([len(v) for v in trgt_classes], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        _, pred_boxes = outputs
        src_boxes = pred_boxes[idx]
        _, trgt_boxes = targets
        test = 0
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(trgt_boxes, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: zip of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = self.matcher(outputs, targets)
        # Compute the number of target boxes accross all nodes
        trgt_classes, _ = targets
        num_boxes = sum(len(t) for t in trgt_classes)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs)).device)
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        loss_dict_scaled = {k: v * self.weight_dict[k]
                                    for k, v in losses.items() if k in self.weight_dict}
        return sum(loss_dict_scaled.values())

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class DETR(torch.nn.Module):
    def __init__(self, 
                 backbone: nn.Module,
                 num_classes: int, 
                 max_detections: int, 
                 d_model: int, 
                 n_head: int,
                 num_layers_enc: int, 
                 num_layers_dec: int, 
                 dropout: float = 0.1):
        """
        DETR from "End-to-End Object Detection with Transformers" (Carion, 2020)
        :param backbone: backbone
               for example `nn.Sequential(*list(resnet50(pretrained=True).children())[:-2], nn.Conv2d(2048, d_model, 1))`
        :param num_classes: number of classes
        :param max_detections: maximum number of detections
        :param d_model: dimension of the model
        :param n_head: number of heads in the multi-head attention
        :param num_layers_enc: number of encoder layers
        :param num_layers_dec: number of decoder layers
        """
        super().__init__()
        self.backbone = backbone
        self.position_encoding_x = AbsolutePositionalEncoding2D()
        self.position_encoding_query = AbsolutePositionalEncoding()
        self.transformer = Transformer(num_layers_enc, num_layers_dec, d_model, n_head, None, dropout)
        self.lin_class = nn.Linear(d_model, num_classes + 1)
        self.lin_bbox = nn.Linear(d_model, 4)
        self.query_pos = nn.Parameter(torch.rand(1, max_detections, d_model))
        # Xavier initialization
        for p in self.query_pos.data:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.shape[0]
        x_emb = self.backbone(x)
        x_pos_emb = self.position_encoding_x(x_emb)
        query_pos = ein.repeat(self.query_pos, '1 m d -> b m d', b=batch_size)
        query_pos_emb = self.position_encoding_query(query_pos)
        h = self.transformer(x_emb, query_pos, x_pos_emb, query_pos_emb)
        classes_logits, bbox = self.lin_class(h), self.lin_bbox(h).sigmoid()
        return classes_logits, bbox
    
    def backbone_parameters(self):
        return self.backbone.parameters()
    
    def transformer_parameters(self):
        # Take all the parameters except the backbone parameters
        return [param for name, param in self.named_parameters() if 'backbone' not in name]

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    