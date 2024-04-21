import math
import PIL
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision import tv_tensors
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

import einops as ein

from hungarian_matcher import HungarianMatcher
from transformer import Transformer
from absolute_positional_encoding import *
import boxes as box_ops

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
DISPLAY_DATA = False
NUMBER_OF_BATCHES_TO_SHOW = 5
GLOBAL_STEP = 0

transform_train = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(size=(480, 480), pad_if_needed=True),
    v2.Resize(size=(480, 480)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transform_val = v2.Compose([
    v2.Resize(size=(480, 480)),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Create a collate function to pad the images and the targets to the biggest image in the batch
def collate_fn_train(batch):
    img_list = []
    target_list = []
    bbox_list = []
    for i, (img, targets) in enumerate(batch):
        list_of_target_classes = []
        list_of_target_bbox = []
        if(len(targets) == 0):
            list_of_target_bbox = torch.empty((0, 4))
        for target in targets:
            list_of_target_classes.append(target['category_id']-1)
            bbox = target['bbox'].copy()
            list_of_target_bbox.append(bbox)
        boxes = tv_tensors.BoundingBoxes(list_of_target_bbox, format='XYWH', canvas_size=img.shape[-2:])
        out_img, out_boxes = transform_train(img, boxes)
        # Normalize the bounding boxes
        for j in range(len(out_boxes)):
            out_boxes[j][0] /= out_img.shape[2]
            out_boxes[j][1] /= out_img.shape[1]
            out_boxes[j][2] /= out_img.shape[2]
            out_boxes[j][3] /= out_img.shape[1]
        list_of_target_bbox = out_boxes
        img_list.append(out_img)
        list_of_target_classes = torch.tensor(list_of_target_classes, dtype=torch.int64)
        list_of_target_bbox = list_of_target_bbox.reshape(len(list_of_target_bbox), 4)
        target_list.append(list_of_target_classes)
        bbox_list.append(list_of_target_bbox)
    return (torch.utils.data._utils.collate.default_collate(img_list), (target_list, bbox_list))

# Create a collate function to pad the images and the targets to the biggest image in the batch
def collate_fn_val(batch):
    img_list = []
    target_list = []
    bbox_list = []
    for i, (img, targets) in enumerate(batch):
        list_of_target_classes = []
        list_of_target_bbox = []
        if(len(targets) == 0):
            list_of_target_bbox = torch.empty((0, 4))
        for target in targets:
            list_of_target_classes.append(target['category_id'])
            bbox = target['bbox'].copy()
            list_of_target_bbox.append(bbox)
        boxes = tv_tensors.BoundingBoxes(list_of_target_bbox, format='XYWH', canvas_size=img.shape[-2:])
        out_img, out_boxes = transform_val(img, boxes)
        # Normalize the bounding boxes
        for j in range(len(out_boxes)):
            out_boxes[j][0] /= out_img.shape[2]
            out_boxes[j][1] /= out_img.shape[1]
            out_boxes[j][2] /= out_img.shape[2]
            out_boxes[j][3] /= out_img.shape[1]
        list_of_target_bbox = out_boxes
        img_list.append(out_img)
        list_of_target_classes = torch.tensor(list_of_target_classes, dtype=torch.int64)
        list_of_target_bbox = list_of_target_bbox.reshape(len(list_of_target_bbox), 4)
        target_list.append(list_of_target_classes)
        bbox_list.append(list_of_target_bbox)
    return (torch.utils.data._utils.collate.default_collate(img_list), (target_list, bbox_list))

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

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
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

class DETR(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, max_detections: int, d_model: int, n_head: int,
                 num_layers_enc: int, num_layers_dec: int, dropout: float = 0.1, tb_logger : pt.TensorBoardLogger = None):
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
        self.tensorboard = tb_logger
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

    def log_image(self, x, global_step):
        classes_logits, bbox =self.forward(x)
        # Convert the image to numpy
        img = x[0].permute(1, 2, 0).cpu().numpy()
        # Convert img to Umat
        img = cv2.UMat(img)
        classes = classes_logits.argmax(-1)
        # Display the bounding boxes on the image
        for i in range(len(classes[0])):
            if(classes[0][i].item() == 91):
                continue
            bbox_x = bbox[0][i][0] * x.shape[3]
            bbox_y = bbox[0][i][1] * x.shape[2]
            bbox_h = bbox[0][i][2] * x.shape[3]
            bbox_w = bbox[0][i][3] * x.shape[2]
            img = cv2.rectangle(img, (int(bbox_x.item()), int(bbox_y.item())), (int(bbox_x.item()) + int(bbox_w.item()), int(bbox_y.item()) + int(bbox_h.item())), (255, 0, 0), 2)
        # Convert the image to tensor
        img = torch.tensor(img.get()).permute(2, 0, 1)
        tb_logger.writer.add_image("pred", img, global_step=0)
    
    def backbone_parameters(self):
        return self.backbone.parameters()
    
    def transformer_parameters(self):
        # Take all the parameters except the backbone parameters
        return [param for name, param in self.named_parameters() if 'backbone' not in name]

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    


if __name__ == '__main__':
    import pathlib

    # Training parameters
    epoch = 300
    batch_size = 64
    learning_rate_transformer = 1e-4
    learning_rate_backbone = 1e-5
    train_split = 0.8
    val_split = 0.1
    device = 'cuda' if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else 'cpu')

    # Model parameters
    d_model = 256
    n_head = 8
    n_latent = 128
    n_layer = 6
    n_output = 128
    out_dim = 512
    num_classes = 91
    max_detections = 100
    dropout = 0.1
    cost_class = 1
    cost_bbox = 1
    cost_giou = 1
    pooling = 'mean'
    num_workers = 10

    # Data
    train_dataset = torchvision.datasets.CocoDetection(root='/home/wilah/datasets/train2017',
                                                        annFile='/home/wilah/datasets/annotations/instances_train2017.json',
                                                       transform=transforms.ToTensor())
    valid_dataset = torchvision.datasets.CocoDetection(root='/home/wilah/datasets/val2017', 
                                                       annFile='/home/wilah/datasets/annotations/instances_val2017.json',
                                                       transform=transforms.ToTensor())
    
    # Create subsets of the dataset with only 1 image
    #train_dataset = Subset(train_dataset, [0])
    #valid_dataset = Subset(valid_dataset, [0])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_train)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_val)

    writer = SummaryWriter('runs')
    tb_logger = pt.TensorBoardLogger(writer)

    # Model and optimizer
    resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
    backbone = nn.Sequential(*list(resnet50.children())[:-2], nn.Conv2d(2048, d_model, 1))
    model = DETR(backbone, num_classes, max_detections, d_model, n_head, n_layer, n_layer, dropout, tb_logger)
    optimizer = torch.optim.AdamW(
        [
            {"params" : model.backbone_parameters(), "lr" : learning_rate_backbone},
            {"params" : model.transformer_parameters(), "lr" : learning_rate_transformer, "weight_decay":1e-4}
        ],
        lr=learning_rate_transformer,
    )
    #model.freeze_backbone()
    matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    eos_coef = 0.1
    losses = ['labels', 'boxes', 'cardinality']
    model_pt = pt.Model(model, optimizer, Criterion(num_classes, matcher, weight_dict, eos_coef, losses))
    model_pt.to(device)

    # Display a data from the training set with bounding boxes
    if(DISPLAY_DATA):
        for k in range(NUMBER_OF_BATCHES_TO_SHOW):
            batch = next(iter(valid_loader))
            img_list = batch[0]
            target_list = batch[1][0]
            bbox_list = batch[1][1]
            for idx in range(len(img_list)):
                img = img_list[idx]
                target_classes = target_list[idx]
                target_bbox = bbox_list[idx]
                fig, ax = plt.subplots(1)
                ax.imshow(img.permute(1, 2, 0))
                for j in range(len(target_classes)):
                    bbox = target_bbox[j]
                    rect = patches.Rectangle((bbox[0] * img.shape[2], bbox[1] * img.shape[1]), bbox[2] * img.shape[2], bbox[3] * img.shape[1], linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                plt.show()
        # Wait for the user to close the windows
        input("Press Enter to continue...")

    # Training
    pathlib.Path('logs').mkdir(parents=True, exist_ok=True)
    history = model_pt.fit_generator(train_loader, None, epochs=epoch, callbacks=[
        pt.ModelCheckpoint('logs/detr_best_epoch_{epoch}.ckpt', monitor='loss', mode='min',
                           save_best_only=True,
                           keep_only_last_best=True, restore_best=False, verbose=True,
                           temporary_filename='best_epoch.ckpt.tmp'),
        pt.ClipNorm(model.parameters(), max_norm=0.1, norm_type=2.0),
        pt.StepLR(step_size=200, gamma=0.1),
        tb_logger
    ])
    
    # Test
    #test_loss, test_acc = model.evaluate_generator(test_loader)
    #print('test_loss: {:.4f} test_acc: {:.2f}'.format(test_loss, test_acc))
