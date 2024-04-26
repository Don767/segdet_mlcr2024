import copy
from typing import List, Tuple, Union

import torch.cuda
import torch.nn.functional as F
import torchvision.models.vision_transformer
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor
from transformers import ViTMAEModel, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAEEmbeddings

from segdet.models.vitdet.custom_two_stage_detector import CustomTwoStageDetector
from segdet.models.vitdet.vitdet_neck import VitDetNeck
from segdet.scheduler.linear_warmup_scheduler import LinearWarmupScheduler

LinearWarmupScheduler

pretrained_vit = {
    "vit_b_16": torchvision.models.vit_b_16,
    "vit_b_32": torchvision.models.vit_b_32,
    "vit_l_16": torchvision.models.vit_l_16,
    "vit_l_32": torchvision.models.vit_l_32,
    "vit_h_14": torchvision.models.vit_h_14,
}

pretrained_vit_weights = {
    "vit_b_16": torchvision.models.vision_transformer.ViT_B_16_Weights,
    "vit_b_32": torchvision.models.vision_transformer.ViT_B_32_Weights,
    "vit_l_16": torchvision.models.vision_transformer.ViT_L_16_Weights,
    "vit_l_32": torchvision.models.vision_transformer.ViT_L_32_Weights,
    "vit_h_14": torchvision.models.vision_transformer.ViT_H_14_Weights,
}


class Model(CustomTwoStageDetector):
    def __init__(
            self,
            image_size: int,
            backbone: ConfigType,
            neck: OptConfigType,
            rpn_head: OptConfigType,
            roi_head: OptConfigType,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptMultiConfig = None,
    ) -> None:
        backbone = ViTMAEModel.from_pretrained(**backbone)
        conf = copy.deepcopy(backbone.config)
        conf.image_size = image_size
        # backbone.embeddings = ViTMAEEmbeddings(conf)
        neck = VitDetNeck(neck)
        super().__init__(
            data_preprocessor=MODELS.build(data_preprocessor), init_cfg=init_cfg,
            backbone=backbone, neck=neck, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg, test_cfg=test_cfg
        )
        # TODO apply changes made by the others on ViT
        # self.backbone = pretrained_vit[backbone["type"]](pretrained=backbone["type"])
        # outputs = model(**inputs)
        # last_hidden_states = outputs.last_hidden_state
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
