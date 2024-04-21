import copy
from typing import List, Tuple, Union

import torch.cuda
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
        backbone.embeddings = ViTMAEEmbeddings(conf)
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

    # def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
    #     x = self.extract_feat(batch_inputs)
    #     return self.bbox_head.loss(x, batch_data_samples)
    #
    # def predict(
    #         self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    # ) -> SampleList:
    #     x = self.extract_feat(batch_inputs)
    #     results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
    #     return self.add_pred_to_datasample(batch_data_samples, results_list)
    #
    # def _forward(self, batch_inputs: Tensor, *args, **kwargs) -> Tuple[List[Tensor]]:
    #     x = self.extract_feat(batch_inputs)
    #     return self.bbox_head(x)
    #
    # def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
    #     x = self.backbone(batch_inputs)
    #     return self.neck(x)
    #
    # def _load_from_state_dict(
    #         self,
    #         state_dict: dict,
    #         prefix: str,
    #         local_metadata: dict,
    #         strict: bool,
    #         missing_keys: Union[List[str], str],
    #         unexpected_keys: Union[List[str], str],
    #         error_msgs: Union[List[str], str],
    # ) -> None:
    #     bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
    #     bbox_head_keys = [
    #         k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
    #     ]
    #     rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
    #     rpn_head_keys = [k for k in state_dict.keys() if k.startswith(rpn_head_prefix)]
    #     if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
    #         for rpn_head_key in rpn_head_keys:
    #             bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix):]
    #             state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
    #     super()._load_from_state_dict(
    #         state_dict,
    #         prefix,
    #         local_metadata,
    #         strict,
    #         missing_keys,
    #         unexpected_keys,
    #         error_msgs,
    #     )
