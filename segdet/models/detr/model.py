from typing import Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch import nn
import torchvision

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models import BaseDetector
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.structures import InstanceData

from .detr import DETR, FrozenBatchNorm2d

class Model(BaseDetector):
    def __init__(
                self,
                backbone: ConfigType,
                head: OptConfigType,
                data_preprocessor: OptConfigType = None,
                init_cfg: OptMultiConfig = None,
        ) -> None:
        super().__init__(
            data_preprocessor=MODELS.build(data_preprocessor), init_cfg=init_cfg
        )
        #resnet50 = MODELS.build(backbone)
        num_classes = head.get("num_classes")
        max_detections = head.get("max_detection_points")
        d_model = head.get("d_model")
        n_head = head.get("n_head")
        num_encoder_layers = head.get("num_layers_enc")
        num_decoder_layers = head.get("num_layers_dec")
        dropout = head.get("dropout")
        resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, norm_layer=FrozenBatchNorm2d)
        backbone = nn.Sequential(*list(resnet50.children())[:-2], nn.Conv2d(2048, d_model, 1))
        self.detr = DETR(backbone,
                        num_classes=num_classes,
                        max_detections=max_detections,
                        d_model=d_model,
                        n_head=n_head,
                        num_layers_enc=num_encoder_layers,
                        num_layers_dec=num_decoder_layers,
                        dropout=dropout)
        
    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        return

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        logits, bbox = self.detr(batch_inputs)
        # Convert logits to probabilities
        class_probs = torch.nn.functional.softmax(logits, dim=-1)
        # Get the probability of the class label
        class_probs, class_labels = torch.max(class_probs, dim=-1)
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        # Convert bbox tensor to list
        bbox = bbox.tolist()
        class_probs = class_probs.tolist()
        class_labels = class_labels.tolist()
        pred_instances = []
        for img_metas, class_label, class_prob, bbox in zip(batch_img_metas, class_labels, class_probs, bbox):
            index_to_remove = []
            for i in range(len(class_label)):
                if class_label[i] >= 80:
                    index_to_remove.insert(0,i)
            for i in index_to_remove:
                class_label.pop(i)
                class_prob.pop(i)
                bbox.pop(i)
            
            # Print the len of class_label
            pred_instance = InstanceData(metainfo=img_metas)
            pred_instance.labels = torch.tensor(class_label)
            pred_instance.scores = torch.tensor(class_prob)
            # Multiply every bbox by the image size
            for i in range(len(bbox)):
                bbox[i][0] *= img_metas["img_shape"][1]
                bbox[i][1] *= img_metas["img_shape"][0]
                bbox[i][2] = bbox[i][2]*img_metas["img_shape"][1] + bbox[i][0]
                bbox[i][3] = bbox[i][3]*img_metas["img_shape"][0] + bbox[i][1]
            # Convert bbox to tensor
            bbox = torch.tensor(bbox)
            pred_instance.bboxes = bbox
            pred_instances.append(pred_instance)
        for data_sample, pred_instance in zip(batch_data_samples, pred_instances):
            data_sample.pred_instances = pred_instance
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, *args, **kwargs) -> Tuple[List[Tensor]]:
        return
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        return