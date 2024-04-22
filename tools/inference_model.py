import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint

from loading_utils import build_model


class InferenceModel(nn.Module):
    def __init__(
            self,
            model_name: str,
            model_cfg: Config,
            checkpoint: str = None,
            device: str = "cuda",
            half_precision: bool = True,
            **kwargs,
    ) -> None:
        super(InferenceModel, self).__init__()
        self.model_cfg = model_cfg
        self.model = build_model(model=model_name, model_cfg=self.model_cfg, **kwargs)
        self.checkpoint = checkpoint
        if self.checkpoint is None:
            logger = MMLogger.get_current_instance()
            logger.warning(
                "No checkpoint to load. The performances might be lower than expected."
            )
        else:
            load_checkpoint(self.model, self.checkpoint, map_location="cpu")
        self.device = device
        self.half_precision = half_precision
        self.model.cfg = self.model_cfg
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, data: str):
        with torch.cuda.amp.autocast(enabled=self.half_precision):
            inputs = data["inputs"].to(self.device)
            outputs = self.model.predict(inputs, data["data_samples"])
        return outputs


class InferenceWrapper(nn.Module):
    def __init__(self, model, half_precision: bool = True):
        super().__init__()
        self.model = model
        self.half_precision = half_precision

    def forward(self, data: str):
        with torch.cuda.amp.autocast(enabled=self.half_precision):
            inputs = data["inputs"].to('cuda')
            outputs = self.model.predict(inputs, data["data_samples"])
        return outputs
