from typing import List, Union
import torch
import numpy as np
from mmcv.transforms import Compose
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadImageFromNDArray
from mmdet.structures import DetDataSample
from mmengine.model import ImgDataPreprocessor


class InferenceDataPreProcessor(ImgDataPreprocessor):
    SUPPORTED_FIRST_STEP = [
        LoadImageFromFile,
        LoadImageFromNDArray,
    ]

    def __init__(
        self,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        pipeline: List[dict] = [
            dict(backend_args=None, type=LoadImageFromFile),
        ],
    ):
        self.pipeline = Compose(pipeline)
        self.first_step = self.pipeline.transforms[0]
        super().__init__(mean=mean, std=std, bgr_to_rgb=False)

    def forward(self, raw_data: Union[List[np.ndarray], np.ndarray, List[str], str]):
        if not isinstance(raw_data, list):
            raw_data = [raw_data]
        data = {"inputs": [], "data_samples": []}
        for idx, single_data in enumerate(raw_data):
            single_data = self._data_formatting(single_data, idx)
            self.pipeline(single_data)
            data["inputs"].append(self.preprocess_np_img(single_data.pop("img")))
            data["data_samples"].append(DetDataSample(metainfo=single_data))
        return super().forward(data, training=False)

    def preprocess_np_img(self, img: np.ndarray):
        return torch.tensor(img.transpose(2, 0, 1))

    def _data_formatting(self, data: Union[np.ndarray, str], idx: int = 0):
        if isinstance(self.first_step, LoadImageFromFile):
            assert isinstance(
                data, str
            ), f"The registered first step: LoadImageFromFile requires a string path to the image, but got {type(data)} instead."
            data = dict(img_path=data, img_id=idx)
        elif isinstance(self.first_step, LoadImageFromNDArray):
            assert isinstance(
                data, np.ndarray
            ), f"The registered first step: LoadImageFromNDArray requires a np.ndarray, but got {type(data)} instead."
            data = dict(img=data, img_id=idx)
        else:
            raise NotImplementedError(
                f"First step of the pipeline must be one of {self.SUPPORTED_FIRST_STEP}"
            )
        return data
