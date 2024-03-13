import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.utils import OptMultiConfig
from mmdet.models import CSPLayer


class RTMNeck(BaseModule):
    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        expand_ratio: float = 0.5,
        conv_cfg: bool = None,
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        super().__init__(init_cfg)
        self.norm_cfg = dict(type="SyncBN")
        self.act_cfg = dict(type="SiLU", inplace=True)
        self.in_channels = in_channels

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=False,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=False,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            )

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
        return tuple(outs)
