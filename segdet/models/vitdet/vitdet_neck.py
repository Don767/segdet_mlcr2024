from mmdet.utils import OptConfigType
from torch import nn, Tensor
import einops as ein

from segdet.models.vitdet.utils import PrintShape


class VitDetNeck(nn.Module):
    def __init__(self, config: OptConfigType):
        super(VitDetNeck, self).__init__()
        self.config = config
        self.neck = nn.ModuleList()
        self.scale_factors = config.scale_factors
        self.in_features = config.in_features
        self.out_features = config.out_features

        for i, scale_factor in enumerate(self.scale_factors):
            self.neck.append(self._build_neck_block(scale_factor))

    def _build_neck_block(self, scale_factor):
        if scale_factor == 4.0:
            layers = [
                nn.ConvTranspose2d(
                    in_channels=self.in_features,
                    out_channels=self.out_features // 2,  # WGM: not specified in the paper
                    kernel_size=2,
                    stride=2,
                ),
                nn.LayerNorm([self.out_features // 2, 64, 64]),
                nn.GELU(),
                nn.ConvTranspose2d(
                    in_channels=self.out_features // 2,
                    out_channels=self.out_features // 4,  # WGM: not specified in the paper
                    kernel_size=2,
                    stride=2,
                ),
            ]
            out = self.out_features // 4
            spatial_size = 128
        elif scale_factor == 2.0:
            layers = [nn.ConvTranspose2d(
                in_channels=self.in_features,
                out_channels=self.out_features // 2,  # WGM: not specified in the paper
                kernel_size=2,
                stride=2,
            )]
            out = self.out_features // 2
            spatial_size = 64
        elif scale_factor == 1.0:
            layers = []
            out = self.in_features
            spatial_size = 32
        elif scale_factor == 0.5:
            layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            out = self.in_features
            spatial_size = 16
        else:
            raise ValueError(f"Invalid scale factor: {scale_factor}")

        layers.extend([
            nn.Conv2d(
                in_channels=out,
                out_channels=self.out_features,
                kernel_size=1,
            ),
            nn.LayerNorm([self.out_features, spatial_size, spatial_size]),
            nn.Conv2d(
                in_channels=self.out_features,
                out_channels=self.out_features,
                kernel_size=3,
                padding=1,
            ),
            nn.LayerNorm([self.out_features, spatial_size, spatial_size]),
        ])

        return nn.Sequential(*layers)

    def forward(self, batch_inputs: Tensor, *args, **kwargs):
        batch_inputs = batch_inputs.last_hidden_state
        # remove first cls token and reshape to 2D (undo .flatten(2).transpose(1, 2))
        b, n, c = batch_inputs.shape
        input = ein.rearrange(batch_inputs[:, 1:], 'b (h w) c -> b c h w', h=32)
        out = []
        for i, n in enumerate(self.neck):
            x = n(input)
            out.append(x)
        return out
