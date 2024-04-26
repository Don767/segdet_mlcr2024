from mmengine.config import read_base

with read_base():
    from ._shared_ import *

model = dict(
    type="detr",
    data_preprocessor=data_preprocessor,
    # Resnet50 backbone
    backbone=dict(
        type="ResNet",
        depth=50,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    head=dict(
        num_classes=91,
        max_detection_points=100,
        d_model=256,
        n_head=8,
        num_layers_enc=6,
        num_layers_dec=6,
        dropout=0.1
    ),
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

work_dir = "logs/detr"