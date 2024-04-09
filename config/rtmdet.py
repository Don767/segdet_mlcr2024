from mmengine.config import read_base

with read_base():
    from ._shared_ import *

checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth"

model = dict(
    type="rtmdet",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU", inplace=True),
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=checkpoint),
    ),
    neck=dict(
        type="RTMNeck",
        in_channels=[128, 256, 512],
        num_csp_blocks=1,
        out_channels=128,
        expand_ratio=0.5,
    ),
    bbox_head=dict(
        type="RTMDetHead",
        num_classes=80,
        feat_channels=128,
        in_channels=128,
    ),
    train_cfg=dict(
        assigner=dict(type="DynamicSoftLabelAssigner", topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type="nms", iou_threshold=0.65),
        max_per_img=100,
    ),
)

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
        # dict(
        #     type="WandbVisBackend",
        #     init_kwargs=dict(project="rtmdet", entity="gif-7010"),
        # ),
    ],
)
work_dir = "logs/rtmdet"
