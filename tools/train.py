import argparse

import torch.cuda
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
from mmengine.runner import Runner

from tools.loading_utils import check_file, build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="model config file path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for evaluation")
    args = parser.parse_args()

    return args


def main(args):
    setup_cache_size_limit_of_dynamo()
    config_file = check_file(args.config)  # check file
    cfg = Config.fromfile(config_file)

    gpu = args.gpu
    torch.cuda.set_device(gpu)

    model_cfg = cfg.get("model")
    runner = Runner(
        model=build_model(model_cfg.pop("type"), model_cfg),
        work_dir=cfg.get("work_dir"),
        train_dataloader=cfg.get("train_dataloader"),
        val_dataloader=cfg.get("val_dataloader"),
        test_dataloader=cfg.get("val_dataloader"),
        train_cfg=cfg.get("train_cfg"),
        val_cfg=cfg.get("val_cfg"),
        test_cfg=cfg.get("test_cfg"),
        auto_scale_lr=cfg.get("auto_scale_lr"),
        optim_wrapper=cfg.get("optim_wrapper"),
        param_scheduler=cfg.get("param_scheduler"),
        val_evaluator=cfg.get("val_evaluator"),
        test_evaluator=cfg.get("val_evaluator"),
        default_hooks=cfg.get("default_hooks"),
        custom_hooks=cfg.get("custom_hooks"),
        data_preprocessor=cfg.get("data_preprocessor"),
        load_from=cfg.get("load_from"),
        resume=cfg.get("resume", False),
        launcher=cfg.get("launcher", "none"),
        env_cfg=cfg.get("env_cfg", dict(dist_cfg=dict(backend="nccl"))),
        log_processor=cfg.get("log_processor"),
        log_level=cfg.get("log_level", "INFO"),
        visualizer=cfg.get("visualizer"),
        default_scope=cfg.get("default_scope", "mmengine"),
        randomness=cfg.get("randomness", dict(seed=None)),
        experiment_name=cfg.get("experiment_name"),
        cfg=cfg,
    )
    runner.train()


if __name__ == "__main__":
    main(parse_args())
