import argparse
import copy
import csv
import os
import pathlib
import sys
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.cuda
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine import RUNNERS
from mmengine.config import Config
from mmengine.logging import MMLogger
from mmengine.runner import Runner
from tabulate import tabulate
from tqdm import tqdm

from tools.loading_utils import check_file, build_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_model import InferenceModel, InferenceWrapper
from inference_data_preprocessor import InferenceDataPreProcessor

# Monkey patch the Config constructor to disable _format_dict
_backup_init = Config.__init__


def __init__(self,
             cfg_dict: dict = None,
             cfg_text: Optional[str] = None,
             filename: Optional[Union[str, pathlib.Path]] = None,
             env_variables: Optional[dict] = None,
             format_python_code: bool = False):
    _backup_init(self, cfg_dict, cfg_text, filename, env_variables, format_python_code)


Config.__init__ = __init__


def batchify_lst(lst, batch_size):
    arr = np.array(lst)
    return np.split(arr, range(batch_size, len(arr), batch_size))


def link_batch(img_path, batch):
    return [os.path.join(img_path, img) for img in batch]


def warmup(model, data):
    for _ in range(10):
        model(data)


def log_and_return_metrics(logger, times, title):
    table = [
        ["Mean", np.mean(times)],
        ["Std", np.std(times)],
        ["Min", np.min(times)],
        ["Median", np.median(times)],
        ["Max", np.max(times)],
    ]
    logger.info(
        "\n"
        + tabulate(
            table,
            headers=[title, "Inference Time (s)"],
            tablefmt="pretty",
        )
    )
    return table


def speed_benchmark(batch_size, config_file, weights, imgs_path, is_mmdet, is_fp16=False):
    logger = MMLogger.get_current_instance()
    cfg = Config.fromfile(config_file)

    cfg.work_dir = pathlib.Path('./work_dirs') / pathlib.Path(config_file).stem
    if is_fp16:
        cfg.fp16 = dict(loss_scale='dynamic')

    if is_mmdet:
        cfg.data_preprocessor = dict(
            type="DetDataPreprocessor",
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False,
            batch_augments=None,
        )
        img_size = (640, 640)
        cfg.inference_pipeline = [
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(
                keep_ratio=True,
                scale=img_size,
                type="Resize",
            ),
            dict(
                pad_val=dict(
                    img=(
                        114,
                        114,
                        114,
                    )
                ),
                size=img_size,
                type="Pad",
            ),
        ]
        runner = make_runner(cfg, is_mmdet, weights)
        infer_model = InferenceWrapper(runner.model, half_precision=is_fp16)
    else:
        model_cfg = cfg.get("model")
        infer_model = InferenceModel(
            model_name=model_cfg.pop("type"),
            model_cfg=model_cfg,
            checkpoint=weights,
            half_precision=is_fp16,
        )
    inf_data_preprocessor = InferenceDataPreProcessor(
        mean=cfg.get("data_preprocessor").get("mean"),
        std=cfg.get("data_preprocessor").get("std"),
        pipeline=cfg.get("inference_pipeline"),
    )
    if not os.path.exists(imgs_path):
        raise ValueError(f"The path {imgs_path} does not exist.")
    if not os.path.isdir(imgs_path):
        raise ValueError(f"The path {imgs_path} is not a directory.")
    logger.info(
        f"Splitting the images from {imgs_path} into batches of {batch_size} images."
    )
    img_list = os.listdir(imgs_path)
    if len(img_list) == 0:
        raise ValueError(f"No images found in {imgs_path}.")
    if len(img_list) < batch_size:
        raise ValueError(
            f"Not enough images in {imgs_path} to form a batch of {batch_size} images."
        )
    if len(img_list) % batch_size != 0:
        imgs_to_ignore = len(img_list) % batch_size
        img_list = img_list[:-imgs_to_ignore]
        logger.warning(
            f"The number of images in {imgs_path} is not a multiple of {batch_size}. The last {imgs_to_ignore} images will be ignored."
        )
    batches = batchify_lst(img_list, batch_size)
    logger.info(
        f"The benchmark will be performed on {len(batches)} batches of {batch_size} images."
    )

    logger.info("Warming up the model...")
    imgs_to_warmup = inf_data_preprocessor(link_batch(imgs_path, batches[0]))
    with torch.no_grad():
        warmup(infer_model, imgs_to_warmup)
    logger.info("Warmup done.")

    logger.info(f"Starting the benchmark with batch size {batch_size}...")
    with torch.no_grad():
        times = []
        for batch in tqdm(batches):
            data = inf_data_preprocessor(link_batch(imgs_path, batch))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = infer_model(data)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    logger.info("Benchmark done.")
    table = log_and_return_metrics(logger, times, "Batch Metrics")
    log_and_return_metrics(logger, np.array(times) / batch_size, "Per Image Metrics")
    return table


def model_size_of(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 ** 2


def make_runner(cfg, is_mmdet, weights):
    if is_mmdet:
        cfg.load_from = weights
        if 'runner_type' not in cfg:
            runner = Runner.from_cfg(cfg)
        else:
            runner = RUNNERS.build(cfg)
    else:
        model_cfg = copy.deepcopy(cfg.get("model"))
        model = build_model(model_cfg.pop("type"), model_cfg)

        runner = Runner(
            model=model,
            work_dir=cfg.get("work_dir"),
            test_dataloader=cfg.get("val_dataloader"),
            test_cfg=cfg.get("test_cfg"),
            test_evaluator=cfg.get("test_evaluator"),
            default_hooks=cfg.get("default_hooks"),
            custom_hooks=cfg.get("custom_hooks"),
            data_preprocessor=cfg.get("data_preprocessor"),
            load_from=weights,
            resume=cfg.get("resume", True),
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
    return runner


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--conf", help="model config file path")
    parser.add_argument("--weights", help="model config file path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for evaluation")
    parser.add_argument("--mmdet", action="store_true", help="Test mmdet model instead of custom one")
    parser.add_argument('--fp16', action="store_true", help="Whether to enable half-precision")
    args = parser.parse_args()

    return args


def main(args):
    setup_cache_size_limit_of_dynamo()

    config_file = check_file(args.conf)  # check file
    cfg = Config.fromfile(config_file)
    weights = check_file(args.weights)  # check file
    is_fp16 = args.fp16

    if is_fp16:
        cfg.fp16 = dict(loss_scale='dynamic')

    # Create a csv named <config_file>_<weights_file>.csv
    # Strip the path and extension from the config file
    config_file_name = os.path.splitext(os.path.basename(config_file))[0]
    # Strip the path and extension from the weights file
    weights_name = os.path.splitext(os.path.basename(weights))[0]

    csv_path = pathlib.Path('csv')
    csv_path.mkdir(parents=True, exist_ok=True)
    output_csv = f"{csv_path}/{config_file_name}_{weights_name}.csv"

    if pathlib.Path(output_csv).exists():
        print(f'Skipping for {config_file}, csv already generated...')
        exit(0)

    gpu = args.gpu
    torch.cuda.set_device(gpu)

    cfg.work_dir = pathlib.Path('./work_dirs') / pathlib.Path(config_file).stem
    runner = make_runner(cfg, args.mmdet, weights)

    model_size = model_size_of(runner.model)

    metrics = runner.test()
    print(metrics)
    mAP, AP50, AP75, APs, APm, APl = metrics["coco/bbox_mAP"], metrics["coco/bbox_mAP_50"], metrics["coco/bbox_mAP_75"], \
        metrics["coco/bbox_mAP_s"], metrics["coco/bbox_mAP_m"], metrics["coco/bbox_mAP_l"]
    img_path = "./data/coco/val2017/"

    time_1_mean = time_1_std = time_1_min = time_1_median = time_1_max = time_16_mean = time_16_std = time_16_min = time_16_median = time_16_max = time_32_mean = time_32_std = time_32_min = time_32_median = time_32_max = -1

    try:
        batch_size = 1
        table = speed_benchmark(batch_size, config_file, weights, img_path, args.mmdet, is_fp16)
        time_1_mean, time_1_std, time_1_min, time_1_median, time_1_max = table[0][1], table[1][1], table[2][1], \
            table[3][1], \
            table[4][1]
    except:
        ...
    try:
        batch_size = 16
        table = speed_benchmark(batch_size, config_file, weights, img_path, args.mmdet, is_fp16)
        time_16_mean, time_16_std, time_16_min, time_16_median, time_16_max = table[0][1], table[1][1], table[2][1], \
            table[3][1], table[4][1]
    except:
        ...
    try:
        batch_size = 32
        table = speed_benchmark(batch_size, config_file, weights, img_path, args.mmdet, is_fp16)
        time_32_mean, time_32_std, time_32_min, time_32_median, time_32_max = table[0][1], table[1][1], table[2][1], \
            table[3][1], table[4][1]
    except:
        ...

    with open(output_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["config_file",
                         "weights_file",
                         "model_size (MB)",
                         "mAP",
                         "AP50",
                         "AP75",
                         "APs",
                         "APm",
                         "APl",
                         "Time_1_mean", "Time_1_std", "Time_1_min", "Time_1_median", "Time_1_max",
                         "Time_16_mean", "Time_16_std", "Time_16_min", "Time_16_median", "Time_16_max",
                         "Time_32_mean", "Time_32_std", "Time_32_min", "Time_32_median", "Time_32_max"])
        writer.writerow([config_file, weights, model_size, mAP, AP50, AP75, APs, APm, APl,
                         time_1_mean, time_1_std, time_1_min, time_1_median, time_1_max,
                         time_16_mean, time_16_std, time_16_min, time_16_median, time_16_max,
                         time_32_mean, time_32_std, time_32_min, time_32_median, time_32_max])


if __name__ == "__main__":
    main(parse_args())
