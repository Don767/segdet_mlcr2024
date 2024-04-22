import argparse
import csv
import torch.cuda
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config
from mmengine.runner import Runner

from tools.loading_utils import check_file, build_model

import os
import sys
import numpy as np
from mmengine.logging import MMLogger
import time
from tabulate import tabulate
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference_model import InferenceModel
from inference_data_preprocessor import InferenceDataPreProcessor


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

def speed_benchmark(batch_size, config_file, weights, imgs_path):
    logger = MMLogger.get_current_instance()
    cfg = Config.fromfile(config_file)
    model_cfg = cfg.get("model")
    infer_model = InferenceModel(
        model_name=model_cfg.pop("type"),
        model_cfg=model_cfg,
        checkpoint=weights,
        half_precision=False,
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
    warmup(infer_model, imgs_to_warmup)
    logger.info("Warmup done.")

    logger.info("Starting the benchmark...")
    times = []
    for batch in batches:
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--conf", help="model config file path")
    parser.add_argument("--weights", help="model config file path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for evaluation")
    args = parser.parse_args()

    return args


def main(args):
    setup_cache_size_limit_of_dynamo()
    config_file = check_file(args.conf)  # check file
    cfg = Config.fromfile(config_file)
    weights = check_file(args.weights)  # check file

    # Create a csv named <config_file>_<weights_file>.csv
    # Strip the path and extension from the config file
    config_file_name = os.path.splitext(os.path.basename(config_file))[0]
    # Strip the path and extension from the weights file
    weights_name = os.path.splitext(os.path.basename(weights))[0]
    with open(f"{config_file_name}_{weights_name}.csv", "w") as f:
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


    gpu = args.gpu
    torch.cuda.set_device(gpu)

    model_cfg = cfg.get("model")
    model = build_model(model_cfg.pop("type"), model_cfg)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size = (param_size + buffer_size) / 1024**2

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
    metrics = runner.test()
    mAP, AP50, AP75, APs, APm, APl = metrics["coco/bbox_mAP"], metrics["coco/bbox_mAP_50"], metrics["coco/bbox_mAP_75"], metrics["coco/bbox_mAP_s"], metrics["coco/bbox_mAP_m"], metrics["coco/bbox_mAP_l"]
    img_path = "./data/coco/val2017/"
    batch_size = 1
    table = speed_benchmark(batch_size, config_file, weights, img_path)
    time_1_mean, time_1_std, time_1_min, time_1_median, time_1_max = table[0][1], table[1][1], table[2][1], table[3][1], table[4][1]
    batch_size = 16
    table = speed_benchmark(batch_size, config_file, weights, img_path)
    time_16_mean, time_16_std, time_16_min, time_16_median, time_16_max = table[0][1], table[1][1], table[2][1], table[3][1], table[4][1]
    batch_size = 32
    table = speed_benchmark(batch_size, config_file, weights, img_path)
    time_32_mean, time_32_std, time_32_min, time_32_median, time_32_max = table[0][1], table[1][1], table[2][1], table[3][1], table[4][1]

    with open(f"{config_file_name}_{weights_name}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([config_file, weights, model_size, mAP, AP50, AP75, APs, APm, APl, 
                         time_1_mean, time_1_std, time_1_min, time_1_median, time_1_max,
                         time_16_mean, time_16_std, time_16_min, time_16_median, time_16_max,
                         time_32_mean, time_32_std, time_32_min, time_32_median, time_32_max])

if __name__ == "__main__":
    main(parse_args())
