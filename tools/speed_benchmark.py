import os
import sys
import numpy as np
from mmengine.config import Config
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


def log_metrics(logger, times, title):
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


def main(args):
    logger = MMLogger.get_current_instance()
    cfg = Config.fromfile(args.get("config"))
    model_cfg = cfg.get("model")
    infer_model = InferenceModel(
        model_name=model_cfg.pop("type"),
        model_cfg=model_cfg,
        checkpoint=args.get("checkpoint"),
        half_precision=args.get("half_precision", False),
    )
    inf_data_preprocessor = InferenceDataPreProcessor(
        mean=cfg.get("data_preprocessor").get("mean"),
        std=cfg.get("data_preprocessor").get("std"),
        pipeline=cfg.get("inference_pipeline"),
    )
    imgs_path = args.get("images_path")
    if not os.path.exists(imgs_path):
        raise ValueError(f"The path {imgs_path} does not exist.")
    if not os.path.isdir(imgs_path):
        raise ValueError(f"The path {imgs_path} is not a directory.")
    batch_size = args.get("batch_size")
    if not isinstance(batch_size, int):
        raise ValueError(f"The batch size must be an integer, but got {batch_size}.")
    if batch_size <= 0:
        raise ValueError(
            f"The batch size must be a positive integer, but got {batch_size}."
        )
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

    log_metrics(logger, times, "Batch Metrics")
    log_metrics(logger, np.array(times) / batch_size, "Per Image Metrics")


if __name__ == "__main__":
    # TODO : Add argparse
    main(
        {
            "config": "../config/rtmdet.py",
            "checkpoint": None,
            "images_path": "../coco/val2017/",
            "batch_size": 15,
            "half_precision": False,
        }
    )
