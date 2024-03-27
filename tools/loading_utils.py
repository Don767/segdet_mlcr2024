import glob
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib

from mmengine.config import Config


def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == "":
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)  # find file
        assert len(files), f"File Not Found: {file}"  # assert file was found
        assert (
            len(files) == 1
        ), f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def build_model(model: str, model_cfg: Config, **kwargs):
    """Return a nn.Module object from the model name and config
    assuming the model is in the models directory and respect the proposed format.

    Args:
        model (str): The name of the folder containing the model
        model_cfg (mmengine.config.Config): The mmengine.config.Config of the model

    Returns:
        nn.Module: The loaded model
    """
    module = importlib.import_module(f"segdet.models.{model}.model")
    return module.Model(**model_cfg, **kwargs)
