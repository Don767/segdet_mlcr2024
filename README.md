# SEGDet - ML Reproductibility Challenge 2024

## Installation

Python packages required (can be installed via pip or conda):

- cython==3.0.9
- matplotlib==3.8.3
- numpy>=1.26.0
- opencv-python>=4.1.1
- openmim==0.3.9
- Pillow>=7.1.2
- protobuf<4.21.3
- psutil>=5.9.8
- pycocotools==2.0.7
- PyYAML>=5.3.1
- requests>=2.23.0
- setuptools>=69.2.0
- scipy>=1.4.1
- tensorboard==2.16.2
- torch
- torchvision
- tqdm>=4.41.0
- wandb==0.16.4
- git+https://github.com/willGuimont/pipeline
- transformers
- einops
- dill
- requests

Please refer to [docs/INSTALLATION.md](docs/INSTALLATION.md) for more details on how to install the required packages and dependencies.

Details on how to download the datasets used in this project can be found in [docs/DATASETS.md](docs/DATASETS.md).

If desired, use the same seed used for our results below:

```python
# Where seed = 1 at base.
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

### Docker

We provide a `Dockerfile` at the root of this projet to allow easy reproduction of our results.
This is the recommended way of running this project.
We detail instructions on how to use Docker in [docs/TRAINING.md](docs/TRAINING.md)

Scripts in `scripts/` shows how to run our model on a SLURM cluster.

## Training

To train the model in the paper, run this command:

```shell
python tools/train.py --config <path/to/config>
```

Please refer to [docs/TRAINING.md](docs/TRAINING.md) for more details on how train our models with our pipeline (including Docker/SLURM integration).

## Evaluation

To test our models on [MS COCO Val 2017](https://cocodataset.org/#download), run:

```shell
python tools/metrics.py --conf <path/to/config> --weights <path/to/weights> \
  --gpu <GPU ID> --mmdet <True if testing MMDet presets, False if testing custom model> \
  --fp16 <True to enable half-precision>
```

## Pre-trained Models

You can download pretrained models from the `weights/` folder.

## Results

Our models achieves the following performance on :

### [Image Classification on MS COCO 2017](https://cocodataset.org/#home)

Our raw results for each GPU are available in `results/`.

<!---| Model name         | Top 1 Accuracy  | Top 5 Accuracy |-->
<!---| ------------------ |---------------- | -------------- |-->
<!---| My awesome model   |     85%         |      95%       |-->

COMING SOON

## Citation

If you use this code in your research, please cite the following:

@inproceedings{fastDet2024,
  title={RTMDet MLRP2024},
  author={Pierre-Luc Asselin, Vincent Coulombe, William Guimont-Martin, William Larrivée-Hardy},
  year={2024}
}

## Contributing

We want to make contributing to this project as easy and transparent as possible.

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

### License
By contributing, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
