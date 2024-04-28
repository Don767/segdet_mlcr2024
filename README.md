# SEGDet - ML Reproductibility Challenge 2024

## Installation

Please refer to [docs/INSTALLATION.md](docs/INSTALLATION.md) for more details on how to install the required packages and dependencies.

Details on how to download the datasets used in this project can be found in [docs/DATASETS.md](docs/DATASETS.md).

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

Some weights are available in the `weights/` folder.
Additional weights are available [here](https://drive.google.com/drive/folders/1wBi9-aDgOgOUS0r54mH29C57rr5hZXWR?usp=sharing)

We also provide utilities to automatically benchmark models on your hardware.

```
# --gpu is a list of comma-separated values specifying the GPUs used for benchmarking
python3 run_all.py --gpus 0
```

## Results

We detail the performances of our models on [Image Classification on MS COCO 2017](https://cocodataset.org/#home) in `results/`.
We provide an example Jupyter Notebook `table_extractor.ipynb` showing how to parse our results.

<!---| Model name         | Top 1 Accuracy  | Top 5 Accuracy |-->
<!---| ------------------ |---------------- | -------------- |-->
<!---| My awesome model   |     85%         |      95%       |-->

MORE COMING SOON

## Citation

If you use this code in your research, please cite the following:

```
@inproceedings{fastDet2024,
  title={Replication Study and Benchmarking of Real-Time Object Detection Models},
  author={Pierre-Luc Asselin, Vincent Coulombe, William Guimont-Martin, William Larrivée-Hardy},
  year={2024}
}
```

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
