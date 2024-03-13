# Training

Training parameters are provided via YAML configuration files. See `configs/jobs` for examples.
A guide on how to write your own configuration file can be found [here](CONFIGURATION.md).

## Local training

```shell
python train.py --config <path/to/config>
```

## Docker training

```shell
# Build docker image
docker build -t segdet .

# Run docker image
CONFIG=path/to/config> # for example `config/segsdet.yaml`
CUDA_VISIBLE_DEVICES=all # or `0,1` for specific GPUs, will be automatically set by SLURM

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
  --mount type=bind,source=.,target=/code/ \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  segdet python3 main.py path/to/config
```

## Train with sjm to manage SLURM jobs

Install [willGuimont/sjm](https://github.com/willGuimont/sjm).

```shell
sjm pull exx segdet_mlcr2024
sjm run exx slurm_train.sh NAME=experiment_name CONFIG=configs/training/config_path.yml
```
