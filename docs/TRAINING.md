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
export CONFIG=path/to/config> # for example `config/rtmdet.py`
export CUDA_VISIBLE_DEVICES=0 # or `0,1` for specific GPUs, will be automatically set by SLURM

# mm-based dependencies need to be installed while running the Docker image as they require a CUDA runtime
docker run --gpus all --rm -it --ipc host \
  --mount type=bind,source=.,target=/app/ \
  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 tools/train.py $CONFIG --gpu $CUDA_VISIBLE_DEVICES"
```

## Podman training
```shell
# Build the image
buildah build -t segdet .

# Run the image
export CONFIG=path/to/config> # for example `config/rtmdet.py`
export CUDA_VISIBLE_DEVICES=0 # or `0,1` for specific GPUs, will be automatically set by SLURM

# mm-based dependencies need to be installed while running the Docker image as they require a CUDA runtime
podman run --gpus all --rm -it --ipc host \
  -v .:/app/ \
  -v /app/data \
  -v ./data/coco/:/app/data/coco \
  -v /dev/shm:/dev/shm \
  segdet:latest  bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 tools/train.py $CONFIG --gpu $CUDA_VISIBLE_DEVICES"
```

## Train with sjm to manage SLURM jobs

Install [willGuimont/sjm](https://github.com/willGuimont/sjm).

```shell
sjm pull exx segdet_mlcr2024
sjm run exx scripts/slurm_train.sh NAME=experiment_name CONFIG=configs/training/config_path.yml
sjm ps exx
```

