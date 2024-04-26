#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00
#SBATCH --job-name=run_all
#SBATCH --output=%x-%j.out

cd ~/segdet_mlcr2024 || exit
docker build -t segdet .

echo "Running on GPU $CUDA_VISIBLE_DEVICES"

# To debug and get the logs directly in the slurm output
docker run --gpus all --rm --ipc host \
  --mount type=bind,source=.,target=/app/ \
  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 run_all.py --gpus $CUDA_VISIBLE_DEVICES"

#docker run --gpus all -it --rm --ipc host \
#  --mount type=bind,source=.,target=/app/ \
#  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
#  --mount type=bind,source=/dev/shm,target=/dev/shm \
#  segdet bash
