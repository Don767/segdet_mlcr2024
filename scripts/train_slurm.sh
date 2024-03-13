#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=4-00:00
#SBATCH --job-name=$NAME
#SBATCH --output=%x-%j.out

cd ~/segdet_mlcr2024
docker build -t segdet .
container_id=$(
  docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --detach --rm --ipc host \
    --mount type=bind,source=.,target=/code/ \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    segdet python3 main.py $CONFIG
)

stop_container() {
  docker container stop $container_id
  docker logs $container_id
}

trap stop_container EXIT
echo "Container ID: $container_id"
docker wait $container_id
