#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00
#SBATCH --job-name=$NAME
#SBATCH --output=%x-%j.out

cd ~/segdet_mlcr2024 || exit
docker build -t segdet .

echo "Running on GPU $CUDA_VISIBLE_DEVICES"
#echo "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 tools/train.py $CONFIG" > $NAME.sh
#chmod +x $NAME.sh

# To debug and get the logs directly in the slurm output
docker run --gpus all --rm --ipc host \
  --mount type=bind,source=.,target=/app/ \
  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 tools/train.py $CONFIG --gpu $CUDA_VISIBLE_DEVICES"

#docker run --gpus \"device=${CUDA_VISIBLE_DEVICES}\" -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -e GPU=$CUDA_VISIBLE_DEVICES --rm --ipc host \
#  --mount type=bind,source=.,target=/app/ \
#  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
#  --mount type=bind,source=/dev/shm,target=/dev/shm \
#  segdet ./$NAME.sh

#container_id=$(
#  docker run --detach --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
#    --mount type=bind,source=.,target=/app/ \
#    --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
#    --mount type=bind,source=/dev/shm,target=/dev/shm \
#    segdet python3 tools/train.py $CONFIG
#)
#
#stop_container() {
#  docker container stop $container_id
#  docker logs $container_id
#}
#
#trap stop_container EXIT
#echo "Container ID: $container_id"
#docker wait $container_id
