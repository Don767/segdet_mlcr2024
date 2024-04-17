#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00
#SBATCH --job-name=yolov7
#SBATCH --output=%x-%j.out

cd ~/segdet_mlcr2024 || exit
docker build -t segdet .

# To debug and get the logs directly in the slurm output

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
  --mount type=bind,source=.,target=/app/ \
  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
  --mount type=bind,source=/dev/shm,target=/dev/shm \
  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 segdet/models/yolov7/yolo_train.py --workers 8 --device $CUDA_VISIBLE_DEVICES --batch-size 64 --data config/coco.yaml --img 640 640 --cfg config/config_yolov7.yaml --weights yolov7_training.pt --name yolov7-base --hyp config/hyp_scratch_yolov7.yaml"

#--workers 8 --device 0 --batch-size 32 --data ../../config/coco.yaml --img 640 640 --cfg ../../config/config_yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-base --hyp ../../config/hyp_scratch_yolov7.yaml

#docker run --gpus all -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES --rm --ipc host \
#  --mount type=bind,source=.,target=/app/ \
#  --mount type=bind,source=$(pwd)/data/coco,target=/app/data/coco \
#  --mount type=bind,source=/dev/shm,target=/dev/shm \
#  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 yolo_train.py --workers 8 --device 0 --batch-size 32 --data config/coco.yaml --img 640 640 --cfg config/config_yolov7.yaml --weights 'yolov7.pt' --name yolov7-base --hyp config/hyp_scratch_yolov7.yaml"
#  segdet bash -c "mim install mmcv==2.1.0 mmdet==3.3.0 && python3 segdet/models/yolov7/yolo_train.py --workers 8 --device 0 --batch-size 32 --data config/coco.yaml --img 640 640 --cfg config/config_yolov7.yaml --weights 'yolov7.pt' --name yolov7-base --hyp config/hyp_scratch_yolov7.yaml"

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
