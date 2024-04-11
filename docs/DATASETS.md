# Dataset preparation

Instructions on how to prepare the datasets used for this project.

## MS COCO

Download the following files from the [official website](https://cocodataset.org/#home).
You can put the data anywhere you want, but you will need to do a symlink to the `data` folder in the root of the project.

To download the data, you can use the following commands:

```shell
# Images
wget http://images.cocodataset.org/zips/train2017.zip \
  http://images.cocodataset.org/zips/val2017.zip \
  http://images.cocodataset.org/zips/test2017.zip
  
# Optional http://images.cocodataset.org/zips/unlabeled2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip \
  http://images.cocodataset.org/annotations/image_info_test2017.zip
  
# Optional http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip
```

Then, you can extract the files and create the symlinks using the following commands:
```shell
# Extract files
for file in *.zip; do
    unzip "$file" &
done && wait

# Create symlink
ln -s /path/to/coco/ data/coco

# For YOLOv7, you need to copy the data
mkdir data/coco/images
cp -r data/coco/train2017 data/coco/images/train2017
cp -r data/coco/val2017 data/coco/images/val2017
# And download the split files
d='./data/' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
wget "$url$f" && unzip -qo "$f" -d "$d" && rm "$f" & # download, unzip, remove in background
```
