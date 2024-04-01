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
```
