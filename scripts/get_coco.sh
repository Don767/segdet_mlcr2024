#!/bin/bash
# COCO 2017 dataset http://cocodataset.org

set -ex

# Download/unzip labels
d='./data/' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels-segments.zip' # or 'coco2017labels.zip', 68 MB
echo 'Downloading' $url$f ' ...'
wget "$url$f" && unzip -qo "$f" -d "$d" && rm "$f" & # download, unzip, remove in background

# Download/unzip images
d='./data/images' # unzip directory
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
f2='val2017.zip'   # 1G, 5k images
f3='test2017.zip'  # 7G, 41k images (optional)
for f in $f1 $f2 $f3; do
  echo 'Downloading' $url$f '...'
  wget "$url$f" && unzip -qo "$f" -d "$d" && rm $f & # download, unzip, remove in background
done
wait # finish background tasks