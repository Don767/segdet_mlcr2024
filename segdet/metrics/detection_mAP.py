import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here

if annType == 'keypoints':
    prefix = 'person_keypoints'
else:
    prefix = 'instances'


#initialize COCO ground truth api
dataDir = '../../coco'
dataType = 'val2017'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt = COCO(annFile)

#initialize COCO detections api
resFile = '%s/results/%s_%s_fake%s100_results.json'
resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt = cocoGt.loadRes(resFile)

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]
imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()