# HRNet-Semantic-Segmentation 용 json 파일을 lst 파일로 변경해주는 프로그램

import os
from pycocotools.coco import COCO
import numpy as np
import json

def cocotolst(dataDir='../input/data', filename='test'):
    annPath = '%s/%s.json' % (dataDir, filename)

    with open(annPath, 'r') as f:
        coco = json.load(f)
    # coco = json.loads(annPath)
    # imgIds = coco.getImgIds()

    # print(type(coco["images"][0]))
    # print(type(imgIds))

    with open(filename+".lst", 'w') as f:
        for img_info in coco["images"]:
            # print(img_info)
            img_file_name = img_info["file_name"]
            img_path = img_file_name
            mask_path = "pngmask/" + img_file_name[:-3] + "png"
            f.write(img_path + " " + mask_path + "\n")
            # f.write(img_path + "\n")
   

if __name__ == "__main__":
    cocotolst()