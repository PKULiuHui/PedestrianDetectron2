# Train a fast-rcnn model on caltech data
import os
import numpy as np
import json
import cv2

from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def get_caltech_dicts(split):
    json_file = split + '_annos.json'
    with open(json_file) as f:
        imgs_anns = json.load(f)

    for i in range(len(imgs_anns)):
        for j in range(len(imgs_anns[i]['annotations'])):
            imgs_anns[i]['annotations'][j]['bbox_mode'] = BoxMode.XYWH_ABS
        imgs_anns[i]['proposal_bbox_mode'] = BoxMode.XYXY_ABS

    return imgs_anns


for d in ["train", "test"]:
    DatasetCatalog.register("caltech_" + d, lambda d=d: get_caltech_dicts(d))
    MetadataCatalog.get("caltech_" + d).set(thing_classes=["person"])
caltech_metadata = MetadataCatalog.get("caltech_train")

cfg = get_cfg()
cfg.merge_from_file("fast_rcnn.yaml")

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
