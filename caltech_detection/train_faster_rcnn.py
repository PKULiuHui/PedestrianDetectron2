# Train a faster-rcnn model on caltech data, use default trainer
import os
import json

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


# register caltech dataset
for d in ["train", "test"]:
    DatasetCatalog.register("caltech_" + d, lambda d=d: get_caltech_dicts(d))
    MetadataCatalog.get("caltech_" + d).set(thing_classes=["person"])
caltech_metadata = MetadataCatalog.get("caltech_train")

cfg = get_cfg()
cfg.merge_from_file("./configs/faster_rcnn.yaml")
# Train begin with coco-detection model. If you want to train from scratch, comment it out.
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
