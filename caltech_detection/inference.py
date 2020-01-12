# inference on caltech data

import os
import json
import torch
from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import Checkpointer
from models import AttentionRCNN
from models import FeatureTransformationRCNN


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
cfg.merge_from_file("./configs/frcn_dt.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .0   # set to 0 to achieve smaller miss rate
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("caltech_test", cfg, False, output_dir='./output/')
val_loader = build_detection_test_loader(cfg, "caltech_test")
model = build_model(cfg)
ckpt = Checkpointer(model)
ckpt.load(os.path.join(cfg.OUTPUT_DIR, "model_0049999.pth"))
# inference_on_dataset(model, val_loader, evaluator)  # compute map value

# convert to caltech data eval format
res = {}
with torch.no_grad():
    model.eval()
    for inputs in tqdm(val_loader):
        outputs = model(inputs)
        for i, o in zip(inputs, outputs):
            fn = i['file_name']
            idx = fn.rfind('/') + 1
            fn = fn[idx:-4].split('_')
            sid, vid, frame = fn[0], fn[1], int(fn[2]) + 1
            if sid not in res:
                res[sid] = {}
            if vid not in res[sid]:
                res[sid][vid] = []
            scores = o['instances'].scores.tolist()
            bboxes = o['instances'].pred_boxes.tensor.tolist()
            for b, s in zip(bboxes, scores):
                cur_res = (frame, b[0], b[1], b[2] - b[0], b[3] - b[1], s)
                res[sid][vid].append(cur_res)

out_dir = '/envs/shareB/liuhui/Detection/caltech/data/res/frcn_dt/'
if os.path.exists(out_dir):
    os.system('rm -rf %s' % out_dir)
os.mkdir(out_dir)
for s in res:
    os.mkdir(os.path.join(out_dir, s))
    for v in res[s]:
        with open(os.path.join(out_dir, s, v + '.txt'), 'w') as f:
            for r in res[s][v]:
                f.write('%d %.2f %.2f %.2f %.2f %.6f\n' % (r[0], r[1], r[2], r[3], r[4], r[5]))
