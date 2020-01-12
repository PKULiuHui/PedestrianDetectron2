# 先找到明显正例和明显负例集合，再使用训练好的FRCN模型计算正例中心和负例中心

import os
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm

from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import Checkpointer
from models import FeatureTransformationRCNN

preprocess = False
anno_path = 'train_annos.json'
prop_path = 'train_prop.pkl'
cand_path = 'cand_prop.pkl'
res_path = 'center.pkl'
max_neg_per_image = 40


def get_candidate():
    annos = json.load(open(anno_path, 'r'))
    props = pickle.load(open(prop_path, 'rb'))
    cands = {'ids': [], 'boxes': [], 'objectness_logits': []}
    pos_num, neg_num = 0, 0
    for i in tqdm(range(len(annos))):
        assert i == annos[i]['image_id'] and i == props['ids'][i]
        gt_boxes = [x['bbox'] for x in annos[i]['annotations']]
        occl = [x['occl'] for x in annos[i]['annotations']]
        prop_boxes = props['boxes'][i]
        if len(gt_boxes) == 0:  # 全都是明显负例
            cands['ids'].append(i)
            cands['boxes'].append(np.array(props['boxes'][i][:max_neg_per_image]))
            cands['objectness_logits'].append(np.array([0.0] * len(props['boxes'][i][:max_neg_per_image])))
            neg_num += len(props['boxes'][i][:max_neg_per_image])
        else:
            gt_boxes = np.array(gt_boxes)
            gt_boxes[:, 2] += gt_boxes[:, 0]
            gt_boxes[:, 3] += gt_boxes[:, 1]
            ious = calc_ious(np.array(prop_boxes), gt_boxes)
            max_ious = ious.max(axis=1)
            max_idx = ious.argmax(axis=1)
            bboxes, logits = [], []
            cur_neg = 0
            for j in range(len(prop_boxes)):
                if max_ious[j] >= 0.7 and occl[max_idx[j]] == 0:
                    bboxes.append(list(prop_boxes[j]))
                    logits.append(1.0)
                    pos_num += 1
                elif max_ious[j] < 0.1:
                    if cur_neg >= max_neg_per_image:
                        continue
                    bboxes.append(list(prop_boxes[j]))
                    logits.append(0.0)
                    neg_num += 1
                    cur_neg += 1

            cands['ids'].append(i)
            cands['boxes'].append(np.array(bboxes))
            cands['objectness_logits'].append(np.array(logits))
    print(len(cands['boxes']))
    print(pos_num)
    print(neg_num)
    pickle.dump(cands, open(cand_path, 'wb'))
    return


def calc_ious(ex_rois, gt_rois):
    ex_area = (ex_rois[:, 2] - ex_rois[:, 0]) * (ex_rois[:, 3] - ex_rois[:, 1])
    gt_area = (gt_rois[:, 2] - gt_rois[:, 0]) * (gt_rois[:, 3] - gt_rois[:, 1])
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))

    lb = np.maximum(ex_rois[:, 0].reshape((-1, 1)), gt_rois[:, 0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:, 2].reshape((-1, 1)), gt_rois[:, 2].reshape((1, -1)))
    tb = np.maximum(ex_rois[:, 1].reshape((-1, 1)), gt_rois[:, 1].reshape((1, -1)))
    ub = np.minimum(ex_rois[:, 3].reshape((-1, 1)), gt_rois[:, 3].reshape((1, -1)))

    width = np.maximum(rb - lb, 0.)
    height = np.maximum(ub - tb, 0.)
    area_i = width * height
    area_u = area_sum - area_i
    ious = area_i / area_u
    return ious


def get_center():
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
    cfg.merge_from_file("./configs/frcn_attn_0_center.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .0  # set to 0 to achieve smaller miss rate
    val_loader = build_detection_test_loader(cfg, "caltech_train")
    model = build_model(cfg)
    ckpt = Checkpointer(model)
    ckpt.load(os.path.join(cfg.OUTPUT_DIR, "model_0044999.pth"))

    # convert to caltech data eval format
    pos_center, neg_center = .0, .0
    pos_cnt, neg_cnt = .0, .0
    with torch.no_grad():
        model.eval()
        for j, inputs in enumerate(tqdm(val_loader)):
            box_features, box_scores, box_labels = model.predict(inputs)
            box_features = box_features.view(box_features.size(0), -1)
            for i in range(len(box_features)):
                if box_labels[i] < 0.01 and box_scores[i] < 0.01:
                    neg_center += box_features[i]
                    neg_cnt += 1
                elif box_labels[i] > 0.9 and box_scores[i] >= 0.9:
                    pos_center += box_features[i]
                    pos_cnt += 1
    pos_center /= pos_cnt
    neg_center /= neg_cnt
    pos_center, neg_center = np.array(pos_center.tolist()), np.array(neg_center.tolist())
    print(pos_center.shape)
    print(neg_center.shape)
    print(cos_sim(pos_center, neg_center))
    pickle.dump({'pos_center': pos_center, 'neg_center': neg_center}, open(res_path, 'wb'))
    return


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    return cos


if __name__ == '__main__':
    if preprocess:
        get_candidate()
    else:
        get_center()
