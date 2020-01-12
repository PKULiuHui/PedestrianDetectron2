# 从结果里面挑选例子可视化，主要比较frcn和frcn_attn_0的结果

import os
import numpy as np
import cv2
from copy import deepcopy

caltech_img_path = '/envs/shareB/liuhui/Detection/caltech/data/images/'
gt_path = '/envs/shareB/liuhui/Detection/caltech/data/res/gt/'
frcn_path = '/envs/shareB/liuhui/Detection/caltech/data/res/frcn/'
frcn_a0_path = '/envs/shareB/liuhui/Detection/caltech/data/res/frcn_attn_0/'
frcn_dt_path = '/envs/shareB/liuhui/Detection/caltech/data/res/frcn_dt/'
vis_path = 'vis/'
if os.path.exists(vis_path):
    os.system('rm -rf %s' % vis_path)
os.mkdir(vis_path)
aval_data = []


def load_result(fn):
    res = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            line = line.strip().split()
            if len(line) != 6:
                continue
            score = float(line[-1])
            if score < 0.3:
                continue
            frame = int(line[0]) - 1
            x0, y0, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            if h < 1:
                continue
            x1, y1 = x0 + w, y0 + h
            if frame not in res:
                res[frame] = [(x0, y0, x1, y1, score)]
            else:
                res[frame].append((x0, y0, x1, y1, score))
    return res


def compute_hit_box(gts, preds):
    hit_num = 0
    hit_scores = .0
    hit_box = []
    ex_rois = [[x[0], x[1], x[2], x[3]] for x in preds]
    gt_rois = [[x[0], x[1], x[2], x[3]] for x in gts]
    ious = calc_ious(np.array(ex_rois), np.array(gt_rois))
    max_ious = ious.max(axis=1)
    max_idx = ious.argmax(axis=1)
    record = [False] * len(gts)
    for i in range(len(max_ious)):
        if max_ious[i] > 0.5 and not record[max_idx[i]]:
            hit_num += 1
            hit_scores += preds[i][-1]
            hit_box.append([int(x) for x in preds[i][:-1]] + [float("%.3f" % preds[i][-1])])
            record[max_idx[i]] = True
    return hit_num, hit_scores, hit_box


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

font = cv2.FONT_HERSHEY_SIMPLEX
for sid in os.listdir(gt_path):
    for vid in os.listdir(os.path.join(gt_path, sid)):
        cur_gt = load_result(os.path.join(gt_path, sid, vid))
        cur_frcn = load_result(os.path.join(frcn_path, sid, vid))
        cur_frcn_0 = load_result(os.path.join(frcn_a0_path, sid, vid))
        cur_frcn_dt = load_result(os.path.join(frcn_dt_path, sid, vid))
        for frame in cur_gt:
            if len(cur_gt[frame]) > 3:
                continue
            if frame not in cur_frcn or frame not in cur_frcn_0 or frame not in cur_frcn_dt:
                continue
            n1, s1, b1 = compute_hit_box(cur_gt[frame], cur_frcn[frame])
            n2, s2, b2 = compute_hit_box(cur_gt[frame], cur_frcn_0[frame])
            n3, s3, b3 = compute_hit_box(cur_gt[frame], cur_frcn_dt[frame])
            if n2 > n1 or (n2 == n1 and s2 - s1 > 0.3):
                cur_dir = os.path.join(vis_path, "%s_%s_%d" % (sid, vid[:-4], frame))
                os.mkdir(cur_dir)
                img_path = os.path.join(caltech_img_path, "%s_%s_%d.jpg" % (sid, vid[:-4], frame))
                img_frcn = cv2.imread(img_path)
                img_attn, img_dt = deepcopy(img_frcn), deepcopy(img_frcn)
                for i, b in enumerate(b1):
                    cv2.rectangle(img_frcn, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    if i % 3 == 0:
                        cv2.putText(img_frcn, str(b[-1]), (b[0], b[1] - 10), font, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(img_frcn, str(b[-1]), (b[0], b[3] + 30), font, 0.7, (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(cur_dir, "frcn.jpg"), img_frcn)
                for i, b in enumerate(b2):
                    cv2.rectangle(img_attn, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    if i % 3 == 0:
                        cv2.putText(img_attn, str(b[-1]), (b[0], b[1] - 10), font, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(img_attn, str(b[-1]), (b[0], b[3] + 30), font, 0.7, (255, 0, 0), 2)
                    cv2.imwrite(os.path.join(cur_dir, "attn.jpg"), img_attn)
                flag = n3 > n1 or (n3 == n1 and s3 - s1 > 0.3)
                for i, b in enumerate(b3):
                    cv2.rectangle(img_dt, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    if i % 3 == 0:
                        cv2.putText(img_dt, str(b[-1]), (b[0], b[1] - 10), font, 0.7, (255, 0, 0), 2)
                    else:
                        cv2.putText(img_dt, str(b[-1]), (b[0], b[3] + 30), font, 0.7, (255, 0, 0), 2)
                    if not flag:
                        cv2.imwrite(os.path.join(cur_dir, "dt.jpg"), img_dt)
                    else:
                        cv2.imwrite(os.path.join(cur_dir, "dt_mark.jpg"), img_dt)

