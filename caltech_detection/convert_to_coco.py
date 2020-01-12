# convert caltech dataset and proposals into coco format
import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from detectron2.structures import BoxMode

caltech_anno_path = '/envs/shareB/liuhui/Detection/caltech/data/consistent_annotations.json'
caltech_img_path = '/envs/shareB/liuhui/Detection/caltech/data/images/'
caltech_prop_path = '/envs/shareB/liuhui/Detection/caltech/data/proposals/'
train_annos, test_annos = [], []
train_props = {'ids': [], 'boxes': [], 'objectness_logits': []}
test_props = {'ids': [], 'boxes': [], 'objectness_logits': []}
caltech_annos = json.load(open(caltech_anno_path, 'r'))
print('In total there are {} annotations'.format(len(caltech_annos)))

for i, k in enumerate(tqdm(caltech_annos)):
    set_id = int(k[3:5])
    file_name = os.path.join(caltech_img_path, k + '.jpg')
    image_id = i
    height, width = 480, 640
    annos =[]
    for person in caltech_annos[k]:
        bbox = person['pos']
        if bbox[3] < 50:
            continue
        if person['occl'] == 1:
            # filter bboxes which are occluded more than 70%
            vbbox = person['posv']
            if isinstance(vbbox, int):
                # it seems that sometimes 'posv' is mixed up with 'lock'
                vbbox = person['lock']
                assert isinstance(vbbox, list) and len(vbbox) == 4
            w1, h1, w2, h2 = bbox[2], bbox[3], vbbox[2], vbbox[3]
            if w2 * h2 / (w1 * h1) <= 0.3:
                continue
        annos.append({'bbox': bbox, 'category_id': 0, 'iscrowd': 0, 'occl': person['occl']})

    proposal_boxes, proposal_objectness_logits = [], []
    with open(os.path.join(caltech_prop_path, k + '.txt')) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            assert len(line) == 5
            line = [float(x) for x in line]
            proposal_boxes.append(line[:4])
            proposal_objectness_logits.append(line[-1])
    proposal_boxes, proposal_objectness_logits = np.array(proposal_boxes), np.array(proposal_objectness_logits)
    img_anno = {
        'file_name': file_name,
        'image_id': image_id,
        'height': height,
        'width': width,
        'annotations': annos,
        # 'proposal_boxes': proposal_boxes,
        # 'proposal_objectness_logits': proposal_objectness_logits,
        # 'proposal_bbox_mode': proposal_bbox_mode,
    }
    if set_id <= 5:
        train_annos.append(img_anno)
        train_props['ids'].append(image_id)
        train_props['boxes'].append(proposal_boxes)
        train_props['objectness_logits'].append(proposal_objectness_logits)
    else:
        test_annos.append(img_anno)
        test_props['ids'].append(image_id)
        test_props['boxes'].append(proposal_boxes)
        test_props['objectness_logits'].append(proposal_objectness_logits)

json.dump(train_annos, open('train_annos.json', 'w'))
json.dump(test_annos, open('test_annos.json', 'w'))
pickle.dump(train_props, open('train_prop.pkl', 'wb'))
pickle.dump(test_props, open('test_prop.pkl', 'wb'))
