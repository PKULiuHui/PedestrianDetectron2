# Build a new model, a varient of fast-rcnn with attention and transformation module
import logging
import pickle
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs


# attention module only make sense with resnet 'res4' feature
def build_attention(cfg):
    attention = None
    if not cfg.MODEL.ATTENTION:
        return attention
    if not len(cfg.MODEL.RESNETS.OUT_FEATURES) != ['res4']:
        return attention
    attention = nn.Sequential(
        nn.Conv2d(1024, 128, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 1, 1),
        nn.Sigmoid(),
    )
    return attention


def build_transformation():
    return nn.Sequential(
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 1),
        nn.ReLU(inplace=True),
    )


@META_ARCH_REGISTRY.register()
class FeatureTransformationRCNN(nn.Module):
    """
    Feature Transformation R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.attention = build_attention(cfg)
        self.mse_loss = nn.MSELoss(reduction="sum") if cfg.MODEL.ATTENTION_LOSS else None
        self.mse_weight = cfg.MODEL.ATTENTION_LOSS_WEIGHT
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.tmp = nn.Linear(10, 10)

        trans_center = pickle.load(open(cfg.MODEL.TRANSFORM_CENTER, 'rb'))
        trans_center['pos_center'] = torch.FloatTensor(trans_center['pos_center']).to(self.device)
        trans_center['neg_center'] = torch.FloatTensor(trans_center['neg_center']).to(self.device)
        self.trans_center = trans_center
        self.transformation = build_transformation()
        self.box_head = deepcopy(self.roi_heads.box_head)
        self.box_predictor = deepcopy(self.roi_heads.box_predictor)
        self.sl1_loss = nn.SmoothL1Loss(reduction="none") if cfg.MODEL.TRANSFORM_LOSS else None
        self.sl1_weight = cfg.MODEL.TRANSFORM_LOSS_WEIGHT
        self.reg_loss = cfg.MODEL.REG_LOSS

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """

        inputs = [x for x in batched_inputs]
        prop_boxes = [p for p in proposals]
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(inputs, prop_boxes):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        losses = {}
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        if self.attention is not None:
            attn_weights = self.attention(features['res4'])
            features['res4'] = features['res4'] * attn_weights.repeat(1, features['res4'].size(1), 1, 1)
            if self.mse_loss is not None:
                attn_loss = self.attn_loss(attn_weights, gt_instances)
                losses.update(attn_loss)

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        proposals, box_features = self.roi_heads.transform_forward(images, features, proposals, gt_instances)
        box_features_T = box_features + self.transformation(box_features)
        box_features = self.box_head(box_features_T)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        transform_losses = self.transform_loss(pred_class_logits, pred_proposal_deltas, proposals, box_features_T)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses.update(transform_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.attention is not None:
            attn_weights = self.attention(features['res4'])
            features['res4'] = features['res4'] * attn_weights.repeat(1, features['res4'].size(1), 1, 1)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        proposals, box_features_0 = self.roi_heads.transform_forward(images, features, proposals)

        box_features = self.roi_heads.box_head(box_features_0)
        pred_class_logits_0, pred_proposal_deltas_0 = self.roi_heads.box_predictor(box_features)

        box_features_T = box_features_0 + self.transformation(box_features_0)
        box_features = self.box_head(box_features_T)
        pred_class_logits_1, pred_proposal_deltas_1= self.box_predictor(box_features)

        outputs = FastRCNNOutputs(
            self.roi_heads.box2box_transform,
            pred_class_logits_1,
            pred_proposal_deltas_0,
            proposals,
            self.roi_heads.smooth_l1_beta,
        )

        results, _ = outputs.inference(
            self.roi_heads.test_score_thresh, self.roi_heads.test_nms_thresh, self.roi_heads.test_detections_per_img
        )

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def transform_inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def attn_loss(self, attn_weights, gt_instances):
        # loss = torch.Tensor([.0])[0].to(self.device)
        img_size = (480, 640)
        gt_attn = np.zeros((attn_weights.size(0), 1, 480, 640))
        for i, ins in enumerate(gt_instances):
            for box in gt_instances[0].gt_boxes:
                box = box.tolist()
                l, t, r, b = int(round(box[1])), int(round(box[0])), int(round(box[3])), int(round(box[2]))
                gt_attn[i, 0, l:r, t:b] = 1
        gt_attn = gt_attn[:, :, 8:img_size[0]:16, 8:img_size[1]:16]
        gt_attn = torch.FloatTensor(gt_attn).to(self.device)
        loss = self.mse_weight * self.mse_loss(attn_weights, gt_attn) / attn_weights.size(0)
        return {'loss_attn': loss}

    def transform_loss(self, pred_class_logits, pred_proposal_deltas, proposals, box_features_T):
        losses = {}
        outputs = FastRCNNOutputs(
            self.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.roi_heads.smooth_l1_beta,
        )
        loss1 = outputs.losses()
        losses.update({"loss_cls": loss1["loss_cls"]})
        if self.reg_loss:
            losses.update({"loss_box_reg": loss1["loss_box_reg"]})
        if self.sl1_loss is None:
            return losses

        box_features_T = box_features_T.view(box_features_T.size(0), -1)
        pos_center = self.trans_center['pos_center'].unsqueeze(0).repeat(box_features_T.size(0), 1)
        neg_center = self.trans_center['neg_center'].unsqueeze(0).repeat(box_features_T.size(0), 1)
        gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        loss_pos = torch.mean(self.sl1_loss(box_features_T, pos_center), dim=-1) * (1 - gt_classes).float()
        loss_neg = torch.mean(self.sl1_loss(box_features_T, neg_center), dim=-1) * gt_classes.float()
        loss_trans = self.sl1_weight * torch.mean(loss_pos + loss_neg, dim=-1)
        losses.update({"loss_trans": loss_trans})
        return losses

    def predict(self, batched_inputs):
        """
        Compute scores for pos/neg cand_props
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.attention is not None:
            attn_weights = self.attention(features['res4'])
            features['res4'] = features['res4'] * attn_weights.repeat(1, features['res4'].size(1), 1, 1)

        if self.proposal_generator:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        box_features, box_scores = self.roi_heads.get_prop_feature(images, features, proposals, None)

        return box_features, box_scores, batched_inputs[0]['proposals'].objectness_logits
