# Train a fast-rcnn model with attention module and transformation branch
import os
import json
import logging
import torch

from detectron2.structures import BoxMode
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.logger import setup_logger
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
cfg.merge_from_file("./configs/frcn_tb.yaml")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
logger = logging.getLogger("frcn")
if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
    setup_logger()

model = build_model(cfg)
logger.info("Model:\n{}".format(model))
model.train()
optimizer = build_optimizer(cfg, model)
scheduler = build_lr_scheduler(cfg, optimizer)

checkpointer = DetectionCheckpointer(
    model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
)
start_iter = (
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False).get("iteration", -1) + 1
)
ckpt = Checkpointer(model)
ckpt.load("./frcn_attn_0/model_0044999.pth")
max_iter = cfg.SOLVER.MAX_ITER

periodic_checkpointer = PeriodicCheckpointer(
    checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
)

writers = (
    [
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
        TensorboardXWriter(cfg.OUTPUT_DIR),
    ]
    if comm.is_main_process()
    else []
)

# compared to "train_net.py", we do not support accurate timing and
# precise BN here, because they are not trivial to implement
data_loader = build_detection_train_loader(cfg)
logger.info("Starting training from iteration {}".format(start_iter))
with EventStorage(start_iter) as storage:
    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        iteration = iteration + 1
        storage.step()
        loss_dict = model(data)
        losses = sum(loss for loss in loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict

        loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if comm.is_main_process():
            storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
        scheduler.step()

        if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)

