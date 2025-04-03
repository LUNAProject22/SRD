# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
import pickle
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training')
    model = build_detection_model(cfg) 
    debug_print(logger, 'end model construction')

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
 
    fix_eval_modules(eval_modules)

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",]
    else:
        slow_heads = []

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, 'end optimizer and shcedule')
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[get_rank()], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # cp checkpoint path to output_dir, load it
    assert os.path.exists(cfg.MODEL.PRETRAINED_DETECTOR_CKPT), "Pretrain ckpt doesn't exist!"
    out_ckpt_file = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    with open(out_ckpt_file, "wt") as f:
        f.write(cfg.MODEL.PRETRAINED_DETECTOR_CKPT)

    if checkpointer.has_checkpoint():
        _ = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,
                                update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)

    debug_print(logger, 'end load checkpointer')
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    debug_print(logger, 'end dataloader')

    logger.info("Start training")
    dic = {}
    to_save = []
    end = False
    for iteration, (images, targets, indexes) in enumerate(train_data_loader):
        model.eval()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            (relation_logits, rel_pair_idxs, rel_labels) = model(images, targets)

        if iteration % 200 == 0:
            logger.info("iters: {}, {}".format(iteration, len(dic)))
        for t, logits, pairs, rel_label in zip(targets, relation_logits, rel_pair_idxs, rel_labels):
            assert logits.shape[0]==pairs.shape[0]
            cur_data = t.get_field("train_data")
            # assert np.all(cur_data["triples"][:, 2] == t.get_field("relation_labels").nonzero()[:, 1].cpu().numpy())
            cur_data = modify_logits(cur_data, logits)
            cur_data['rel_pair_idxs'] = pairs.cpu().numpy()
            cur_data['rel_fg_bg'] = rel_label.cpu().numpy()
            image_id = cur_data['image_id']
            if image_id in dic:
                end = True
                break
            else:
                to_save.append(cur_data)
                dic[image_id] = None
        if end:
            break
    print(len(dic))
    import json
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    json.dump(dic, open(cfg.OUTPUT_DIR+str(local_rank)+".json", "w"))
    pickle.dump(to_save, open(cfg.OUTPUT_DIR+str(local_rank)+".pk", "wb"))
    return list(dic.keys())


def modify_logits(cur_data, rel_logits):
    # just append the logits
    cur_data["logits"] = rel_logits.cpu().numpy()
    cur_data["rel_prob"] = F.softmax(rel_logits, -1).cpu().numpy()
    return cur_data


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/relation_deduction/infer_logits.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else: # use 1 gpu, for debug
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE set manually: {rank}/{world_size}")

    if args.distributed:
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    # logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    image_ids = train(cfg, args.local_rank, args.distributed, logger)

    l = pickle.load(open(cfg.OUTPUT_DIR + "0.pk", "rb"))
    dic = {}
    for d in l:
        dic[d['image_id']] = d

    save_path = cfg.USE_LOGITS.DATA_WITH_LOGITS
    pickle.dump(dic, open(save_path, "wb"))
    print("Saved pickle file: ", save_path)


if __name__ == "__main__":
    main()
