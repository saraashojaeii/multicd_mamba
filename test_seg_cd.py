"""
Testing script for semantic change detection models.
Evaluates both semantic segmentation and binary change detection with comprehensive metrics.
"""

import os
import sys
import argparse
import logging
from itertools import islice
from datetime import datetime
from collections import OrderedDict

# ---- CUDA mem config must come before torch import ----
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchinfo import summary
except Exception:
    summary = None

import numpy as np
import random
from tqdm import tqdm
import wandb

# project deps
import data as Data
import models as Model
import core.metrics as Metrics
from core.utils import *
from core.logger import parse as parse_cfg
from core.logger import setup_logger, dict2str, dict_to_nonedict
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import *
from core.metrics import (
    compute_semantic_metrics_on_changed, 
    compute_semantic_metrics_on_predicted_changed,
    compute_per_class_metrics, 
    compute_transition_metrics
)


# ----------------------------- helpers ----------------------------- #
def set_all_seeds(seed: int | None):
    seed = 42 if seed is None else int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_stamped_dirs(opt: dict, exp_folder: str):
    """Stamp log/result/checkpoint dirs with a unique suffix."""
    if 'path_cd' in opt and isinstance(opt['path_cd'], dict):
        for k in ['log', 'result', 'checkpoint']:
            if k in opt['path_cd'] and isinstance(opt['path_cd'][k], str):
                base_dir = opt['path_cd'][k]
                stamped = os.path.join(base_dir, exp_folder)
                opt['path_cd'][k] = stamped
                os.makedirs(stamped, exist_ok=True)
        opt['path_cd']['exp_folder'] = exp_folder
    else:
        print("[warn] opt['path_cd'] not found; skipping folder stamping")


def derive_change_bin(seg_t1: torch.Tensor, seg_t2: torch.Tensor) -> torch.Tensor:
    """Binary change (0/1) from label maps [B,H,W]."""
    return (seg_t1 != seg_t2).long()


def safe_to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.uint8)
    return np.squeeze(arr)


def unpack_outputs(outputs):
    """Unpack model outputs into seg_t1, seg_t2, change."""
    seg1 = seg2 = change = aux = None
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 2:
            seg1, seg2 = outputs
        elif len(outputs) == 3:
            seg1, seg2, change = outputs
        elif len(outputs) >= 4:
            seg1, seg2, change, aux = outputs[0], outputs[1], outputs[2], outputs[3]
    elif isinstance(outputs, dict):
        seg1 = outputs.get('seg_t1')
        seg2 = outputs.get('seg_t2')
        change = outputs.get('change')
        aux = outputs.get('aux')
    else:
        seg1 = outputs
    return seg1, seg2, change, aux


# ----------------------------- main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--tag', type=str, default='', help='Experiment tag')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights for testing')
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--max_test_batches', type=int, default=-1)
    parser.add_argument('--change_threshold', type=float, default=0.5, help='Threshold for binary change (if using sigmoid)')
    args = parser.parse_args()

    # Parse config
    opt = parse_cfg(args)
    opt = dict_to_nonedict(opt)

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    set_all_seeds(args.seed)

    # Experiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_folder = f"{opt['name']}_{args.dataset}_test_{args.tag}_{timestamp}" if args.tag else f"{opt['name']}_{args.dataset}_test_{timestamp}"
    make_stamped_dirs(opt, exp_folder)

    # Logger
    setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # W&B
    use_wandb = bool(args.wandb_project or opt.get('wandb', {}).get('project', ''))
    if use_wandb:
        wandb.init(
            project=args.wandb_project or opt['wandb']['project'],
            name=exp_folder,
            config=opt,
            tags=['test']
        )
        logger.info(f"✓ W&B initialized: project={wandb.run.project}, name={wandb.run.name}")
    else:
        logger.info("✗ W&B not initialized")

    # ----------------------------- data ----------------------------- #
    logger.info("Creating test dataset...")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)

    # Add colormap to dataset options if present
    if 'colormap' in opt:
        opt['datasets']['test']['colormap'] = opt['colormap']
    
    test_dataset = Data.create_scd_dataset(opt['datasets']['test'], 'test')
    logger.info(f'Test dataset length: {len(test_dataset)}')
    
    test_loader = Data.create_cd_dataloader(test_dataset, opt['datasets']['test'], 'test', seed_worker, g)
    logger.info(f"Test batches: {len(test_loader)}")

    # ----------------------------- model ----------------------------- #
    logger.info("Creating model...")
    cd_model = Model.create_CD_model(opt)
    cd_model = cd_model.to(device)

    if summary is not None:
        try:
            summary(cd_model, input_size=[(1, opt['model']['in_channels'], 256, 256), (1, opt['model']['in_channels'], 256, 256)])
        except Exception as e:
            logger.warning(f"torchinfo summary failed: {e}")

    # Load weights
    logger.info(f"Loading checkpoint from {args.weights}")
    ckpt = torch.load(args.weights, map_location=device)
    if 'model' in ckpt:
        cd_model.load_state_dict(ckpt['model'])
        logger.info(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    else:
        cd_model.load_state_dict(ckpt)
    logger.info("✓ Model weights loaded successfully")

    # Model config
    num_classes = opt['model']['n_classes']
    ignore_index = opt.get('train', {}).get('ignore_index', 255)
    colormap = opt.get('colormap', None)

    # ----------------------------- testing ----------------------------- #
    logger.info("=" * 80)
    logger.info("Starting comprehensive testing")
    logger.info("=" * 80)
    
    cd_model.eval()
    test_metric = ConfuseMatrixMeter(n_class=num_classes)
    test_tp = test_fp = test_fn = test_tn = 0

    _max_test = args.max_test_batches
    _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
    _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader

    # Accumulators for batch-level metrics
    all_p1 = []
    all_p2 = []
    all_y1 = []
    all_y2 = []
    all_chg_mask = []

    with torch.no_grad():
        for tstep, tb in enumerate(tqdm(_test_iter, total=_test_total, desc="Testing")):
            t1 = tb['A'].to(device)
            t2 = tb['B'].to(device)
            y1 = tb['L1'].to(device).long()
            y2 = tb['L2'].to(device).long()

            # Forward pass
            outputs = cd_model(t1, t2)
            seg_t1, seg_t2, change_pred, _ = unpack_outputs(outputs)

            # Semantic predictions
            p1 = torch.argmax(seg_t1, dim=1)
            p2 = torch.argmax(seg_t2, dim=1)
            test_metric.update_cm(pr=safe_to_numpy_uint8(p1), gt=safe_to_numpy_uint8(y1))
            test_metric.update_cm(pr=safe_to_numpy_uint8(p2), gt=safe_to_numpy_uint8(y2))

            # Change detection metrics
            if change_pred is not None:
                if change_pred.size(1) == 2:
                    chg_mask = torch.argmax(change_pred, dim=1)
                else:
                    chg_mask = (torch.sigmoid(change_pred[:, 0]) > args.change_threshold).long()
                
                gt_change = derive_change_bin(y1, y2)
                pr_np, gt_np = chg_mask.cpu().numpy(), gt_change.cpu().numpy()
                
                tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                tn = np.logical_and(pr_np == 0, gt_np == 0).sum()
                
                test_tp += tp
                test_fp += fp
                test_fn += fn
                test_tn += tn

                # Store for aggregated metrics
                all_chg_mask.append(chg_mask.cpu().numpy())
            else:
                all_chg_mask.append(None)

            # Store predictions and GT for aggregated metrics
            all_p1.append(p1.cpu().numpy())
            all_p2.append(p2.cpu().numpy())
            all_y1.append(y1.cpu().numpy())
            all_y2.append(y2.cpu().numpy())

    # ----------------------------- compute metrics ----------------------------- #
    logger.info("Computing comprehensive metrics...")
    
    # Overall semantic segmentation metrics
    test_scores = test_metric.get_scores()
    test_mf1 = test_scores['mf1']
    test_miou = test_scores['miou']
    test_acc = test_scores['acc']

    # Binary change detection metrics
    if (test_tp + test_fp + test_fn) > 0:
        test_chg_prec = test_tp / max(test_tp + test_fp, 1e-8)
        test_chg_rec = test_tp / max(test_tp + test_fn, 1e-8)
        test_chg_f1 = 2 * test_tp / max(2 * test_tp + test_fp + test_fn, 1e-8)
        test_chg_iou = test_tp / max(test_tp + test_fp + test_fn, 1e-8)
        test_chg_acc = (test_tp + test_tn) / max(test_tp + test_tn + test_fp + test_fn, 1e-8)
    else:
        test_chg_prec = test_chg_rec = test_chg_f1 = test_chg_iou = test_chg_acc = 0.0

    # Aggregate all batches for detailed metrics
    all_p1_np = np.concatenate(all_p1, axis=0)
    all_p2_np = np.concatenate(all_p2, axis=0)
    all_y1_np = np.concatenate(all_y1, axis=0)
    all_y2_np = np.concatenate(all_y2, axis=0)

    # Metrics on GT changed pixels
    test_changed_metrics = compute_semantic_metrics_on_changed(
        all_p1_np, all_p2_np, all_y1_np, all_y2_np, num_classes, ignore_index
    )

    # Metrics on PREDICTED changed pixels
    if all_chg_mask[0] is not None:
        all_chg_mask_np = np.concatenate(all_chg_mask, axis=0)
        test_pred_changed_metrics = compute_semantic_metrics_on_predicted_changed(
            all_p1_np, all_p2_np, all_y1_np, all_y2_np, all_chg_mask_np, num_classes, ignore_index
        )
    else:
        test_pred_changed_metrics = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'num_pixels': 0}

    # Per-class metrics
    test_per_class_metrics = compute_per_class_metrics(
        all_p1_np, all_p2_np, all_y1_np, all_y2_np, num_classes, ignore_index
    )

    # Transition metrics
    top_transitions = [(1, 4), (0, 1), (1, 0), (0, 4)]
    test_transition_metrics = compute_transition_metrics(
        all_p1_np, all_p2_np, all_y1_np, all_y2_np, top_transitions, num_classes, ignore_index
    )

    # ----------------------------- logging ----------------------------- #
    logger.info("=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Overall Semantic Segmentation:")
    logger.info(f"  mF1:  {test_mf1:.4f}")
    logger.info(f"  mIoU: {test_miou:.4f}")
    logger.info(f"  OA:   {test_acc:.4f}")
    logger.info(f"")
    logger.info(f"Binary Change Detection:")
    logger.info(f"  Precision: {test_chg_prec:.4f}")
    logger.info(f"  Recall:    {test_chg_rec:.4f}")
    logger.info(f"  F1:        {test_chg_f1:.4f}")
    logger.info(f"  IoU:       {test_chg_iou:.4f}")
    logger.info(f"  Accuracy:  {test_chg_acc:.4f}")
    logger.info(f"")
    logger.info(f"Semantic on GT Changed Pixels:")
    logger.info(f"  IoU: {test_changed_metrics['iou']:.4f}")
    logger.info(f"  F1:  {test_changed_metrics['f1']:.4f}")
    logger.info(f"  Acc: {test_changed_metrics['accuracy']:.4f}")
    logger.info(f"")
    logger.info(f"Semantic on PREDICTED Changed Pixels:")
    logger.info(f"  IoU:   {test_pred_changed_metrics['iou']:.4f}")
    logger.info(f"  F1:    {test_pred_changed_metrics['f1']:.4f}")
    logger.info(f"  Acc:   {test_pred_changed_metrics['accuracy']:.4f}")
    logger.info(f"  Count: {test_pred_changed_metrics['num_pixels']}")
    logger.info("=" * 80)

    # W&B logging
    if use_wandb:
        test_metrics = {
            # Overall semantic segmentation
            'test/epoch_mF1': test_mf1,
            'test/epoch_mIoU': test_miou,
            'test/epoch_OA': test_acc,
            # Binary change detection
            'test/epoch_change_prec': test_chg_prec,
            'test/epoch_change_rec': test_chg_rec,
            'test/epoch_change_f1': test_chg_f1,
            'test/epoch_change_iou': test_chg_iou,
            'test/epoch_change_acc': test_chg_acc,
            # Metrics on GT changed pixels (ground truth change mask)
            'test/changed_pixels_gt_iou': test_changed_metrics['iou'],
            'test/changed_pixels_gt_f1': test_changed_metrics['f1'],
            'test/changed_pixels_gt_acc': test_changed_metrics['accuracy'],
            # Metrics on PREDICTED changed pixels (predicted change mask)
            'test/changed_pixels_pred_iou': test_pred_changed_metrics['iou'],
            'test/changed_pixels_pred_f1': test_pred_changed_metrics['f1'],
            'test/changed_pixels_pred_acc': test_pred_changed_metrics['accuracy'],
            'test/changed_pixels_pred_count': test_pred_changed_metrics['num_pixels'],
            # Per-class metrics
            'test/class_building_iou': test_per_class_metrics['iou_per_class'][4] if len(test_per_class_metrics['iou_per_class']) > 4 else 0.0,
            'test/class_building_f1': test_per_class_metrics['f1_per_class'][4] if len(test_per_class_metrics['f1_per_class']) > 4 else 0.0,
            'test/class_nvg_surf_iou': test_per_class_metrics['iou_per_class'][1] if len(test_per_class_metrics['iou_per_class']) > 1 else 0.0,
            'test/class_nvg_surf_f1': test_per_class_metrics['f1_per_class'][1] if len(test_per_class_metrics['f1_per_class']) > 1 else 0.0,
            'test/class_water_iou': test_per_class_metrics['iou_per_class'][3] if len(test_per_class_metrics['iou_per_class']) > 3 else 0.0,
            'test/class_water_f1': test_per_class_metrics['f1_per_class'][3] if len(test_per_class_metrics['f1_per_class']) > 3 else 0.0,
            'test/class_playground_iou': test_per_class_metrics['iou_per_class'][5] if len(test_per_class_metrics['iou_per_class']) > 5 else 0.0,
            'test/class_playground_f1': test_per_class_metrics['f1_per_class'][5] if len(test_per_class_metrics['f1_per_class']) > 5 else 0.0,
        }

        # Add transition metrics
        for trans_key, trans_val in test_transition_metrics.items():
            test_metrics[f'test/transition_{trans_key}_acc'] = trans_val['accuracy']
            test_metrics[f'test/transition_{trans_key}_count'] = trans_val['count']

        # Add per-class F1 and IoU for all classes
        if 'f1_per_class' in test_scores:
            for cls_idx, f1_val in enumerate(test_scores['f1_per_class']):
                test_metrics[f'test/class_{cls_idx}_f1'] = f1_val
        if 'iou_per_class' in test_scores:
            for cls_idx, iou_val in enumerate(test_scores['iou_per_class']):
                test_metrics[f'test/class_{cls_idx}_iou'] = iou_val

        wandb.log(test_metrics)
        logger.info("✓ Metrics logged to W&B")

        # Create summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Overall mF1", f"{test_mf1:.4f}"],
                ["Overall mIoU", f"{test_miou:.4f}"],
                ["Overall Accuracy", f"{test_acc:.4f}"],
                ["Change F1", f"{test_chg_f1:.4f}"],
                ["Change IoU", f"{test_chg_iou:.4f}"],
                ["Changed Pixels F1 (GT)", f"{test_changed_metrics['f1']:.4f}"],
                ["Changed Pixels F1 (Pred)", f"{test_pred_changed_metrics['f1']:.4f}"],
            ]
        )
        wandb.log({"test/summary": summary_table})

    logger.info("✓ Testing complete!")
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
