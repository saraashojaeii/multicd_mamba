"""
Testing script for change-detection models.
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
import torch.optim as optim
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
from misc.torchutils import get_scheduler, save_network
from models.loss import *

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


# ----------------------------- main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='test')
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
    exp_folder = f"{opt['name']}_{args.dataset}_{args.tag}_{timestamp}" if args.tag else f"{opt['name']}_{args.dataset}_{timestamp}"
    make_stamped_dirs(opt, exp_folder)

    # Logger
    setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # W&B
    use_wandb = bool(args.wandb_project or opt.get('wandb', {}).get('project', ''))
    logger.info(f"W&B check: args.wandb_project={args.wandb_project}, config_project={opt.get('wandb', {}).get('project', '')}, use_wandb={use_wandb}")
    if use_wandb:
        wandb.init(
            project=args.wandb_project or opt['wandb']['project'],
            name=exp_folder,
            config=opt
        )
        logger.info(f"✓ W&B initialized: project={wandb.run.project}, name={wandb.run.name}, run_id={wandb.run.id}")
    else:
        logger.info(f"✗ W&B not initialized (use_wandb={use_wandb}, phase={args.phase})")

    # ----------------------------- data ----------------------------- #
    logger.info("Creating test dataset...")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)

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
    else:
        cd_model.load_state_dict(ckpt)
    logger.info("✓ Model weights loaded successfully")

    # ----------------------------- testing ----------------------------- #
    logger.info("=" * 60)
    logger.info("Starting testing (change-only mode)")
    logger.info("=" * 60)
    cd_model.eval()

    # Initialize metrics
    test_metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection with SeK
    test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0
    
    # Initialize from-to transition matrix
    n_classes = opt['model']['n_classes']
    transition_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    _max_test = args.max_test_batches
    _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
    _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader

    # Create test result directory
    test_result_path = os.path.join(opt['path_cd']['result'], 'test')
    os.makedirs(test_result_path, exist_ok=True)

    with torch.no_grad():
        for tstep, tb in enumerate(tqdm(_test_iter, total=_test_total, desc="Test")):
            ti1 = tb['A'].to(device)
            ti2 = tb['B'].to(device)
            y1 = tb['L1'].to(device).long()
            y2 = tb['L2'].to(device).long()

            # Forward pass
            outputs = cd_model(ti1, ti2)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                change_pred = outputs.get('change', None)
            elif isinstance(outputs, (list, tuple)):
                if len(outputs) == 3:
                    _, _, change_pred = outputs
                else:
                    change_pred = outputs[0]
            else:
                change_pred = outputs

            # Derive ground truth change
            change_gt = derive_change_bin(y1, y2)

            # Get binary prediction
            if change_pred.size(1) == 2:
                cmask = torch.argmax(change_pred, dim=1)
            else:
                cmask = (torch.sigmoid(change_pred[:, 0]) > args.change_threshold).long()

            pr_np, gt_np = cmask.cpu().numpy(), change_gt.cpu().numpy()
            
            # Update confusion matrix for SeK computation
            test_metric.update_cm(pr=pr_np.astype(np.uint8), gt=gt_np.astype(np.uint8))
            
            # Update from-to transition matrix (semantic class transitions)
            y1_np = y1.cpu().numpy().flatten()
            y2_np = y2.cpu().numpy().flatten()
            for from_class, to_class in zip(y1_np, y2_np):
                if 0 <= from_class < n_classes and 0 <= to_class < n_classes:
                    transition_matrix[from_class, to_class] += 1
            
            # Manual TP/FP/FN/TN tracking
            tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
            fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
            fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
            tn = np.logical_and(pr_np == 0, gt_np == 0).sum()

            test_tp += tp; test_fp += fp; test_fn += fn; test_tn += tn

            # Save visualizations for first few batches
            if tstep < 10:
                # Save input images
                img_A = Metrics.tensor2img(tb['A'], out_type=np.uint8, min_max=(-1, 1))
                img_B = Metrics.tensor2img(tb['B'], out_type=np.uint8, min_max=(-1, 1))
                
                # Save predictions and ground truth
                pred_tensor = cmask.unsqueeze(1) if cmask.dim() == 3 else cmask.unsqueeze(0).unsqueeze(0)
                gt_tensor = change_gt.unsqueeze(1) if change_gt.dim() == 3 else change_gt.unsqueeze(0).unsqueeze(0)
                
                pred_cm = Metrics.tensor2img(pred_tensor.repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))
                gt_cm = Metrics.tensor2img(gt_tensor.repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))

                Metrics.save_img(img_A, f'{test_result_path}/img_A_{tstep}.png')
                Metrics.save_img(img_B, f'{test_result_path}/img_B_{tstep}.png')
                Metrics.save_img(pred_cm, f'{test_result_path}/img_pred_cm_{tstep}.png')
                Metrics.save_img(gt_cm, f'{test_result_path}/img_gt_cm_{tstep}.png')

    # Final metrics from confusion matrix
    test_scores = test_metric.get_scores()
    test_prec = test_tp / (test_tp + test_fp + 1e-8)
    test_rec = test_tp / (test_tp + test_fn + 1e-8)
    test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)
    test_iou = test_tp / (test_tp + test_fp + test_fn + 1e-8)
    test_acc = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn + 1e-8)
    test_sek = test_scores.get('SCD_Sek', 0.0)  # SeK metric from confusion matrix
    
    # Normalize transition matrix to percentages
    total_pixels = transition_matrix.sum()
    transition_matrix_pct = (transition_matrix / total_pixels * 100) if total_pixels > 0 else transition_matrix

    logger.info("=" * 60)
    logger.info("Test Results:")
    logger.info(f"  Precision: {test_prec:.4f}")
    logger.info(f"  Recall:    {test_rec:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    logger.info(f"  IoU:       {test_iou:.4f}")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  SeK:       {test_sek:.4f}")
    logger.info("=" * 60)
    
    # Log transition matrix
    logger.info("\nFrom-To Transition Matrix (percentages):")
    logger.info("=" * 60)
    
    # Define class names (adjust based on your dataset)
    class_names = ['no_veg', 'veg_turf', 'tree', 'water', 'building', 'playground']
    if n_classes != len(class_names):
        class_names = [f'class_{i}' for i in range(n_classes)]
    
    # Print header
    header = "From\\To  " + "  ".join([f"{name:>10}" for name in class_names])
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print matrix rows
    for i, from_name in enumerate(class_names):
        row_str = f"{from_name:>10}  " + "  ".join([f"{transition_matrix_pct[i, j]:>9.2f}%" for j in range(n_classes)])
        logger.info(row_str)
    
    logger.info("=" * 60)
    logger.info(f"Results saved to {test_result_path}")
    
    # Save transition matrix to file
    import json
    transition_data = {
        'matrix_counts': transition_matrix.tolist(),
        'matrix_percentages': transition_matrix_pct.tolist(),
        'class_names': class_names,
        'total_pixels': int(total_pixels)
    }
    with open(os.path.join(test_result_path, 'transition_matrix.json'), 'w') as f:
        json.dump(transition_data, f, indent=2)
    logger.info(f"Transition matrix saved to {os.path.join(test_result_path, 'transition_matrix.json')}")

    if use_wandb:
        # Log scalar metrics
        wandb.log({
            'test/precision': test_prec,
            'test/recall': test_rec,
            'test/f1': test_f1,
            'test/iou': test_iou,
            'test/accuracy': test_acc,
            'test/sek': test_sek
        })
        
        # Log transition matrix as heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(transition_matrix_pct, annot=True, fmt='.2f', cmap='Greens', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Percentage (%)'},
                    ax=ax)
        ax.set_xlabel('To (T2)')
        ax.set_ylabel('From (T1)')
        ax.set_title('From-To Transition Matrix (%)')
        plt.tight_layout()
        
        wandb.log({'test/transition_matrix': wandb.Image(fig)})
        plt.close(fig)
        
        wandb.finish()


if __name__ == '__main__':
    main()
