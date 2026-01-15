"""
Training script for change-detection-only models (CDMamba_change).
Simplified version without segmentation heads.
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
from torch.optim.lr_scheduler import CosineAnnealingLR

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


def log_first_batch_to_wandb(prefix, batch, seg_t1, seg_t2, change_pred):
    """Log first batch images to W&B for change-only model."""
    if wandb.run is None:
        return
    
    def _norm(img):
        """Normalize image tensor to [0, 255] uint8 for visualization."""
        img = img.detach().cpu()
        if img.min() < 0: 
            img = (img + 1.0) / 2.0  # From [-1, 1] to [0, 1]
        img = (img * 255.0).clamp(0, 255).byte()
        return img.permute(1, 2, 0).numpy()
    
    # Get first sample from batch
    A0 = batch['A'][0]  # T1 image
    B0 = batch['B'][0]  # T2 image
    
    # Ground truth change (derived from segmentation labels)
    chg_gt = derive_change_bin(seg_t1, seg_t2)[0].cpu().numpy() * 255
    
    # Predicted change
    if change_pred.size(1) == 2:
        # 2-class output: [no-change, change]
        change_probs = torch.softmax(change_pred[0], dim=0)[1].detach().cpu().numpy() * 255
        change_mask = torch.argmax(change_pred[0], dim=0).detach().cpu().numpy() * 255
    else:
        # Single-channel sigmoid output
        p = torch.sigmoid(change_pred[0, 0]).detach().cpu().numpy()
        change_probs = (p * 255)
        change_mask = ((p > 0.5).astype(np.uint8) * 255)
    
    # Log to wandb
    d = {
        f"{prefix}/input_T1": [wandb.Image(_norm(A0))],
        f"{prefix}/input_T2": [wandb.Image(_norm(B0))],
        f"{prefix}/gt_change": [wandb.Image(chg_gt)],
        f"{prefix}/pred_change_prob": [wandb.Image(change_probs)],
        f"{prefix}/pred_change_mask": [wandb.Image(change_mask)],
    }
    
    wandb.log(d)


# ----------------------------- main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file path')
    parser.add_argument('--phase', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--tag', type=str, default='', help='Experiment tag')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--max_train_batches', type=int, default=-1)
    parser.add_argument('--max_val_batches', type=int, default=-1)
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
    if args.phase == 'train':
        setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train', level=logging.INFO, screen=True)
        setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO)
    else:
        setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # W&B
    use_wandb = bool(args.wandb_project or opt.get('wandb', {}).get('project', ''))
    logger.info(f"W&B check: args.wandb_project={args.wandb_project}, config_project={opt.get('wandb', {}).get('project', '')}, use_wandb={use_wandb}")
    if use_wandb and args.phase == 'train':
        wandb.init(
            project=args.wandb_project or opt['wandb']['project'],
            name=exp_folder,
            config=opt
        )
        logger.info(f"✓ W&B initialized: project={wandb.run.project}, name={wandb.run.name}, run_id={wandb.run.id}")
    else:
        logger.info(f"✗ W&B not initialized (use_wandb={use_wandb}, phase={args.phase})")

    # ----------------------------- data ----------------------------- #
    logger.info("Creating datasets...")
    for phase in ['train', 'val', 'test']:
        if phase in opt['datasets']:
            dataset = Data.create_scd_dataset(opt['datasets'][phase], phase)
            logger.info(f'{phase} dataset length: {len(dataset)}')
            if phase == 'train':
                train_dataset = dataset
            elif phase == 'val':
                val_dataset = dataset
            elif phase == 'test':
                test_dataset = dataset

    # Dataloaders
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)

    if args.phase == 'train':
        train_loader = Data.create_cd_dataloader(train_dataset, opt['datasets']['train'], 'train', seed_worker, g)
        val_loader = Data.create_cd_dataloader(val_dataset, opt['datasets']['val'], 'val', seed_worker, g)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    else:
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

    # Resume
    if args.resume_path:
        logger.info(f"Loading checkpoint from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)
        if 'model' in ckpt:
            cd_model.load_state_dict(ckpt['model'])
        else:
            cd_model.load_state_dict(ckpt)

    # ----------------------------- loss ----------------------------- #
    loss_type = opt['model'].get('loss', 'ce')
    logger.info(f"Loss type: {loss_type}")

    if loss_type == 'ce':
        loss_fun = nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        loss_fun = DiceLoss(num_classes=opt['model'].get('n_classes', 2))
    elif loss_type == 'cedice':
        loss_fun = CEDiceLoss(num_classes=opt['model'].get('n_classes', 2))
    else:
        raise ValueError(f"Unsupported loss type for change-only model: {loss_type}")

    # ----------------------------- optimizer ----------------------------- #
    if args.phase == 'train':
        opt_type = opt['train']['optimizer']['type'].lower()
        lr = opt['train']['optimizer']['lr']

        if opt_type == 'adam':
            optimizer = optim.Adam(cd_model.parameters(), lr=lr, betas=(0.9, 0.999))
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(cd_model.parameters(), lr=lr, weight_decay=0.01)
        elif opt_type == 'sgd':
            optimizer = optim.SGD(cd_model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        # Scheduler
        scheduler = get_scheduler(optimizer, opt['train'])
        logger.info(f"Optimizer: {opt_type}, LR: {lr}, Scheduler: {opt['train']['sheduler']['lr_policy']}")

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        # Gradient accumulation
        accumulation_steps = opt['train'].get('grad_accum', 1)

    # ----------------------------- metrics ----------------------------- #
    # For change detection only, we don't need segmentation metrics

    # ----------------------------- training ----------------------------- #
    if args.phase == 'train':
        logger.info("=" * 60)
        logger.info("Starting training (change-only mode)")
        logger.info("=" * 60)

        n_epochs = opt['train']['n_epoch']
        val_freq = opt['train'].get('val_freq', 1)
        train_print_iter = opt['train'].get('train_print_iter', 100)
        val_print_iter = opt['train'].get('val_print_iter', 50)

        best_val_f1 = 0.0
        global_step = 0

        for epoch in range(1, n_epochs + 1):
            cd_model.train()

            # Training metrics
            train_loss_sum = 0.0
            train_batches = 0
            t_tp, t_fp, t_fn, t_tn = 0, 0, 0, 0

            _max_train = args.max_train_batches
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader

            for step, batch in enumerate(tqdm(_train_iter, total=_train_total, desc=f"Epoch {epoch}/{n_epochs}")):
                im1 = batch['A'].to(device, non_blocking=True)
                im2 = batch['B'].to(device, non_blocking=True)
                seg_t1 = batch['L1'].to(device).long()
                seg_t2 = batch['L2'].to(device).long()

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    # Change-only model outputs single tensor
                    change_pred = cd_model(im1, im2)  # [B, 2, H, W]

                    # Derive binary change from segmentation labels
                    change_bin = derive_change_bin(seg_t1, seg_t2)  # [B, H, W]

                    # Compute loss
                    loss = loss_fun(change_pred, change_bin)
                    loss_scaled = loss / accumulation_steps

                scaler.scale(loss_scaled).backward()

                do_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == _train_total)
                if do_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                # Metrics
                with torch.no_grad():
                    if change_pred.size(1) == 2:
                        chg_mask = torch.argmax(change_pred, dim=1)
                    else:
                        chg_mask = (torch.sigmoid(change_pred[:, 0]) > args.change_threshold).long()

                    pr_np, gt_np = chg_mask.cpu().numpy(), change_bin.cpu().numpy()
                    tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                    fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                    fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                    tn = np.logical_and(pr_np == 0, gt_np == 0).sum()

                    t_tp += tp; t_fp += fp; t_fn += fn; t_tn += tn

                train_loss_sum += float(loss.item())
                train_batches += 1

                # Log first batch images to wandb
                if step == 0 and use_wandb:
                    log_first_batch_to_wandb("train", batch, seg_t1, seg_t2, change_pred)

                # Logging
                if (step + 1) % train_print_iter == 0:
                    avg_loss = train_loss_sum / train_batches
                    prec = t_tp / (t_tp + t_fp + 1e-8)
                    rec = t_tp / (t_tp + t_fn + 1e-8)
                    f1 = 2 * prec * rec / (prec + rec + 1e-8)
                    iou = t_tp / (t_tp + t_fp + t_fn + 1e-8)

                    logger.info(f"[Epoch {epoch}/{n_epochs}] Step {step+1}/{_train_total} | Loss: {avg_loss:.4f} | "
                                f"Change - Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}")

                    if use_wandb:
                        metrics = {
                            'train/loss': avg_loss,
                            'train/change_prec': prec,
                            'train/change_rec': rec,
                            'train/change_f1': f1,
                            'train/change_iou': iou,
                            'train/lr': optimizer.param_groups[0]['lr']
                        }
                        wandb.log(metrics, step=global_step)
                        if step + 1 == train_print_iter:  # Log once at first print
                            logger.info(f"✓ Logged to W&B: {list(metrics.keys())}")
                    elif step + 1 == train_print_iter:
                        logger.warning(f"✗ W&B logging skipped: use_wandb={use_wandb}")

            # Epoch summary
            avg_train_loss = train_loss_sum / train_batches
            train_prec = t_tp / (t_tp + t_fp + 1e-8)
            train_rec = t_tp / (t_tp + t_fn + 1e-8)
            train_f1 = 2 * train_prec * train_rec / (train_prec + train_rec + 1e-8)
            train_iou = t_tp / (t_tp + t_fp + t_fn + 1e-8)

            logger.info(f"\n[Epoch {epoch}/{n_epochs}] Train Summary:")
            logger.info(f"  Loss: {avg_train_loss:.4f}")
            logger.info(f"  Change - Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}, IoU: {train_iou:.4f}\n")

            # Log epoch summary to W&B
            if use_wandb:
                wandb.log({
                    'train/epoch_loss': avg_train_loss,
                    'train/epoch_prec': train_prec,
                    'train/epoch_rec': train_rec,
                    'train/epoch_f1': train_f1,
                    'train/epoch_iou': train_iou,
                    'epoch': epoch
                }, step=global_step)
                logger.info(f"✓ Logged train epoch summary to W&B")

            # ----------------------------- validation ----------------------------- #
            if epoch % val_freq == 0:
                cd_model.eval()
                logger.info(f"Running validation...")

                v_loss_sum = 0.0
                v_batches = 0
                v_tp, v_fp, v_fn, v_tn = 0, 0, 0, 0

                _max_val = args.max_val_batches
                _val_total = min(len(val_loader), _max_val) if _max_val > 0 else len(val_loader)
                _val_iter = islice(val_loader, _max_val) if _max_val > 0 else val_loader

                with torch.no_grad():
                    for vstep, vbatch in enumerate(tqdm(_val_iter, total=_val_total, desc=f"Val {epoch}")):
                        v1 = vbatch['A'].to(device)
                        v2 = vbatch['B'].to(device)
                        y1 = vbatch['L1'].to(device).long()
                        y2 = vbatch['L2'].to(device).long()

                        v_change = cd_model(v1, v2)
                        change_gt = derive_change_bin(y1, y2)

                        v_loss = loss_fun(v_change, change_gt)
                        v_loss_sum += float(v_loss.item())
                        v_batches += 1

                        # Metrics
                        if v_change.size(1) == 2:
                            chg_mask = torch.argmax(v_change, dim=1)
                        else:
                            chg_mask = (torch.sigmoid(v_change[:, 0]) > args.change_threshold).long()

                        pr_np, gt_np = chg_mask.cpu().numpy(), change_gt.cpu().numpy()
                        tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                        fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                        fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                        tn = np.logical_and(pr_np == 0, gt_np == 0).sum()

                        v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn

                        # Log first batch images to wandb
                        if vstep == 0 and use_wandb:
                            log_first_batch_to_wandb("val", vbatch, y1, y2, v_change)

                avg_val_loss = v_loss_sum / v_batches
                val_prec = v_tp / (v_tp + v_fp + 1e-8)
                val_rec = v_tp / (v_tp + v_fn + 1e-8)
                val_f1 = 2 * val_prec * val_rec / (val_prec + val_rec + 1e-8)
                val_iou = v_tp / (v_tp + v_fp + v_fn + 1e-8)

                logger.info(f"\n[Epoch {epoch}/{n_epochs}] Validation Summary:")
                logger.info(f"  Loss: {avg_val_loss:.4f}")
                logger.info(f"  Change - Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, IoU: {val_iou:.4f}\n")

                if use_wandb:
                    wandb.log({
                        'val/loss': avg_val_loss,
                        'val/change_prec': val_prec,
                        'val/change_rec': val_rec,
                        'val/change_f1': val_f1,
                        'val/change_iou': val_iou,
                        'epoch': epoch
                    }, step=global_step)
                    logger.info(f"✓ Logged validation metrics to W&B")

                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    save_network(opt, epoch, cd_model, optimizer, is_best_model=True)
                    logger.info(f"✓ Saved best model (F1: {best_val_f1:.4f})")

            # Save epoch checkpoint
            if epoch % 10 == 0:
                save_network(opt, epoch, cd_model, optimizer, is_best_model=False)

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        logger.info("=" * 60)
        logger.info(f"Training complete! Best val F1: {best_val_f1:.4f}")
        logger.info("=" * 60)

    # ----------------------------- testing ----------------------------- #
    else:
        logger.info("=" * 60)
        logger.info("Starting testing (change-only mode)")
        logger.info("=" * 60)

        cd_model.eval()

        test_tp, test_fp, test_fn, test_tn = 0, 0, 0, 0

        _max_test = args.max_test_batches
        _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
        _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader

        with torch.no_grad():
            for tstep, tb in enumerate(tqdm(_test_iter, total=_test_total, desc="Test")):
                ti1 = tb['A'].to(device)
                ti2 = tb['B'].to(device)
                y1 = tb['L1'].to(device).long()
                y2 = tb['L2'].to(device).long()

                chg = cd_model(ti1, ti2)
                change_gt = derive_change_bin(y1, y2)

                if chg.size(1) == 2:
                    cmask = torch.argmax(chg, dim=1)
                else:
                    cmask = (torch.sigmoid(chg[:, 0]) > args.change_threshold).long()

                pr_np, gt_np = cmask.cpu().numpy(), change_gt.cpu().numpy()
                tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                tn = np.logical_and(pr_np == 0, gt_np == 0).sum()

                test_tp += tp; test_fp += fp; test_fn += fn; test_tn += tn

        # Final metrics
        test_prec = test_tp / (test_tp + test_fp + 1e-8)
        test_rec = test_tp / (test_tp + test_fn + 1e-8)
        test_f1 = 2 * test_prec * test_rec / (test_prec + test_rec + 1e-8)
        test_iou = test_tp / (test_tp + test_fp + test_fn + 1e-8)
        test_acc = (test_tp + test_tn) / (test_tp + test_tn + test_fp + test_fn + 1e-8)

        logger.info("=" * 60)
        logger.info("Test Results:")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall:    {test_rec:.4f}")
        logger.info(f"  F1-Score:  {test_f1:.4f}")
        logger.info(f"  IoU:       {test_iou:.4f}")
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info("=" * 60)

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
