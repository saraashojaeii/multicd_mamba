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
import matplotlib.pyplot as plt  # noqa: F401  (kept for parity; used by utils in some setups)
from core.utils import *
from core.logger import parse as parse_cfg
from core.logger import setup_logger, dict2str, dict_to_nonedict
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler, save_network  # keep compatibility
from models.loss import *            # noqa: F401,F403
from models.loss import MultiClassCDLoss  # explicit import
from core.metrics import compute_semantic_metrics_on_changed, compute_per_class_metrics, compute_transition_metrics


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


def unpack_outputs(outputs):
    """
    Accepts:
      (seg1, seg2)
      (seg1, seg2, change)
      (seg1, seg2, change, aux)
    Returns: seg1, seg2, change (or None), aux (dict or None)
    """
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
        aux = outputs.get('aux', None)
    else:
        raise ValueError(f"Unexpected output type: {type(outputs)}")
    return seg1, seg2, change, aux


def derive_change_bin(seg_t1: torch.Tensor, seg_t2: torch.Tensor) -> torch.Tensor:
    """Binary change (0/1) from label maps [B,H,W]."""
    return (seg_t1 != seg_t2).long()


def safe_to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.uint8)
    return np.squeeze(arr)


def log_first_batch_to_wandb(prefix, batch, seg_t1, seg_t2, seg_logits_t1, seg_logits_t2, change_pred, num_classes):
    def _norm(img):
        img = img.detach().cpu()
        if img.min() < 0: img = (img + 1.0) / 2.0
        img = (img * 255.0).clamp(0, 255).byte()
        return img.permute(1, 2, 0).numpy()

    A0 = batch['A'][0]
    B0 = batch['B'][0]
    pred1 = torch.argmax(seg_logits_t1[0], dim=0)
    pred2 = torch.argmax(seg_logits_t2[0], dim=0)

    d = {
        f"{prefix}/input_T1": [wandb.Image(_norm(A0))],
        f"{prefix}/input_T2": [wandb.Image(_norm(B0))],
        f"{prefix}/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=num_classes))],
        f"{prefix}/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=num_classes))],
        f"{prefix}/pred_seg_t1": [wandb.Image(create_color_mask(pred1, num_classes=num_classes))],
        f"{prefix}/pred_seg_t2": [wandb.Image(create_color_mask(pred2, num_classes=num_classes))],
    }

    # probability maps
    p1 = torch.softmax(seg_logits_t1[0], dim=0).max(dim=0)[0].detach().cpu().numpy()
    p2 = torch.softmax(seg_logits_t2[0], dim=0).max(dim=0)[0].detach().cpu().numpy()
    d[f"{prefix}/pred_seg_t1_prob"] = [wandb.Image(p1)]
    d[f"{prefix}/pred_seg_t2_prob"] = [wandb.Image(p2)]

    # change GT (derived) and prediction (if any)
    chg_gt = derive_change_bin(seg_t1, seg_t2)[0].cpu().numpy() * 255
    d[f"{prefix}/gt_change"] = [wandb.Image(chg_gt)]
    if change_pred is not None:
        if change_pred.size(1) == 2:
            change_probs = torch.softmax(change_pred[0], dim=0)[1].detach().cpu().numpy() * 255
            change_mask = torch.argmax(change_pred[0], dim=0).detach().cpu().numpy() * 255
        else:
            p = torch.sigmoid(change_pred[0, 0]).detach().cpu().numpy()
            change_probs = (p * 255)
            change_mask = ((p > 0.5).astype(np.uint8) * 255)
        d[f"{prefix}/pred_change_prob"] = [wandb.Image(change_probs)]
        d[f"{prefix}/pred_change_mask"] = [wandb.Image(change_mask)]

    wandb.log(d)


# ----------------------------- main ----------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='JSON config file')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='', help='(for run naming only)')
    parser.add_argument('--dataset', type=str, default='', help='(for run naming only)')
    parser.add_argument('--tag', type=str, default='', help='(for run naming only)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max_train_batches', type=int, default=0)
    parser.add_argument('--max_val_batches', type=int, default=0)
    parser.add_argument('--max_test_batches', type=int, default=0)
    parser.add_argument('--change_threshold', type=float, default=0.2)
    parser.add_argument('--resume_epoch', type=int, default=None, help='Resume training from this epoch index (loads matching checkpoints).')
    parser.add_argument('--resume_model_path', type=str, default=None, help='Existing stamped experiment folder to reuse (log/result/checkpoint).')
    parser.add_argument('--exp_folder', type=str, default=None, help='Experiment folder name (if not provided, auto-generated from dataset/tag/seed).')
    args = parser.parse_args()

    # Parse JSON config
    opt = parse_cfg(args)
    opt = dict_to_nonedict(opt)

    # Build or reuse a stamped run folder name
    if args.exp_folder:
        exp_folder = args.exp_folder
    else:
        exp_timestamp = datetime.now().strftime('%m%d_%H')
        parts = [x for x in [args.dataset, args.tag, f"seed{args.seed}" if args.seed is not None else ""] if x]
        suffix = "_".join(parts)
        exp_folder = f"{suffix}_{exp_timestamp}" if suffix else exp_timestamp
    make_stamped_dirs(opt, exp_folder)

    set_all_seeds(args.seed)

    # Logging
    setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train', level=logging.INFO, screen=True)
    setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))
    logger.info(f"[run] phase={args.phase} tag={args.tag} seed={args.seed} exp_folder={opt['path_cd']['exp_folder']}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # WANDB
    if opt.get('wandb') and opt['wandb'].get('project'):
        wandb.init(project=opt['wandb']['project'], config=opt, name=opt['path_cd']['exp_folder'])
    else:
        wandb.init(mode="disabled")

    # Dataloaders
    train_loader = val_loader = test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            train_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase='train')
            
            # Optional: Use balanced sampler for training
            use_balanced_sampler = opt.get('train', {}).get('use_balanced_sampler', False)
            if use_balanced_sampler:
                from data.balanced_sampler import BalancedChangeSampler
                sampler_cfg = opt.get('train', {}).get('balanced_sampler', {})
                train_sampler = BalancedChangeSampler(
                    train_set,
                    change_threshold=sampler_cfg.get('change_threshold', 0.01),
                    rare_classes=sampler_cfg.get('rare_classes', [3, 5]),  # water, playground
                    oversample_factor=sampler_cfg.get('oversample_factor', 2.0),
                    precompute_stats=sampler_cfg.get('precompute_stats', True),
                    max_precompute=sampler_cfg.get('max_precompute', 1000),
                    stats_file=sampler_cfg.get('stats_file', None),  # Load precomputed stats if provided
                )
                logger.info(f"Using BalancedChangeSampler with change_threshold={sampler_cfg.get('change_threshold', 0.01)}")
                # Create dataloader with custom sampler (note: shuffle must be False when using sampler)
                train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=dataset_opt['batch_size'],
                    sampler=train_sampler,
                    num_workers=dataset_opt.get('num_workers', 4),
                    pin_memory=True,
                )
            else:
                train_loader = Data.create_cd_dataloader(train_set, dataset_opt, 'train', seed_worker=None, g=None)
            
            opt['len_train_dataloader'] = len(train_loader)
        elif phase == 'val' and args.phase != 'test':
            val_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase='val')
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, 'val', seed_worker=None, g=None)
            opt['len_val_dataloader'] = len(val_loader)
        elif phase == 'test':
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase='test')
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, 'test', seed_worker=None, g=None)
            opt['len_test_dataloader'] = len(test_loader)

    # Class weights
    num_classes = int(opt['model']['n_classes'])
    ignore_index = int(opt.get('train', {}).get('ignore_index', 255))
    if train_loader is not None:
        counts = estimate_class_counts(train_loader, num_classes=num_classes, ignore_index=ignore_index, max_batches=200)
        ce_weights = compute_class_weights(counts, method="median_frequency").to(device)
        dice_weights = ce_weights.clone()
        
        # Compute transition weights for rebalancing rare transitions
        logger.info("Computing transition matrix from training data...")
        from core.utils import estimate_transition_matrix, compute_transition_weights
        transition_matrix = estimate_transition_matrix(train_loader, num_classes=num_classes, 
                                                       ignore_index=ignore_index, max_batches=200)
        transition_weights = compute_transition_weights(transition_matrix, method="inverse_frequency").to(device)
        logger.info(f"Transition matrix computed. Total transitions: {transition_matrix.sum()}")
        logger.info(f"Transition weights range: [{transition_weights.min():.3f}, {transition_weights.max():.3f}]")
    else:
        ce_weights = dice_weights = transition_weights = None

    # Model
    cd_model = Model.create_CD_model(opt)
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight, gain=0.1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if hasattr(m, 'weight') and m.weight is not None: nn.init.constant_(m.weight, 1)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    cd_model.apply(init_weights)
    cd_model.to(device)

    total_params = sum(p.numel() for p in cd_model.parameters())
    trainable_params = sum(p.numel() for p in cd_model.parameters() if p.requires_grad)
    logger.info(f'Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)')

    # GFLOPs (CUDA + torchinfo)
    if device.type == 'cuda' and summary is not None:
        try:
            # dual input by default for CD model
            input_size = [(2, opt['model']['in_nc'], 512, 512), (2, opt['model']['in_nc'], 512, 512)]
            model_summary = summary(cd_model, input_size=input_size, device=device, verbose=0,
                                    col_names=["input_size", "output_size", "num_params", "mult_adds"])
            gflops = model_summary.total_mult_adds / 1e9
            logger.info(f"Model GFLOPs (512²): {gflops:.2f}")
            print(model_summary)
        except Exception as e:
            logger.warning(f'GFLOPs summary failed: {e}')

    if torch.cuda.is_available():
        logger.info(f'Model on device: {next(cd_model.parameters()).device}')

    if hasattr(cd_model, 'gradient_checkpointing_enable'):
        cd_model.gradient_checkpointing_enable()

    # Losses
    loss_type = opt['model']['loss']
    num_classes = int(opt['model']['n_classes'])
    if loss_type == 'ce_dice':
        loss_fun = CEDiceLoss(num_classes=num_classes).to(device)
        loss_fun_change = CEDiceLoss(num_classes=2).to(device)
    elif loss_type == 'ce':
        loss_fun = cross_entropy_loss_fn
        loss_fun_change = cross_entropy_loss_fn
    elif loss_type == 'dice':
        loss_fun = DiceOnlyLoss(num_classes=num_classes).to(device)
        loss_fun_change = DiceOnlyLoss(num_classes=2).to(device)
    elif loss_type == 'extended_triplet':
        # Per-pixel segmentation loss for change-aware weighting
        cfg = opt['model'].get('extended_triplet', {})
        seg_loss_type = cfg.get('seg_loss', 'weighted_ce_dice')  # 'weighted_ce_dice' or 'focal_ce_dice'
        
        if seg_loss_type == 'focal_ce_dice':
            # Focal CE + Dice (for extreme class imbalance)
            base_seg = FocalCEDicePerPixel(
                num_classes=num_classes,
                class_weights=ce_weights,
                gamma=cfg.get('focal_gamma', 2.0),
                ignore_index=ignore_index,
                lambda_focal=0.5,
                lambda_dice=0.5,
            )
            logger.info(f"Using FocalCEDicePerPixel with gamma={cfg.get('focal_gamma', 2.0)}")
        else:
            # Weighted CE + Dice (recommended for most cases)
            base_seg = WeightedCEDicePerPixel(
                num_classes=num_classes,
                class_weights=ce_weights,
                ignore_index=ignore_index,
                lambda_ce=0.5,
                lambda_dice=0.5,
                label_smoothing=0.0,
            )
            logger.info("Using WeightedCEDicePerPixel (recommended for rare classes)")
        
        loss_fun = TripletChangeSegLoss(
            seg_loss_fn=base_seg,
            lambda_seg=cfg.get('lambda_seg', 1.0),
            lambda_cd=cfg.get('lambda_cd', 1.0),
            lambda_unch=cfg.get('lambda_unch', 0.2),
            lambda_ch=cfg.get('lambda_ch', 0.2),
            lambda_cpl=cfg.get('lambda_cpl', 0.5),
            T=cfg.get('T', 4.0),
            margin=cfg.get('margin', 0.3),
            boost=cfg.get('boost', 5.0),
            ignore_index=ignore_index,
            transition_weights=transition_weights,  # Pass transition weights for rebalancing
        ).to(device)
        loss_fun_change = loss_fun
    elif loss_type == 'seg_loss':
        loss_fun = ComboSegLoss(
            class_weights_ce=ce_weights,
            class_weights_dice=dice_weights,
            ignore_index=ignore_index,
            lambda_ce=0.5, lambda_dice=0.5,
            label_smoothing=0.0, include_bg=True,
        ).to(device)
        loss_fun_change = loss_fun
    elif loss_type == 'multi_class_cd':
        lw = opt['model'].get('loss_weights', {'seg_t1': 1.0, 'seg_t2': 1.0, 'change': 1.0})
        loss_fun = MultiClassCDLoss(num_classes=num_classes, seg_loss="cedice", change_loss="ce", loss_weights=lw)
        loss_fun_change = loss_fun
        if isinstance(loss_fun, nn.Module): loss_fun.to(device)
    else:
        raise ValueError(f"Unsupported loss: {loss_type}")

    # Optimizer & Scheduler
    opt_cfg = opt['train']["optimizer"]
    if opt_cfg["type"] == 'adam':
        optimizer = optim.Adam(cd_model.parameters(), lr=opt_cfg["lr"],
                               betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)))
    elif opt_cfg["type"] == 'adamW':
        optimizer = optim.AdamW(cd_model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 1e-4))
    elif opt_cfg["type"] == 'sgd':
        optimizer = optim.SGD(cd_model.parameters(), lr=opt_cfg["lr"], momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg['type']}")

    scheduler = CosineAnnealingLR(optimizer, T_max=opt['train']['n_epoch'], eta_min=opt_cfg.get("eta_min", 1e-6))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    # ----------------------------- TRAIN ----------------------------- #
    if args.phase == 'train':
        best_mF1 = 0.0
        metric_seg = ConfuseMatrixMeter(n_class=num_classes)
        n_epochs = opt['train']['n_epoch']
        accumulation_steps = int(opt['train'].get('grad_accum', 2))

        # Resume support
        start_epoch = 0
        if args.resume_epoch is not None:
            ckpt_dir = opt['path_cd']['checkpoint']
            gen_path = os.path.join(ckpt_dir, f'cd_model_E{args.resume_epoch}_gen.pth')
            opt_path = os.path.join(ckpt_dir, f'cd_model_E{args.resume_epoch}_opt.pth')
            if os.path.exists(gen_path):
                state = torch.load(gen_path, map_location=device)
                cd_model.load_state_dict(state, strict=True)
                logging.getLogger('base').info(f"[resume] Loaded model weights from {gen_path}")
            else:
                logging.getLogger('base').warning(f"[resume] Model checkpoint not found: {gen_path}")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path, map_location=device)
                if isinstance(opt_state, dict) and 'optimizer' in opt_state:
                    optimizer.load_state_dict(opt_state['optimizer'])
                    logging.getLogger('base').info(f"[resume] Loaded optimizer state from {opt_path}")
            else:
                logging.getLogger('base').warning(f"[resume] Optimizer checkpoint not found: {opt_path}")
            start_epoch = int(args.resume_epoch)
            try:
                scheduler.last_epoch = start_epoch - 1
            except Exception:
                pass

        for epoch in range(start_epoch, n_epochs):
            cd_model.train()
            metric_seg.clear()
            epoch_loss = 0.0

            _max_train = args.max_train_batches or 0
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader

            optimizer.zero_grad(set_to_none=True)

            # running change metrics 
            run_tp = run_fp = run_fn = run_tn = 0

            for step, batch in enumerate(tqdm(_train_iter, total=_train_total, desc=f"Train {epoch}/{n_epochs}")):
                im1 = batch['A'].to(device, non_blocking=True)
                im2 = batch['B'].to(device, non_blocking=True)
                seg_t1 = batch['L1'].to(device).long()
                seg_t2 = batch['L2'].to(device).long()

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = cd_model(im1, im2)
                    seg_logits_t1, seg_logits_t2, change_pred, _ = unpack_outputs(outputs)

                    # ----- loss -----
                    if loss_type == 'multi_class_cd':
                        preds = (seg_logits_t1, seg_logits_t2, change_pred)
                        change_bin = derive_change_bin(seg_t1, seg_t2)
                        targets = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change_bin}
                        loss, loss_components = loss_fun(preds, targets)
                    elif loss_type == 'extended_triplet':
                        if change_pred is None:
                            # create 1-ch dummy to satisfy the interface
                            b, c, h, w = seg_logits_t1.shape
                            change_pred = torch.zeros((b, 1, h, w), device=seg_logits_t1.device)
                        change_bin = derive_change_bin(seg_t1, seg_t2)
                        preds = (seg_logits_t1, seg_logits_t2, change_pred[:, 1:2] if change_pred.size(1) == 2 else change_pred)
                        targets = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change_bin}
                        loss, _ = loss_fun(preds, targets)
                    else:
                        # separate seg + change
                        l1 = loss_fun(seg_logits_t1, seg_t1) if isinstance(loss_fun, nn.Module) else loss_fun(seg_logits_t1, seg_t1)
                        l2 = loss_fun(seg_logits_t2, seg_t2) if isinstance(loss_fun, nn.Module) else loss_fun(seg_logits_t2, seg_t2)
                        if change_pred is not None:
                            change_bin = derive_change_bin(seg_t1, seg_t2)
                            lc = loss_fun_change(change_pred, change_bin) if isinstance(loss_fun_change, nn.Module) else loss_fun_change(change_pred, change_bin)
                            loss = l1 + l2 + lc
                        else:
                            loss = l1 + l2

                    loss_scaled = loss / accumulation_steps

                scaler.scale(loss_scaled).backward()

                do_step = ((step + 1) % accumulation_steps == 0) or ((step + 1) == _train_total)
                if do_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # metrics
                with torch.no_grad():
                    pred1 = torch.argmax(seg_logits_t1, dim=1)
                    pred2 = torch.argmax(seg_logits_t2, dim=1)
                    metric_seg.update_cm(pr=safe_to_numpy_uint8(pred1), gt=safe_to_numpy_uint8(seg_t1))
                    metric_seg.update_cm(pr=safe_to_numpy_uint8(pred2), gt=safe_to_numpy_uint8(seg_t2))

                    if change_pred is not None:
                        if change_pred.size(1) == 2:
                            chg_mask = torch.argmax(change_pred, dim=1)
                        else:
                            chg_mask = (torch.sigmoid(change_pred[:, 0]) > args.change_threshold).long()
                        gt_bin = derive_change_bin(seg_t1, seg_t2)
                        pr_np, gt_np = chg_mask.cpu().numpy(), gt_bin.cpu().numpy()
                        tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                        fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                        fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                        tn = np.logical_and(pr_np == 0, gt_np == 0).sum()
                        run_tp += tp; run_fp += fp; run_fn += fn; run_tn += tn

                epoch_loss += float(loss_scaled.item())

                # first-batch visuals
                if step == 0:
                    log_first_batch_to_wandb("train", batch, seg_t1, seg_t2, seg_logits_t1, seg_logits_t2, change_pred, num_classes)

            # epoch-end: seg metrics
            seg_scores = metric_seg.get_scores()
            train_epoch_mf1 = seg_scores['mf1']
            train_epoch_miou = seg_scores['miou']
            train_epoch_acc = seg_scores['acc']
            
            # Compute additional metrics on changed pixels only (aggregate from last batch for efficiency)
            # Note: For full accuracy, we'd need to accumulate across all batches, but this gives a good estimate
            if step > 0:  # Use last batch as representative
                pred1_np = pred1.cpu().numpy()
                pred2_np = pred2.cpu().numpy()
                seg_t1_np = seg_t1.cpu().numpy()
                seg_t2_np = seg_t2.cpu().numpy()
                
                # Metrics on changed pixels only
                changed_metrics = compute_semantic_metrics_on_changed(
                    pred1_np, pred2_np, seg_t1_np, seg_t2_np, num_classes, ignore_index
                )
                
                # Per-class metrics
                per_class_metrics = compute_per_class_metrics(
                    pred1_np, pred2_np, seg_t1_np, seg_t2_np, num_classes, ignore_index
                )
                
                # Top transition metrics (nvg_surf=1, low_veg=0, building=4)
                top_transitions = [(1, 4), (0, 1), (1, 0), (0, 4)]  # nvg_surf→building, low_veg↔nvg_surf, low_veg→building
                transition_metrics = compute_transition_metrics(
                    pred1_np, pred2_np, seg_t1_np, seg_t2_np, top_transitions, num_classes, ignore_index
                )
            else:
                changed_metrics = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0}
                per_class_metrics = {'iou_per_class': [0.0]*num_classes, 'f1_per_class': [0.0]*num_classes}
                transition_metrics = {}

            # e-e: change metrics
            if (run_tp + run_fp + run_fn) > 0:
                chg_f1 = 2 * run_tp / max(2 * run_tp + run_fp + run_fn, 1e-8)
                chg_iou = run_tp / max(run_tp + run_fp + run_fn, 1e-8)
                chg_acc = (run_tp + run_tn) / max(run_tp + run_tn + run_fp + run_fn, 1e-8)
            else:
                chg_f1 = chg_iou = chg_acc = 0.0

            avg_epoch_loss = epoch_loss / max(1, _train_total)
            
            # Build comprehensive metrics dict
            train_metrics = {
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch_mF1_seg': train_epoch_mf1,
                'train/epoch_mIoU_seg': train_epoch_miou,
                'train/epoch_OA_seg': train_epoch_acc,
                'train/epoch_change_f1': chg_f1,
                'train/epoch_change_iou': chg_iou,
                'train/epoch_change_acc': chg_acc,
                # Metrics on changed pixels only
                'train/changed_pixels_iou': changed_metrics['iou'],
                'train/changed_pixels_f1': changed_metrics['f1'],
                'train/changed_pixels_acc': changed_metrics['accuracy'],
                # Per-class metrics (key classes: building=4, nvg_surf=1, water=3, playground=5)
                'train/class_building_iou': per_class_metrics['iou_per_class'][4] if len(per_class_metrics['iou_per_class']) > 4 else 0.0,
                'train/class_building_f1': per_class_metrics['f1_per_class'][4] if len(per_class_metrics['f1_per_class']) > 4 else 0.0,
                'train/class_nvg_surf_iou': per_class_metrics['iou_per_class'][1] if len(per_class_metrics['iou_per_class']) > 1 else 0.0,
                'train/class_nvg_surf_f1': per_class_metrics['f1_per_class'][1] if len(per_class_metrics['f1_per_class']) > 1 else 0.0,
                'train/class_water_iou': per_class_metrics['iou_per_class'][3] if len(per_class_metrics['iou_per_class']) > 3 else 0.0,
                'train/class_water_f1': per_class_metrics['f1_per_class'][3] if len(per_class_metrics['f1_per_class']) > 3 else 0.0,
                'train/class_playground_iou': per_class_metrics['iou_per_class'][5] if len(per_class_metrics['iou_per_class']) > 5 else 0.0,
                'train/class_playground_f1': per_class_metrics['f1_per_class'][5] if len(per_class_metrics['f1_per_class']) > 5 else 0.0,
                'epoch': epoch
            }
            
            # Add transition metrics
            for trans_key, trans_val in transition_metrics.items():
                train_metrics[f'train/transition_{trans_key}_acc'] = trans_val['accuracy']
                train_metrics[f'train/transition_{trans_key}_count'] = trans_val['count']
            
            wandb.log(train_metrics)
            logging.getLogger('base').info(
                f"[Train] ep {epoch} loss {avg_epoch_loss:.5f} | seg mF1 {train_epoch_mf1:.4f} | chg F1 {chg_f1:.4f}"
            )

            # ----------------- VALIDATION ----------------- #
            if val_loader is not None:
                cd_model.eval()
                val_metric = ConfuseMatrixMeter(n_class=num_classes)
                v_loss_sum, v_batches = 0.0, 0
                v_tp = v_fp = v_fn = v_tn = 0

                _max_val = args.max_val_batches or 0
                _val_total = min(len(val_loader), _max_val) if _max_val > 0 else len(val_loader)
                _val_iter = islice(val_loader, _max_val) if _max_val > 0 else val_loader

                with torch.no_grad():
                    for vstep, vbatch in enumerate(tqdm(_val_iter, total=_val_total, desc=f"Val {epoch}")):
                        v1 = vbatch['A'].to(device)
                        v2 = vbatch['B'].to(device)
                        y1 = vbatch['L1'].to(device).long()
                        y2 = vbatch['L2'].to(device).long()

                        vouts = cd_model(v1, v2)
                        v_seg1, v_seg2, v_change, _ = unpack_outputs(vouts)

                        # loss
                        if loss_type == 'multi_class_cd':
                            v_loss, _ = loss_fun((v_seg1, v_seg2, v_change), {'seg_t1': y1, 'seg_t2': y2, 'change': derive_change_bin(y1, y2)})
                        elif loss_type == 'extended_triplet':
                            v_change_1ch = v_change[:, 1:2] if (v_change is not None and v_change.size(1) == 2) else (v_change if v_change is not None else torch.zeros((v_seg1.size(0), 1, v_seg1.size(2), v_seg1.size(3)), device=v_seg1.device))
                            v_loss, _ = loss_fun((v_seg1, v_seg2, v_change_1ch), {'seg_t1': y1, 'seg_t2': y2, 'change': derive_change_bin(y1, y2)})
                        else:
                            l1 = loss_fun(v_seg1, y1) if isinstance(loss_fun, nn.Module) else loss_fun(v_seg1, y1)
                            l2 = loss_fun(v_seg2, y2) if isinstance(loss_fun, nn.Module) else loss_fun(v_seg2, y2)
                            if v_change is not None:
                                v_loss = l1 + l2 + (loss_fun_change(v_change, derive_change_bin(y1, y2)) if isinstance(loss_fun_change, nn.Module) else loss_fun_change(v_change, derive_change_bin(y1, y2)))
                            else:
                                v_loss = l1 + l2

                        v_loss_sum += float(v_loss.item()); v_batches += 1

                        # seg metrics
                        p1 = torch.argmax(v_seg1, dim=1)
                        p2 = torch.argmax(v_seg2, dim=1)
                        val_metric.update_cm(pr=safe_to_numpy_uint8(p1), gt=safe_to_numpy_uint8(y1))
                        val_metric.update_cm(pr=safe_to_numpy_uint8(p2), gt=safe_to_numpy_uint8(y2))

                        # change metrics
                        if v_change is not None:
                            if v_change.size(1) == 2:
                                chg_mask = torch.argmax(v_change, dim=1)
                            else:
                                chg_mask = (torch.sigmoid(v_change[:, 0]) > args.change_threshold).long()
                            gt = derive_change_bin(y1, y2)
                            pr_np, gt_np = chg_mask.cpu().numpy(), gt.cpu().numpy()
                            tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                            fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                            fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                            tn = np.logical_and(pr_np == 0, gt_np == 0).sum()
                            v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn

                        # first val batch visuals
                        if vstep == 0:
                            log_first_batch_to_wandb("val", vbatch, y1, y2, v_seg1, v_seg2, v_change, num_classes)

                val_scores = val_metric.get_scores()
                val_mf1 = val_scores['mf1']
                val_miou = val_scores['miou']
                val_acc = val_scores['acc']
                
                # Compute additional metrics on changed pixels only (aggregate from last batch)
                if vstep > 0:
                    p1_np = p1.cpu().numpy()
                    p2_np = p2.cpu().numpy()
                    y1_np = y1.cpu().numpy()
                    y2_np = y2.cpu().numpy()
                    
                    val_changed_metrics = compute_semantic_metrics_on_changed(
                        p1_np, p2_np, y1_np, y2_np, num_classes, ignore_index
                    )
                    
                    val_per_class_metrics = compute_per_class_metrics(
                        p1_np, p2_np, y1_np, y2_np, num_classes, ignore_index
                    )
                    
                    top_transitions = [(1, 4), (0, 1), (1, 0), (0, 4)]
                    val_transition_metrics = compute_transition_metrics(
                        p1_np, p2_np, y1_np, y2_np, top_transitions, num_classes, ignore_index
                    )
                else:
                    val_changed_metrics = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0}
                    val_per_class_metrics = {'iou_per_class': [0.0]*num_classes, 'f1_per_class': [0.0]*num_classes}
                    val_transition_metrics = {}

                # Change detection metrics
                if (v_tp + v_fp + v_fn) > 0:
                    v_chg_prec = v_tp / max(v_tp + v_fp, 1e-8)
                    v_chg_rec = v_tp / max(v_tp + v_fn, 1e-8)
                    v_chg_f1 = 2 * v_tp / max(2 * v_tp + v_fp + v_fn, 1e-8)
                    v_chg_iou = v_tp / max(v_tp + v_fp + v_fn, 1e-8)
                    v_chg_acc = (v_tp + v_tn) / max(v_tp + v_tn + v_fp + v_fn, 1e-8)
                else:
                    v_chg_prec = v_chg_rec = v_chg_f1 = v_chg_iou = v_chg_acc = 0.0

                val_loss_avg = v_loss_sum / max(1, v_batches)
                
                # Comprehensive W&B logging
                wandb_metrics = {
                    'val/epoch_loss': val_loss_avg,
                    'val/epoch_mF1': val_mf1,
                    'val/epoch_mIoU': val_miou,
                    'val/epoch_OA': val_acc,
                    'val/epoch_change_prec': v_chg_prec,
                    'val/epoch_change_rec': v_chg_rec,
                    'val/epoch_change_f1': v_chg_f1,
                    'val/epoch_change_iou': v_chg_iou,
                    'val/epoch_change_acc': v_chg_acc,
                    # Metrics on changed pixels only
                    'val/changed_pixels_iou': val_changed_metrics['iou'],
                    'val/changed_pixels_f1': val_changed_metrics['f1'],
                    'val/changed_pixels_acc': val_changed_metrics['accuracy'],
                    # Per-class metrics
                    'val/class_building_iou': val_per_class_metrics['iou_per_class'][4] if len(val_per_class_metrics['iou_per_class']) > 4 else 0.0,
                    'val/class_building_f1': val_per_class_metrics['f1_per_class'][4] if len(val_per_class_metrics['f1_per_class']) > 4 else 0.0,
                    'val/class_nvg_surf_iou': val_per_class_metrics['iou_per_class'][1] if len(val_per_class_metrics['iou_per_class']) > 1 else 0.0,
                    'val/class_nvg_surf_f1': val_per_class_metrics['f1_per_class'][1] if len(val_per_class_metrics['f1_per_class']) > 1 else 0.0,
                    'val/class_water_iou': val_per_class_metrics['iou_per_class'][3] if len(val_per_class_metrics['iou_per_class']) > 3 else 0.0,
                    'val/class_water_f1': val_per_class_metrics['f1_per_class'][3] if len(val_per_class_metrics['f1_per_class']) > 3 else 0.0,
                    'val/class_playground_iou': val_per_class_metrics['iou_per_class'][5] if len(val_per_class_metrics['iou_per_class']) > 5 else 0.0,
                    'val/class_playground_f1': val_per_class_metrics['f1_per_class'][5] if len(val_per_class_metrics['f1_per_class']) > 5 else 0.0,
                    'epoch': epoch
                }
                
                # Add transition metrics
                for trans_key, trans_val in val_transition_metrics.items():
                    wandb_metrics[f'val/transition_{trans_key}_acc'] = trans_val['accuracy']
                    wandb_metrics[f'val/transition_{trans_key}_count'] = trans_val['count']
                
                # Add per-class F1 and IoU if available
                if 'f1_per_class' in val_scores:
                    for cls_idx, f1_val in enumerate(val_scores['f1_per_class']):
                        wandb_metrics[f'val/class_{cls_idx}_f1'] = f1_val
                if 'iou_per_class' in val_scores:
                    for cls_idx, iou_val in enumerate(val_scores['iou_per_class']):
                        wandb_metrics[f'val/class_{cls_idx}_iou'] = iou_val
                
                wandb.log(wandb_metrics)
                
                logger.info(f"[Val] ep {epoch} loss {val_loss_avg:.5f} | seg mF1 {val_mf1:.4f} mIoU {val_miou:.4f} | "
                           f"chg Prec {v_chg_prec:.4f} Rec {v_chg_rec:.4f} F1 {v_chg_f1:.4f} IoU {v_chg_iou:.4f}")
                logger.info(f"✓ Logged {len(wandb_metrics)} validation metrics to W&B")

                # Save best model by seg mF1
                if val_mf1 > best_mF1:
                    best_mF1 = val_mf1
                    best_model_path = os.path.join(opt['path_cd']['checkpoint'], f'best_net_{epoch}.pth')
                    state = cd_model.module.state_dict() if isinstance(cd_model, nn.DataParallel) else cd_model.state_dict()
                    torch.save(state, best_model_path)
                    save_network(opt, epoch, cd_model, optimizer, is_best_model=True)
                    wandb.log({'best_val_mF1': best_mF1, 'best_model_epoch': epoch})
                    logger.info(f'New best (mF1={best_mF1:.5f}) saved to {best_model_path}')
                else:
                    logger.info(f'No improvement over best mF1={best_mF1:.5f}')

            # save regular checkpoint
            save_network(opt, epoch, cd_model, optimizer, is_best_model=False)
            scheduler.step()

        args.phase = 'test'

    # ----------------------------- TEST ----------------------------- #
    if args.phase == 'test':
        # load best if exists
        best_path = os.path.join(opt['path_cd']['checkpoint'], 'best_net.pth')
        if os.path.exists(best_path):
            cd_model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
            logger.info(f'Loaded best model from {best_path}')
        else:
            logger.warning(f'Best model not found at {best_path}; evaluating current weights.')

        cd_model.to(device).eval()
        os.makedirs(os.path.join(opt['path_cd']['result'], 'test'), exist_ok=True)

        test_metric = ConfuseMatrixMeter(n_class=num_classes)
        t_tp = t_fp = t_fn = t_tn = 0

        _max_test = args.max_test_batches or 0
        _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
        _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader

        with torch.no_grad():
            for tstep, tb in enumerate(tqdm(_test_iter, total=_test_total, desc="Test")):
                ti1 = tb['A'].to(device); ti2 = tb['B'].to(device)
                y1 = tb['L1'].to(device).long(); y2 = tb['L2'].to(device).long()

                touts = cd_model(ti1, ti2)
                s1, s2, chg, _ = unpack_outputs(touts)

                p1 = torch.argmax(s1, dim=1); p2 = torch.argmax(s2, dim=1)
                test_metric.update_cm(pr=safe_to_numpy_uint8(p1), gt=safe_to_numpy_uint8(y1))
                test_metric.update_cm(pr=safe_to_numpy_uint8(p2), gt=safe_to_numpy_uint8(y2))

                if chg is not None:
                    if chg.size(1) == 2:
                        cmask = torch.argmax(chg, dim=1)
                    else:
                        cmask = (torch.sigmoid(chg[:, 0]) > args.change_threshold).long()
                    gt = derive_change_bin(y1, y2)
                    pr_np, gt_np = cmask.cpu().numpy(), gt.cpu().numpy()
                    tp = np.logical_and(pr_np == 1, gt_np == 1).sum()
                    fp = np.logical_and(pr_np == 1, gt_np == 0).sum()
                    fn = np.logical_and(pr_np == 0, gt_np == 1).sum()
                    tn = np.logical_and(pr_np == 0, gt_np == 0).sum()
                    t_tp += tp; t_fp += fp; t_fn += fn; t_tn += tn

                # first-batch visuals
                if tstep == 0:
                    log_first_batch_to_wandb("test", tb, y1, y2, s1, s2, chg, num_classes)

        test_scores = test_metric.get_scores()
        if (t_tp + t_fp + t_fn) > 0:
            t_f1 = 2 * t_tp / max(2 * t_tp + t_fp + t_fn, 1e-8)
            t_iou = t_tp / max(t_tp + t_fp + t_fn, 1e-8)
            t_acc = (t_tp + t_tn) / max(t_tp + t_tn + t_fp + t_fn, 1e-8)
        else:
            t_f1 = t_iou = t_acc = 0.0

        wandb.log({
            'test/epoch_mF1': float(test_scores.get('mf1', 0.0)),
            'test/epoch_mIoU': float(test_scores.get('miou', 0.0)),
            'test/epoch_OA': float(test_scores.get('acc', 0.0)),
            'test/epoch_sek': float(test_scores.get('SCD_Sek', 0.0)),
            'test/epoch_fscd': float(test_scores.get('Fscd', 0.0)),
            'test/epoch_iou_mean': float(test_scores.get('SCD_IoU_mean', 0.0)),
            'test/epoch_change_f1': float(t_f1),
            'test/epoch_change_iou': float(t_iou),
            'test/epoch_change_acc': float(t_acc),
        })
        logger.info(f"[Test] seg mF1={test_scores.get('mf1', 0.0):.4f} | chg F1={t_f1:.4f}")


if __name__ == '__main__':
    main()
