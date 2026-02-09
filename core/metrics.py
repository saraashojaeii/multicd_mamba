import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid
from data.colormap import second_colormap


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def Index2Color(pred, cmap=second_colormap):
    colormap = np.asarray(cmap, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

def save_scdimg(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(np.squeeze(img, axis=0), cv2.COLOR_RGB2BGR))

def save_feat(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.applyColorMap(cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET))
    # cv2.imwrite(img_path, cv2.resize(img, (256,256), interpolation=cv2.INTER_))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def compute_semantic_metrics_on_changed(pred_t1, pred_t2, gt_t1, gt_t2, num_classes, ignore_index=255):
    """Compute semantic segmentation metrics only on changed pixels.
    
    Args:
        pred_t1, pred_t2: [B,H,W] predicted class indices
        gt_t1, gt_t2: [B,H,W] ground truth class indices
        num_classes: number of semantic classes
        ignore_index: index to ignore
    
    Returns:
        dict with 'iou', 'f1', 'accuracy' computed only on changed pixels
    """
    # Identify changed pixels
    changed_mask = (gt_t1 != gt_t2)
    valid_mask = (gt_t1 != ignore_index) & (gt_t2 != ignore_index)
    mask = changed_mask & valid_mask
    
    if not mask.any():
        return {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'num_pixels': 0}
    
    # Extract predictions and GT on changed pixels only
    pred_t1_chg = pred_t1[mask]
    pred_t2_chg = pred_t2[mask]
    gt_t1_chg = gt_t1[mask]
    gt_t2_chg = gt_t2[mask]
    
    # Combine T1 and T2 for overall metrics
    pred_all = np.concatenate([pred_t1_chg, pred_t2_chg])
    gt_all = np.concatenate([gt_t1_chg, gt_t2_chg])
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(pred_all, gt_all):
        if 0 <= p < num_classes and 0 <= g < num_classes:
            cm[g, p] += 1
    
    # Compute metrics
    tp_per_class = np.diag(cm)
    fp_per_class = cm.sum(axis=0) - tp_per_class
    fn_per_class = cm.sum(axis=1) - tp_per_class
    
    # IoU per class
    iou_per_class = tp_per_class / (tp_per_class + fp_per_class + fn_per_class + 1e-8)
    
    # F1 per class
    precision_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-8)
    recall_per_class = tp_per_class / (tp_per_class + fn_per_class + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
    
    # Mean metrics (only over classes present in GT)
    present_classes = (tp_per_class + fn_per_class) > 0
    if present_classes.any():
        mean_iou = iou_per_class[present_classes].mean()
        mean_f1 = f1_per_class[present_classes].mean()
    else:
        mean_iou = 0.0
        mean_f1 = 0.0
    
    # Overall accuracy
    accuracy = (pred_all == gt_all).sum() / len(gt_all)
    
    return {
        'iou': float(mean_iou),
        'f1': float(mean_f1),
        'accuracy': float(accuracy),
        'num_pixels': int(mask.sum()),
        'iou_per_class': iou_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
    }


def compute_per_class_metrics(pred_t1, pred_t2, gt_t1, gt_t2, num_classes, ignore_index=255):
    """Compute per-class IoU and F1 for semantic segmentation (all pixels).
    
    Args:
        pred_t1, pred_t2: [B,H,W] predicted class indices
        gt_t1, gt_t2: [B,H,W] ground truth class indices
        num_classes: number of semantic classes
        ignore_index: index to ignore
    
    Returns:
        dict with per-class IoU and F1
    """
    valid_mask_t1 = (gt_t1 != ignore_index)
    valid_mask_t2 = (gt_t2 != ignore_index)
    
    # Combine T1 and T2
    pred_all = np.concatenate([pred_t1[valid_mask_t1], pred_t2[valid_mask_t2]])
    gt_all = np.concatenate([gt_t1[valid_mask_t1], gt_t2[valid_mask_t2]])
    
    # Compute confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, g in zip(pred_all, gt_all):
        if 0 <= p < num_classes and 0 <= g < num_classes:
            cm[g, p] += 1
    
    # Compute metrics per class
    tp_per_class = np.diag(cm)
    fp_per_class = cm.sum(axis=0) - tp_per_class
    fn_per_class = cm.sum(axis=1) - tp_per_class
    
    iou_per_class = tp_per_class / (tp_per_class + fp_per_class + fn_per_class + 1e-8)
    precision_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-8)
    recall_per_class = tp_per_class / (tp_per_class + fn_per_class + 1e-8)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
    
    return {
        'iou_per_class': iou_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
    }


def compute_transition_metrics(pred_t1, pred_t2, gt_t1, gt_t2, transitions, num_classes, ignore_index=255):
    """Compute metrics for specific semantic transitions.
    
    Args:
        pred_t1, pred_t2: [B,H,W] predicted class indices
        gt_t1, gt_t2: [B,H,W] ground truth class indices
        transitions: list of (from_class, to_class) tuples to track
        num_classes: number of semantic classes
        ignore_index: index to ignore
    
    Returns:
        dict with metrics for each transition
    """
    changed_mask = (gt_t1 != gt_t2)
    valid_mask = (gt_t1 != ignore_index) & (gt_t2 != ignore_index)
    mask = changed_mask & valid_mask
    
    if not mask.any():
        return {f'{t[0]}_to_{t[1]}': {'accuracy': 0.0, 'count': 0} for t in transitions}
    
    results = {}
    
    for from_cls, to_cls in transitions:
        # Find pixels with this specific transition in GT
        trans_mask = mask & (gt_t1 == from_cls) & (gt_t2 == to_cls)
        
        if not trans_mask.any():
            results[f'{from_cls}_to_{to_cls}'] = {
                'accuracy': 0.0,
                'count': 0,
                'correct_t1': 0.0,
                'correct_t2': 0.0,
                'correct_both': 0.0,
            }
            continue
        
        # Extract predictions for this transition
        pred_t1_trans = pred_t1[trans_mask]
        pred_t2_trans = pred_t2[trans_mask]
        gt_t1_trans = gt_t1[trans_mask]
        gt_t2_trans = gt_t2[trans_mask]
        
        # Compute accuracy
        correct_t1 = (pred_t1_trans == gt_t1_trans).sum()
        correct_t2 = (pred_t2_trans == gt_t2_trans).sum()
        correct_both = ((pred_t1_trans == gt_t1_trans) & (pred_t2_trans == gt_t2_trans)).sum()
        
        count = trans_mask.sum()
        
        results[f'{from_cls}_to_{to_cls}'] = {
            'accuracy': float(correct_both / count),
            'count': int(count),
            'correct_t1': float(correct_t1 / count),
            'correct_t2': float(correct_t2 / count),
            'correct_both': float(correct_both / count),
        }
    
    return results


def compute_unchanged_stability_metrics(pred_t1, pred_t2, gt_t1, gt_t2, num_classes, ignore_index=255, 
                                       pred_probs_t1=None, pred_probs_t2=None, background_class=0):
    """Compute semantic stability metrics on unchanged pixels.
    
    This metric quantifies how consistent the model's predictions are in unchanged regions,
    which is critical for semantic change detection. High stability means the model correctly
    predicts the same class for both T1 and T2 in unchanged areas.
    
    Args:
        pred_t1, pred_t2: [B,H,W] predicted class indices
        gt_t1, gt_t2: [B,H,W] ground truth class indices
        num_classes: number of semantic classes
        ignore_index: index to ignore
        pred_probs_t1, pred_probs_t2: [B,C,H,W] optional prediction probabilities for entropy
        background_class: class index for background (default 0)
    
    Returns:
        dict with:
            - stability: overall stability (% of unchanged pixels with consistent predictions)
            - stability_per_class: per-class stability
            - unchanged_fp_rate: false positive rate in unchanged regions (non-background predictions)
            - unchanged_entropy: average prediction entropy in unchanged regions (if probs provided)
            - num_unchanged_pixels: total number of unchanged pixels evaluated
    """
    # Identify unchanged pixels in ground truth
    unchanged_mask_gt = (gt_t1 == gt_t2)
    valid_mask = (gt_t1 != ignore_index) & (gt_t2 != ignore_index)
    unchanged_mask = unchanged_mask_gt & valid_mask
    
    if not unchanged_mask.any():
        return {
            'stability': 0.0,
            'stability_per_class': [0.0] * num_classes,
            'unchanged_fp_rate': 0.0,
            'unchanged_entropy': 0.0,
            'num_unchanged_pixels': 0,
        }
    
    # Extract predictions on unchanged pixels
    pred_t1_unch = pred_t1[unchanged_mask]
    pred_t2_unch = pred_t2[unchanged_mask]
    gt_unch = gt_t1[unchanged_mask]  # GT is same for T1 and T2 in unchanged regions
    
    # Overall stability: percentage where pred_t1 == pred_t2
    stable_predictions = (pred_t1_unch == pred_t2_unch)
    overall_stability = stable_predictions.sum() / len(pred_t1_unch)
    
    # Per-class stability
    stability_per_class = []
    for cls in range(num_classes):
        cls_mask = (gt_unch == cls)
        if cls_mask.any():
            cls_stable = stable_predictions[cls_mask].sum() / cls_mask.sum()
            stability_per_class.append(float(cls_stable))
        else:
            stability_per_class.append(0.0)
    
    # Unchanged false positive rate: % of unchanged pixels predicted as non-background
    # (assuming background_class is the "no change" class)
    # This measures spurious change predictions in unchanged areas
    pred_non_bg_t1 = (pred_t1_unch != background_class).sum()
    pred_non_bg_t2 = (pred_t2_unch != background_class).sum()
    gt_non_bg = (gt_unch != background_class).sum()
    
    # FP: predicted non-background when GT is background
    gt_bg_mask = (gt_unch == background_class)
    if gt_bg_mask.any():
        fp_t1 = (pred_t1_unch[gt_bg_mask] != background_class).sum()
        fp_t2 = (pred_t2_unch[gt_bg_mask] != background_class).sum()
        unchanged_fp_rate = (fp_t1 + fp_t2) / (2 * gt_bg_mask.sum())
    else:
        unchanged_fp_rate = 0.0
    
    # Prediction entropy in unchanged regions (if probabilities provided)
    unchanged_entropy = 0.0
    if pred_probs_t1 is not None and pred_probs_t2 is not None:
        # Extract probabilities for unchanged pixels
        # pred_probs shape: [B,C,H,W] -> need to extract [N,C] for unchanged pixels
        B, C, H, W = pred_probs_t1.shape
        
        # Flatten spatial dimensions and extract unchanged pixels
        probs_t1_flat = pred_probs_t1.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)  # [B*H*W, C]
        probs_t2_flat = pred_probs_t2.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)
        unchanged_mask_flat = unchanged_mask.reshape(-1)
        
        probs_t1_unch = probs_t1_flat[unchanged_mask_flat]  # [N_unch, C]
        probs_t2_unch = probs_t2_flat[unchanged_mask_flat]
        
        # Compute entropy: -sum(p * log(p))
        eps = 1e-8
        entropy_t1 = -(probs_t1_unch * np.log(probs_t1_unch + eps)).sum(axis=1).mean()
        entropy_t2 = -(probs_t2_unch * np.log(probs_t2_unch + eps)).sum(axis=1).mean()
        unchanged_entropy = (entropy_t1 + entropy_t2) / 2.0
    
    return {
        'stability': float(overall_stability),
        'stability_per_class': stability_per_class,
        'unchanged_fp_rate': float(unchanged_fp_rate),
        'unchanged_entropy': float(unchanged_entropy),
        'num_unchanged_pixels': int(unchanged_mask.sum()),
        'stable_pixel_count': int(stable_predictions.sum()),
        'unstable_pixel_count': int((~stable_predictions).sum()),
    }
