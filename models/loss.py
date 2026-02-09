import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor, einsum
from misc.torchutils import class2one_hot,simplex
from models.darnet_help.loss_help import FocalLoss, dernet_dice_loss

def _resize_like(inp: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear"):
    """Resize logits/probs inp to have the same HxW as ref (no-op if already same)."""
    if inp.shape[-2:] != ref.shape[-2:]:
        return F.interpolate(inp, size=ref.shape[-2:], mode=mode, align_corners=(mode=='bilinear'))
    return inp


def cross_entropy_loss_fn(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    input:  [N, C, H, W] logits
    target: [N, H, W] (long) or [N,1,H,W]; may contain ignore_index
    """
    # squeeze [N,1,H,W] -> [N,H,W]
    if target.dim() == 4:
        if target.shape[1] == 1:
            target = target[:, 0]
        elif target.shape[-1] == 3:
            # channels-last RGB mask; take first channel as label map
            target = target[..., 0]

    target = target.long()

    # Resize logits to target spatial size if needed
    if input.shape[-2:] != target.shape[-2:]:
        input = F.interpolate(input, size=target.shape[-2:], mode='bilinear', align_corners=True)

    # DO NOT clamp labels; CE will ignore ignore_index properly.
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=255, smooth=1e-6,
                 idc=None, skip_absent=True):
        super().__init__()
        self.num_classes = num_classes
        self.weight = None if weight is None else torch.as_tensor(weight, dtype=torch.float32)
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.idc = list(range(num_classes)) if idc is None else idc
        self.skip_absent = skip_absent

    def forward(self, predicts, target):
        """
        predicts: [B,C,H,W] logits
        target:   [B,H,W] or [B,1,H,W] long in [0..C-1] or ignore_index
        """
        probs = torch.softmax(predicts, dim=1)

        # target to [B,H,W]
        if target.dim() == 4 and target.shape[1] == 1:
            target = target[:, 0]
        elif target.dim() == 4 and target.shape[-1] == 3:
            target = target[..., 0]
        target = target.long()

        B, C, H, W = probs.shape
        if (H, W) != target.shape[-2:]:
            probs = F.interpolate(probs, size=target.shape[-2:], mode='bilinear', align_corners=True)

        # mask ignored pixels
        if self.ignore_index is not None:
            valid = (target != self.ignore_index)            # [B,H,W] bool
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # one-hot (on valid pixels only)
        # Build one-hot as float on the fly to avoid external helpers
        target_clamped = target.clone()
        target_clamped[~valid] = 0  # placeholder; will be masked out
        one_hot = F.one_hot(target_clamped, num_classes=self.num_classes)  # [B,H,W,C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()                      # [B,C,H,W]

        # zero out ignored pixels in both probs and target
        valid_f = valid.float()
        probs = probs * valid_f.unsqueeze(1)
        one_hot = one_hot * valid_f.unsqueeze(1)

        # compute per-class dice and average over present classes
        num = (probs * one_hot).sum(dim=(0, 2, 3)) * 2.0   # [C]
        den = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3)) + self.smooth  # [C]
        dice_c = 1.0 - (num + self.smooth) / den           # [C]

        # skip classes that are absent in both GT and pred
        if self.skip_absent:
            gt_sum = one_hot.sum(dim=(0, 2, 3))            # [C]
            pr_sum = probs.sum(dim=(0, 2, 3))              # [C]
            present = (gt_sum > 0) | (pr_sum > 0)          # [C]
            if present.any():
                dice_c = dice_c[present]
            else:
                return predicts.new_tensor(0.0)

        if self.weight is not None:
            w = self.weight.to(predicts.device)
            if self.skip_absent:
                w = w[present]
            w = w / (w.sum() + 1e-12)
            return (dice_c * w).sum()
        else:
            return dice_c.mean()

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CEDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class DiceOnlyLoss(nn.Module):
    def __init__(self, num_classes, dice_loss_kwargs=None):
        super(DiceOnlyLoss, self).__init__()
        self.num_classes = num_classes
        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        return self.dice_loss(input, target)

class CE2Dice1Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE2Dice1Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class CE1Dice2Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=1.0, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE1Dice2Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)
        
    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

# Note: ce_scl was identical to ce_dice. If it needs specific SCL logic, it requires a separate implementation.
# For now, if 'ce_scl' is chosen, it would need to be mapped to CEDiceLoss or a new SCL specific class.


class MultiClassCDLoss(nn.Module):
    """
    Multi-class Change Detection Loss.
    Computes segmentation loss for T1, T2, and transition/change head.
    - seg_loss: loss function for segmentation ("ce", "dice", "cedice")
    - change_loss: loss function for transition/change ("ce", "cedice")
    - loss_weights: dict with weights for each head ("seg_t1", "seg_t2", "change")
    Usage:
        loss_fn = MultiClassCDLoss(num_classes, seg_loss="cedice", change_loss="ce")
        loss = loss_fn(preds, targets)
        # preds: (seg_logits_t1, seg_logits_t2, change_logits)
        # targets: dict with keys: "seg_t1", "seg_t2", "change"
    """
    def __init__(self, num_classes, seg_loss="cedice", change_loss="ce", loss_weights=None):
        super().__init__()
        self.num_classes = num_classes
        if seg_loss == "ce":
            self.seg_loss_fn = cross_entropy_loss_fn
        elif seg_loss == "dice":
            self.seg_loss_fn = DiceLoss(num_classes)
        else:
            self.seg_loss_fn = CEDiceLoss(num_classes)

        if change_loss == "ce":
            self.change_loss_fn = cross_entropy_loss_fn
        else:
            self.change_loss_fn = CEDiceLoss(num_classes*num_classes)

        self.loss_weights = loss_weights if loss_weights is not None else {"seg_t1": 1.0, "seg_t2": 1.0, "change": 1.0}

    def forward(self, preds, targets):
        # Unpack predictions
        if isinstance(preds, tuple) and len(preds) == 3:
            seg_logits_t1, seg_logits_t2, change_logits = preds
        else:
            raise ValueError(f"Expected preds to be a tuple of 3 tensors, got {type(preds)}")
        
        # Unpack targets and ensure they're properly formatted
        try:
            seg_t1 = targets["seg_t1"]
            seg_t2 = targets["seg_t2"]
            change = targets["change"]
            
            # Print debug info
            # print(f"Target shapes - seg_t1: {seg_t1.shape}, seg_t2: {seg_t2.shape}, change: {change.shape}")
            # print(f"Prediction shapes - seg_t1: {seg_logits_t1.shape}, seg_t2: {seg_logits_t2.shape}, change: {change_logits.shape}")
            
            # Get number of classes
            num_classes = seg_logits_t1.shape[1]
            # print(f"Number of classes: {num_classes}")
            
            # Process targets for segmentation (T1)
            if seg_t1.dim() == 4 and seg_t1.shape[-1] == 3:  # Handle channels-last format
                seg_t1 = seg_t1[..., 0]  # Take first channel
                # print(f"Converted seg_t1 from channels-last format: {seg_t1.shape}")
            
            # Process targets for segmentation (T2)
            if seg_t2.dim() == 4 and seg_t2.shape[-1] == 3:  # Handle channels-last format
                seg_t2 = seg_t2[..., 0]  # Take first channel
                # print(f"Converted seg_t2 from channels-last format: {seg_t2.shape}")
            
            # Process targets for change detection
            if change.dim() == 4 and change.shape[-1] == 3:  # Handle channels-last format
                change = change[..., 0]  # Take first channel
                # print(f"Converted change from channels-last format: {change.shape}")
            
            # Compute losses
            loss_t1 = self.seg_loss_fn(seg_logits_t1, seg_t1)
            loss_t2 = self.seg_loss_fn(seg_logits_t2, seg_t2)
            loss_change = self.change_loss_fn(change_logits, change)
            
        except Exception as e:
            # print(f"Error processing targets: {e}")
            # print(f"Target keys: {targets.keys()}")
            # for k, v in targets.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
            #     else:
            #         print(f"  {k}: {type(v)}")
            raise
        
        # Combine losses with weights
        total = (
            self.loss_weights["seg_t1"] * loss_t1 +
            self.loss_weights["seg_t2"] * loss_t2 +
            self.loss_weights["change"] * loss_change
        )
        
        return total, {"seg_t1": loss_t1.item(), "seg_t2": loss_t2.item(), "change": loss_change.item()}


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit.float(), truth.float(), reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss

def hybrid_loss(predictions, target, weight=[0,2,0.2,0.2,0.2,0.2]):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i,prediction in enumerate(predictions):

        bce = cross_entropy(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i]*(bce + dice) #- ssimloss

    return loss

class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss


class BinaryDiceLoss(nn.Module):
    """Soft Dice on a binary mask, taking logits as input by default."""
    def __init__(self, from_logits: bool = True, smooth: float = 1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: [B,1,H,W] (logits or probs), target: [B,1,H,W] or [B,H,W] in {0,1}
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred = _resize_like(pred, target)
        if self.from_logits:
            prob = torch.sigmoid(pred)
        else:
            prob = pred.clamp_min(0).clamp_max(1)

        target = target.float()
        intersection = (prob * target).sum(dim=(1,2,3))
        union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class ChangeHeadBCEDiceLoss(nn.Module):
    """BCEWithLogits + lambda_dice * BinaryDiceLoss for the change head."""
    def __init__(self, lambda_dice: float = 1.0, pos_weight: float = None):
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.dice = BinaryDiceLoss(from_logits=True)
        self.pos_weight = None if pos_weight is None else torch.tensor([pos_weight])

    def forward(self, change_logits: torch.Tensor, change_gt: torch.Tensor):
        # change_logits: [B,1,H,W], change_gt: [B,1,H,W] or [B,H,W] in {0,1}
        if change_gt.dim() == 3:
            change_gt = change_gt.unsqueeze(1)
        change_logits = _resize_like(change_logits, change_gt)
        bce = F.binary_cross_entropy_with_logits(change_logits, change_gt.float(), pos_weight=self.pos_weight)
        dice = self.dice(change_logits, change_gt)
        return bce + self.lambda_dice * dice

class UnchangedSymmetricKLLoss(nn.Module):
    """Symmetric KL between t1/t2 class distributions on UNCHANGED pixels (c==0).
    Inputs are logits for numerical stability; temperature T softens distributions.
    With optional confidence gating to avoid reinforcing wrong predictions.
    """
    def __init__(self, T: float = 4.0, detach_one_side: bool = True, eps: float = 1e-8,
                 use_conf_gate: bool = False, tau: float = 0.9, conf_method: str = "max_prob"):
        super().__init__()
        self.T = float(T)
        self.detach_one_side = bool(detach_one_side)
        self.eps = eps
        self.use_conf_gate = bool(use_conf_gate)
        self.tau = float(tau)
        assert conf_method in {"max_prob", "cosine"}, f"conf_method must be 'max_prob' or 'cosine', got {conf_method}"
        self.conf_method = conf_method

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_mask: torch.Tensor):
        # z1,z2: [B,C,H,W] logits; change_mask: [B,1,H,W] or [B,H,W] where 0=unchanged,1=changed
        if change_mask.dim() == 3:
            change_mask = change_mask.unsqueeze(1)
        z1 = _resize_like(z1, change_mask)
        z2 = _resize_like(z2, change_mask)

        unch = (change_mask == 0).float()  # [B,1,H,W]
        if unch.sum() < 1:
            return torch.zeros([], device=z1.device, dtype=z1.dtype)

        z1T = z1 / self.T
        z2T = z2 / self.T
        logp1 = F.log_softmax(z1T, dim=1)
        logp2 = F.log_softmax(z2T, dim=1)
        p1 = F.softmax(z1T, dim=1)
        p2 = F.softmax(z2T, dim=1)
        if self.detach_one_side:
            p1 = p1.detach()
            p2 = p2.detach()

        # Apply confidence gating if enabled
        if self.use_conf_gate:
            if self.conf_method == "max_prob":
                # Max-prob confidence: only apply KL where both predictions are confident
                conf1 = p1.max(dim=1, keepdim=True)[0]  # [B,1,H,W]
                conf2 = p2.max(dim=1, keepdim=True)[0]  # [B,1,H,W]
                conf_mask = ((conf1 > self.tau) & (conf2 > self.tau)).float()
            else:  # cosine
                # Cosine similarity of distributions: high similarity = confident agreement
                num = (p1 * p2).sum(dim=1, keepdim=True)  # [B,1,H,W]
                den = (p1.norm(p=2, dim=1, keepdim=True) * p2.norm(p=2, dim=1, keepdim=True)) + self.eps
                cos_sim = num / den  # [B,1,H,W]
                conf_mask = (cos_sim > self.tau).float()
            
            # Apply confidence mask to unchanged pixels
            unch = unch * conf_mask

        kl12 = (p1 * (logp1 - logp2)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        kl21 = (p2 * (logp2 - logp1)).sum(dim=1, keepdim=True)
        sym = (kl12 + kl21) * unch
        return sym.sum() / (unch.sum() + self.eps)

class ChangedDiversityCosineMarginLoss(nn.Module):
    """Encourage dissimilar semantics on CHANGED pixels via cosine similarity margin."""
    def __init__(self, margin: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.margin = float(margin)
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_mask: torch.Tensor):
        # logits -> probs
        if change_mask.dim() == 3:
            change_mask = change_mask.unsqueeze(1)
        z1 = _resize_like(z1, change_mask)
        z2 = _resize_like(z2, change_mask)

        p1 = F.softmax(z1, dim=1)
        p2 = F.softmax(z2, dim=1)
        ch = (change_mask == 1).float()  # [B,1,H,W]
        if ch.sum() < 1:
            return torch.zeros([], device=z1.device, dtype=z1.dtype)

        # cosine similarity along class dim
        num = (p1 * p2).sum(dim=1, keepdim=True)
        den = p1.norm(p=2, dim=1, keepdim=True) * p2.norm(p=2, dim=1, keepdim=True) + self.eps
        s = (num / den)  # [B,1,H,W]
        loss = F.relu(s - self.margin) * ch
        return loss.sum() / (ch.sum() + self.eps)

class CouplingChangeSemanticLoss(nn.Module):
    """Penalize disagreement between change prob q and semantic-derived change score r=1-dot(p1,p2)."""
    def __init__(self, distance: str = "l1", detach_semantics: bool = False, eps: float = 1e-8):
        super().__init__()
        assert distance in {"l1","l2","bce"}
        self.distance = distance
        self.detach_semantics = detach_semantics
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_logits: torch.Tensor):
        # z1,z2: [B,C,H,W], change_logits: [B,1,H,W]
        change_logits = change_logits
        z1 = _resize_like(z1, change_logits)
        z2 = _resize_like(z2, change_logits)

        p1 = F.softmax(z1, dim=1)
        p2 = F.softmax(z2, dim=1)
        if self.detach_semantics:
            p1 = p1.detach(); p2 = p2.detach()

        q = torch.sigmoid(change_logits)           # [B,1,H,W]
        e = (p1 * p2).sum(dim=1, keepdim=True)     # probability of equality
        r = 1.0 - e                                # semantic change score in [0,1]

        if self.distance == "l1":
            return (q - r).abs().mean()
        elif self.distance == "l2":
            return ((q - r) ** 2).mean()
        else:  # 'bce' view: treat r as soft target for q
            r = r.clamp(min=0.0 + self.eps, max=1.0 - self.eps)
            return F.binary_cross_entropy(q, r)



class TripletChangeSegLoss(nn.Module):
    """Convenience wrapper that sums:
        - seg losses t1 & t2 (you provide a callable seg_loss_fn)
        - change loss (BCE+Dice)
        - unchanged symmetric KL
        - changed diversity (cos margin)
        - coupling loss
    You can also use each class independently in your train script.
    """
    def __init__(self,
                 seg_loss_fn: nn.Module,
                 lambda_seg: float = 1.0,
                 lambda_cd: float = 1.0,
                 lambda_unch: float = 0.2,
                 lambda_ch: float = 0.2,
                 lambda_cpl: float = 0.5,
                 lambda_pseudo: float = 0.1,
                 T: float = 4.0,
                 margin: float = 0.3,
                 boost: float = 5.0,
                 ignore_index: int = 255,
                 transition_weights: torch.Tensor = None,
                 use_conf_gate: bool = False,
                 conf_tau: float = 0.9,
                 conf_method: str = "max_prob",
                 pseudo_conf_tau: float = 0.9,
                 seg_mask_mode: str = "changed_only",
                 min_changed_threshold: int = 10,
                 enable_changed_only_supervision: bool = True,
                 enable_unch_conf_gating: bool = True,
                 enable_pseudo_labeling: bool = True):
        super().__init__()
        self.seg_loss_fn = seg_loss_fn
        self.cd_loss = ChangeHeadBCEDiceLoss(lambda_dice=1.0)
        
        # Apply ablation switch for unchanged confidence gating
        use_conf_gate_effective = use_conf_gate and enable_unch_conf_gating
        self.unch_kl = UnchangedSymmetricKLLoss(T=T, use_conf_gate=use_conf_gate_effective, tau=conf_tau, conf_method=conf_method)
        
        self.ch_div = ChangedDiversityCosineMarginLoss(margin=margin)
        self.couple = CouplingChangeSemanticLoss(distance="l1")

        self.lam_seg = lambda_seg
        self.lam_cd = lambda_cd
        self.lam_unch = lambda_unch
        self.lam_ch = lambda_ch
        self.lam_cpl = lambda_cpl
        self.lam_pseudo = lambda_pseudo
        self.boost = float(boost)
        self.ignore_index = int(ignore_index)
        self.pseudo_conf_tau = float(pseudo_conf_tau)
        self.seg_mask_mode = seg_mask_mode
        self.min_changed_threshold = min_changed_threshold
        self.eps = 1e-8
        
        # Ablation switches
        self.enable_changed_only_supervision = enable_changed_only_supervision
        self.enable_unch_conf_gating = enable_unch_conf_gating
        self.enable_pseudo_labeling = enable_pseudo_labeling
        
        # Transition weights [C, C] for rebalancing changed pixels
        self.register_buffer('transition_weights', transition_weights if transition_weights is not None else None)

    def forward(self, preds, targets):
        z1, z2, u = preds
        y1 = targets["seg_t1"]
        y2 = targets["seg_t2"]
        c  = targets["change"]

        # Ensure change mask shape [B,1,H,W]
        if c.dim() == 3:
            c = c.unsqueeze(1)
        c = _resize_like(c, z1)

        # Compute per-pixel seg loss maps; if seg_loss_fn returns a scalar, fall back to CE(no reduction)
        loss_map_t1 = self.seg_loss_fn(z1, y1)
        if not isinstance(loss_map_t1, torch.Tensor) or loss_map_t1.dim() < 2:
            z1r = _resize_like(z1, y1)
            loss_map_t1 = F.cross_entropy(z1r, y1.long(), ignore_index=self.ignore_index, reduction='none')  # [B,H,W]

        loss_map_t2 = self.seg_loss_fn(z2, y2)
        if not isinstance(loss_map_t2, torch.Tensor) or loss_map_t2.dim() < 2:
            z2r = _resize_like(z2, y2)
            loss_map_t2 = F.cross_entropy(z2r, y2.long(), ignore_index=self.ignore_index, reduction='none')  # [B,H,W]

        # Build masks for changed vs unchanged pixels
        changed = (c.squeeze(1) > 0.5).float()  # [B,H,W]
        unchanged = 1.0 - changed  # [B,H,W]
        
        # Valid pixel masks (exclude ignore_index)
        valid1 = (y1 != self.ignore_index).float()
        valid2 = (y2 != self.ignore_index).float()
        
        # Count changed pixels for fallback logic
        changed_count = (changed * valid1).sum() + (changed * valid2).sum()
        
        # Apply segmentation supervision based on seg_mask_mode and ablation switches
        if self.enable_changed_only_supervision and self.seg_mask_mode == "changed_only":
            # Changed-only mode: supervise only changed pixels (or fallback to all valid pixels)
            if changed_count >= self.min_changed_threshold:
                # Normal mode: supervise only changed pixels
                w1 = changed * valid1
                w2 = changed * valid2
                
                # Apply transition-aware weights for changed pixels (optimized)
                if self.transition_weights is not None:
                    y1_clamped = y1.long().clamp(0, self.transition_weights.size(0)-1)
                    y2_clamped = y2.long().clamp(0, self.transition_weights.size(1)-1)
                    trans_w = self.transition_weights[y1_clamped, y2_clamped]  # [B,H,W]
                    w1 = w1 * trans_w
                    w2 = w2 * trans_w
            else:
                # Fallback mode: supervise all valid pixels (no change or very few changed pixels)
                w1 = valid1
                w2 = valid2
        elif self.seg_mask_mode == "mixed":
            # Mixed mode: supervise changed pixels strongly, unchanged pixels weakly
            w1 = changed * valid1 + 0.1 * unchanged * valid1
            w2 = changed * valid2 + 0.1 * unchanged * valid2
        else:
            # Full mode (baseline): supervise all valid pixels equally
            w1 = valid1
            w2 = valid2

        L_t1 = (loss_map_t1 * w1).sum() / (w1.sum() + self.eps)
        L_t2 = (loss_map_t2 * w2).sum() / (w2.sum() + self.eps)
        L_seg = L_t1 + L_t2
        L_cd  = self.cd_loss(u, c)
        L_unch = self.unch_kl(z1, z2, c)
        L_ch   = self.ch_div(z1, z2, c)
        L_cpl  = self.couple(z1, z2, u)

        # Pseudo-labeling on unchanged pixels
        L_pseudo = torch.zeros([], device=z1.device, dtype=z1.dtype)
        if self.lam_pseudo > 0 and self.enable_pseudo_labeling:
            # Build unchanged mask
            unchanged = (c.squeeze(1) <= 0.5).float()  # [B,H,W]
            
            # Compute softmax probabilities
            z1_resized = _resize_like(z1, y1)
            z2_resized = _resize_like(z2, y2)
            p1 = F.softmax(z1_resized, dim=1)  # [B,C,H,W]
            p2 = F.softmax(z2_resized, dim=1)  # [B,C,H,W]
            
            # Compute confidence for each pixel
            conf1 = p1.max(dim=1)[0]  # [B,H,W]
            conf2 = p2.max(dim=1)[0]  # [B,H,W]
            
            # Pick teacher: use t1 if more confident, else t2
            use_t1_as_teacher = (conf1 > conf2).float()  # [B,H,W]
            use_t2_as_teacher = 1.0 - use_t1_as_teacher
            
            # Generate pseudo-labels from more confident prediction
            pseudo_t1 = torch.argmax(p1, dim=1)  # [B,H,W]
            pseudo_t2 = torch.argmax(p2, dim=1)  # [B,H,W]
            pseudo_label = (use_t1_as_teacher * pseudo_t1 + use_t2_as_teacher * pseudo_t2).long()
            
            # Confidence mask: only supervise where teacher is confident
            teacher_conf = torch.max(conf1, conf2)  # [B,H,W]
            conf_mask = (teacher_conf > self.pseudo_conf_tau).float()
            
            # Valid mask: exclude ignore_index pixels
            valid_mask = valid1 * valid2  # Both must be valid
            
            # Final mask: unchanged & confident & valid
            pseudo_mask = unchanged * conf_mask * valid_mask  # [B,H,W]
            
            if pseudo_mask.sum() > 0:
                # Apply CE loss on student logits (the less confident one)
                # Student t1: supervised by pseudo label from t2 (when t2 is teacher)
                loss_pseudo_t1 = F.cross_entropy(z1_resized, pseudo_label, ignore_index=self.ignore_index, reduction='none')
                loss_pseudo_t1 = (loss_pseudo_t1 * pseudo_mask * use_t2_as_teacher).sum() / ((pseudo_mask * use_t2_as_teacher).sum() + self.eps)
                
                # Student t2: supervised by pseudo label from t1 (when t1 is teacher)
                loss_pseudo_t2 = F.cross_entropy(z2_resized, pseudo_label, ignore_index=self.ignore_index, reduction='none')
                loss_pseudo_t2 = (loss_pseudo_t2 * pseudo_mask * use_t1_as_teacher).sum() / ((pseudo_mask * use_t1_as_teacher).sum() + self.eps)
                
                L_pseudo = loss_pseudo_t1 + loss_pseudo_t2

        # Guard numerics
        L_seg  = torch.nan_to_num(L_seg,  nan=0.0, posinf=1e4, neginf=0.0)
        L_cd   = torch.nan_to_num(L_cd,   nan=0.0, posinf=1e4, neginf=0.0)
        L_unch = torch.nan_to_num(L_unch, nan=0.0, posinf=1e4, neginf=0.0)
        L_ch   = torch.nan_to_num(L_ch,   nan=0.0, posinf=1e4, neginf=0.0)
        L_cpl  = torch.nan_to_num(L_cpl,  nan=0.0, posinf=1e4, neginf=0.0)
        L_pseudo = torch.nan_to_num(L_pseudo, nan=0.0, posinf=1e4, neginf=0.0)

        total = (self.lam_seg * L_seg +
                 self.lam_cd  * L_cd  +
                 self.lam_unch* L_unch+
                 self.lam_ch  * L_ch  +
                 self.lam_cpl * L_cpl +
                 self.lam_pseudo * L_pseudo)

        return total, {
            "seg": L_seg.item(),
            "seg_t1": L_t1.item(),
            "seg_t2": L_t2.item(),
            "cd": L_cd.item(),
            "unch_kl": L_unch.item(),
            "ch_div": L_ch.item(),
            "couple": L_cpl.item(),
            "pseudo_unch": L_pseudo.item()
        }


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

# ---------------------------
# Utilities
# ---------------------------

def compute_class_weights(
    class_counts: torch.Tensor,
    method: str = "median_frequency",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        class_counts: [C] tensor with pixel counts per class (exclude ignore_index beforehand)
        method: "inverse" or "median_frequency"
    Returns:
        weights [C] (float32)
    """
    class_counts = class_counts.float().clamp_min(eps)
    if method == "inverse":
        weights = 1.0 / class_counts
        return (weights / weights.sum()) * len(class_counts)  # normalize to ~C
    elif method == "median_frequency":
        freq = class_counts / class_counts.sum()
        med = torch.median(freq)
        weights = med / freq
        return weights
    else:
        raise ValueError(f"Unknown method: {method}")

def one_hot_ignore(
    target: torch.Tensor, num_classes: int, ignore_index: int
) -> torch.Tensor:
    """
    Convert [B,H,W] targets to one-hot [B,C,H,W] with ignored pixels set to 0 across channels.
    """
    b, h, w = target.shape
    oh = torch.zeros(b, num_classes, h, w, device=target.device, dtype=torch.float32)
    mask = (target != ignore_index) & (target >= 0)
    if mask.any():
        oh.scatter_(1, target.clamp(min=0).unsqueeze(1)[mask.unsqueeze(1).expand_as(oh)].view(b, 1, -1), 1.0)
        # The above scatter is tricky; safer approach below:
        oh.zero_()
        valid = mask
        oh[torch.arange(b, device=target.device).unsqueeze(-1).unsqueeze(-1),
           target.clamp(min=0), torch.arange(h, device=target.device).unsqueeze(0).unsqueeze(-1).expand(b, h, w),
           torch.arange(w, device=target.device).unsqueeze(0).unsqueeze(0).expand(b, h, w)] = 1.0
        oh *= valid.unsqueeze(1).float()
    return oh

def one_hot_ignore_safe(target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    # safer, simpler implementation
    b, h, w = target.shape
    oh = torch.zeros(b, num_classes, h, w, device=target.device, dtype=torch.float32)
    valid = (target != ignore_index) & (target >= 0)
    # set ignored labels to 0 to avoid scatter issues
    tgt = target.clone()
    tgt[~valid] = 0
    oh.scatter_(1, tgt.unsqueeze(1), 1.0)
    oh *= valid.unsqueeze(1).float()
    return oh

# ---------------------------
# Losses
# ---------------------------

class WeightedCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W]; target: [B,H,W] with ignore_index
        """
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

class PerPixelDiceLoss(nn.Module):
    """Dice loss that returns per-pixel loss map [B,H,W] for change-aware weighting."""
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W]; target: [B,H,W]
        Returns: [B,H,W] per-pixel dice loss
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # [B,C,H,W]
        
        # Create one-hot target
        target_clamped = target.clone()
        valid = (target != self.ignore_index)
        target_clamped[~valid] = 0
        target_oh = F.one_hot(target_clamped, num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        
        # Mask invalid pixels
        valid_f = valid.unsqueeze(1).float()  # [B,1,H,W]
        probs = probs * valid_f
        target_oh = target_oh * valid_f
        
        # Per-pixel dice: for each pixel, compute dice over classes
        # intersection and union per pixel
        intersection = (probs * target_oh).sum(dim=1)  # [B,H,W]
        union = probs.sum(dim=1) + target_oh.sum(dim=1)  # [B,H,W]
        
        dice_per_pixel = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B,H,W]
        loss_per_pixel = 1.0 - dice_per_pixel
        
        return loss_per_pixel

class WeightedCEDicePerPixel(nn.Module):
    """Weighted CE + Dice loss that returns per-pixel loss map [B,H,W].
    
    This is the recommended loss for rare class handling with change-aware weighting.
    """
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        lambda_ce: float = 0.5,
        lambda_dice: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        
        self.ce = WeightedCrossEntropy(
            class_weights=class_weights,
            ignore_index=ignore_index,
            reduction='none',  # per-pixel
            label_smoothing=label_smoothing,
        )
        self.dice = PerPixelDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=1.0,
        )
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B,H,W] per-pixel loss map
        """
        ce_loss = self.ce(logits, target)  # [B,H,W]
        dice_loss = self.dice(logits, target)  # [B,H,W]
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (per-pixel).
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        num_classes: int,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = 'none',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.register_buffer('alpha', alpha if alpha is not None else None)
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W]; target: [B,H,W]
        Returns: [B,H,W] if reduction='none'
        """
        B, C, H, W = logits.shape
        
        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=1)  # [B,C,H,W]
        probs = torch.exp(log_probs)
        
        # Gather target class probabilities
        target_clamped = target.clone()
        valid = (target != self.ignore_index)
        target_clamped[~valid] = 0
        
        # Get probability of target class for each pixel
        target_oh = F.one_hot(target_clamped, num_classes=C).permute(0, 3, 1, 2).float()  # [B,C,H,W]
        p_t = (probs * target_oh).sum(dim=1)  # [B,H,W]
        log_p_t = (log_probs * target_oh).sum(dim=1)  # [B,H,W]
        
        # Focal term
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        if self.alpha is not None:
            alpha_t = (self.alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * target_oh).sum(dim=1)  # [B,H,W]
            focal_weight = alpha_t * focal_weight
        
        # Focal loss
        loss = -focal_weight * log_p_t  # [B,H,W]
        
        # Mask invalid pixels
        loss = loss * valid.float()
        
        if self.reduction == 'mean':
            return loss.sum() / (valid.sum() + 1e-8)
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class FocalCEDicePerPixel(nn.Module):
    """Focal CE + Dice loss that returns per-pixel loss map [B,H,W].
    
    Alternative to WeightedCEDicePerPixel for extreme class imbalance.
    """
    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = 255,
        lambda_focal: float = 0.5,
        lambda_dice: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        
        self.focal = FocalLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=gamma,
            ignore_index=ignore_index,
            reduction='none',
        )
        self.dice = PerPixelDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            smooth=1.0,
        )
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Returns: [B,H,W] per-pixel loss map
        """
        focal_loss = self.focal(logits, target)  # [B,H,W]
        dice_loss = self.dice(logits, target)  # [B,H,W]
        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss

class SoftDiceLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int = 255,
        smooth: float = 1.0,
        include_bg: bool = True,
        class_weights: Optional[torch.Tensor] = None,  # optional per-class weights for Dice
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.include_bg = include_bg
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,C,H,W]; target: [B,H,W]
        Computes per-class soft dice and averages (optionally weighted).
        """
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)

        # one-hot with ignore handling
        tgt_oh = one_hot_ignore_safe(target, num_classes=c, ignore_index=self.ignore_index)  # [B,C,H,W]
        valid = (target != self.ignore_index).unsqueeze(1).float()  # [B,1,H,W]

        probs = probs * valid
        tgt_oh = tgt_oh * valid

        if not self.include_bg and c > 1:
            probs = probs[:, 1:, ...]
            tgt_oh = tgt_oh[:, 1:, ...]
            c_eff = c - 1
            cw = None if self.class_weights is None else self.class_weights[1:]
        else:
            c_eff = probs.shape[1]
            cw = self.class_weights

        dims = (0, 2, 3)  # sum over B,H,W per class
        intersect = torch.sum(probs * tgt_oh, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(tgt_oh, dim=dims)

        dice_c = (2.0 * intersect + self.smooth) / (denom + self.smooth)  # [C]
        dice_loss_c = 1.0 - dice_c  # per-class loss

        if cw is not None:
            cw = cw.to(dice_loss_c.dtype).to(dice_loss_c.device)
            cw = cw[:c_eff]
            loss = (dice_loss_c * cw).sum() / (cw.sum().clamp_min(1e-8))
        else:
            loss = dice_loss_c.mean()

        return loss

class ComboSegLoss(nn.Module):
    """
    L = lambda_ce * CE(weighted) + lambda_dice * SoftDice(per-class)
    - Accepts logits or a list/tuple of logits for deep supervision. In that case, losses are averaged.
    """
    def __init__(
        self,
        class_weights_ce: Optional[torch.Tensor] = None,
        class_weights_dice: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        lambda_ce: float = 0.5,
        lambda_dice: float = 0.5,
        label_smoothing: float = 0.0,
        include_bg: bool = True,
    ):
        super().__init__()
        self.ce = WeightedCrossEntropy(
            class_weights=class_weights_ce,
            ignore_index=ignore_index,
            reduction="mean",
            label_smoothing=label_smoothing,
        )
        self.dice = SoftDiceLoss(
            ignore_index=ignore_index,
            smooth=1.0,
            include_bg=include_bg,
            class_weights=class_weights_dice,
        )
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice

    def _loss_single(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.lambda_ce * self.ce(logits, target) + self.lambda_dice * self.dice(logits, target)

    def forward(self, logits: torch.Tensor | Sequence[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, (list, tuple)):
            losses = [self._loss_single(l, target) for l in logits]
            return torch.stack(losses).mean()
        return self._loss_single(logits, target)
