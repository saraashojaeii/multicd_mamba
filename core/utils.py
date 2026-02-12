import torch
import os
import random
import numpy as np

def estimate_class_counts(dataloader, num_classes: int, ignore_index: int = 255, max_batches=None):
    counts = torch.zeros(num_classes, dtype=torch.long)
    seen = 0
    for batch in dataloader:
        # adapt to your batch structure
        # assume batch = {"image": X, "label": Y} or (X, Y)
        if isinstance(batch, dict):
            y = batch["L1"]
        else:
            y = batch[1]
        y = y.long()  # [B,H,W]
        mask = (y != ignore_index) & (y >= 0) & (y < num_classes)
        for cls in range(num_classes):
            counts[cls] += (mask & (y == cls)).sum().item()
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    return counts

def estimate_transition_matrix(dataloader, num_classes: int, ignore_index: int = 255, max_batches=None):
    """Compute transition matrix [C, C] from GT labels (changed pixels only).
    
    Returns:
        transition_matrix: [C, C] tensor with counts of transitions from class i to class j
    """
    transition_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    seen = 0
    for batch in dataloader:
        if isinstance(batch, dict):
            y1 = batch.get("L1")
            y2 = batch.get("L2")
        else:
            # Assume (X1, X2, Y1, Y2) or similar
            y1, y2 = batch[2], batch[3]
        
        if y1 is None or y2 is None:
            continue
            
        y1 = y1.long()
        y2 = y2.long()
        
        # Only count changed pixels
        changed_mask = (y1 != y2)
        valid_mask = (y1 != ignore_index) & (y2 != ignore_index) & \
                     (y1 >= 0) & (y1 < num_classes) & \
                     (y2 >= 0) & (y2 < num_classes)
        mask = changed_mask & valid_mask
        
        if mask.any():
            y1_chg = y1[mask].cpu().numpy()
            y2_chg = y2[mask].cpu().numpy()
            for from_cls, to_cls in zip(y1_chg, y2_chg):
                transition_matrix[from_cls, to_cls] += 1
        
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    
    return transition_matrix

def compute_transition_weights(transition_matrix: torch.Tensor, method: str = "inverse_frequency", 
                               eps: float = 1e-8, smooth: float = 1.0) -> torch.Tensor:
    """Compute per-transition weights from transition matrix to rebalance rare transitions.
    
    Args:
        transition_matrix: [C, C] tensor with transition counts
        method: "inverse_frequency" or "sqrt_inverse"
        eps: small constant to avoid division by zero
        smooth: smoothing factor (add to all counts before computing weights)
    
    Returns:
        weights: [C, C] tensor with weights for each transition
    """
    tm = transition_matrix.float() + smooth
    total = tm.sum()
    
    if total < eps:
        # No transitions observed, return uniform weights
        return torch.ones_like(tm)
    
    freq = tm / total  # [C, C] frequencies
    
    if method == "inverse_frequency":
        # w[i,j] = median(freq) / freq[i,j]
        median_freq = torch.median(freq[freq > 0]) if (freq > 0).any() else torch.tensor(1.0)
        weights = median_freq / (freq + eps)
    elif method == "sqrt_inverse":
        # Softer rebalancing: w[i,j] = sqrt(median / freq[i,j])
        median_freq = torch.median(freq[freq > 0]) if (freq > 0).any() else torch.tensor(1.0)
        weights = torch.sqrt(median_freq / (freq + eps))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Clamp to avoid extreme weights
    weights = weights.clamp(min=0.1, max=10.0)
    
    return weights

def set_seed(seed: int):
    """Ensure reproducibility across runs."""
    if seed is None:
        return  # do nothing if no seed is provided
    
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    # Make CUDA deterministic (may slow things down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Control hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Random seed set to {seed}")


def normalize_change_target(seg1: torch.Tensor | None,
                            seg2: torch.Tensor | None,
                            change_gt: torch.Tensor | None) -> torch.Tensor:
    """Return binary change target of shape [B, H, W] (dtype long, {0,1}).

    Handles cases where:
    - change_gt is provided in various formats (NCHW with C=1, NHWC RGB, or [B,H,W] int/float)
    - or must be derived from seg1 and seg2 which may be RGB (NHWC), one-hot/logits (NCHW, C>1),
      or single-channel (NCHW, C=1 or [B,H,W]).
    """
    def _to_index_mask(x: torch.Tensor) -> torch.Tensor:
        # Convert arbitrary segmentation label tensor to [B,H,W] integer indices
        if x.dim() == 4:
            # NCHW
            if x.size(1) == 1:
                return x.squeeze(1).long()
            # NHWC (e.g., RGB mask)
            if x.size(-1) == 3 and x.shape[1] != 3:
                # Assume channels-last
                return x.any(dim=-1).long()  # fallback to binary presence per-pixel
            # Multi-channel: take argmax as class indices
            return torch.argmax(x, dim=1).long()
        elif x.dim() == 3:
            # Already [B,H,W]
            return x.long()
        else:
            raise ValueError(f"Unsupported seg shape: {tuple(x.shape)}")

    if change_gt is not None:
        c = change_gt
        # If NHWC RGB -> any over last channel
        if c.dim() == 4 and c.size(-1) == 3 and (c.shape[1] != 3):
            c = c.any(dim=-1).long()
        elif c.dim() == 4 and c.size(1) == 1:
            c = c.squeeze(1).long()
        elif c.dim() == 3:
            # [B,H,W] possibly float/binary
            c = (c > 0).long()
        else:
            # As a conservative fallback
            c = _to_index_mask(c)
            c = (c > 0).long()
        return c

    # Derive from seg1 and seg2
    if seg1 is None or seg2 is None:
        raise ValueError("seg1/seg2 required when change_gt is None")

    # Handle RGB NHWC
    if seg1.dim() == 4 and seg1.size(-1) == 3 and (seg1.shape[1] != 3) and \
       seg2.dim() == 4 and seg2.size(-1) == 3 and (seg2.shape[1] != 3):
        change = (seg1 != seg2).any(dim=-1).long()
        return change

    # Convert to class indices if needed
    s1 = _to_index_mask(seg1)
    s2 = _to_index_mask(seg2)
    change = (s1 != s2).long()
    return change

def create_color_mask(tensor, num_classes: int = 10, colormap=None):
    """Convert a 2-D label tensor/ndarray to an RGB image with a categorical colormap.

    This is used for logging multi-class segmentation masks to wandb so that they
    appear in color instead of a binary/grayscale mask.
    
    Args:
        tensor: 2D label tensor/array or 3D RGB image
        num_classes: Number of classes in the dataset
        colormap: Optional list of RGB colors [[R,G,B], ...]. If None, uses default SECOND colormap.
    """
    import numpy as _np
    import matplotlib as _mpl

    # Convert to numpy array
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = _np.asarray(tensor)

    # Remove singleton dimensions if they exist (e.g. 1×H×W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = _np.squeeze(arr, axis=0)
    
    # Handle case where ground truth is already RGB (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Already an RGB image, return as uint8
        return arr.astype(_np.uint8)
    
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask or 3-D RGB image, got shape {arr.shape}")

    h, w = arr.shape
    unique_vals = _np.unique(arr)
    
    # Use provided colormap or default to SECOND colormap
    if colormap is None:
        cmap = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]
    else:
        cmap = colormap

    rgb = _np.zeros((h, w, 3), dtype=_np.uint8)
    
    # Custom color mapping to ensure class 0 is visible (not black)
    colors = []
    for i in range(min(num_classes, len(cmap))):
        color = _np.array(cmap[i])
        colors.append(color.astype(_np.uint8))
    
    # Apply color mapping
    for cls in range(min(num_classes, len(colors))):
        if cls in unique_vals:
            rgb[arr == cls] = colors[cls]
    
    return rgb