"""
Balanced patch sampler for change detection datasets.

Oversamples patches with:
- High change ratio (>τ, e.g., 1-2%)
- Rare classes/transitions (e.g., water, playground)
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import List, Optional


class BalancedChangeSampler(Sampler):
    """Sampler that oversamples patches with high change ratio and rare classes/transitions.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'L1', 'L2' (semantic labels)
        change_threshold: Minimum change ratio to be considered "high change" (e.g., 0.01 = 1%)
        rare_classes: List of class indices to oversample (e.g., [3, 5] for water, playground)
        oversample_factor: How much to oversample high-change/rare patches (e.g., 2.0 = 2x)
        precompute_stats: Whether to precompute change ratios (faster but uses more memory)
        max_precompute: Maximum number of samples to precompute (None = all)
    """
    
    def __init__(
        self,
        dataset,
        change_threshold: float = 0.01,
        rare_classes: Optional[List[int]] = None,
        oversample_factor: float = 2.0,
        precompute_stats: bool = True,
        max_precompute: Optional[int] = None,
    ):
        self.dataset = dataset
        self.change_threshold = change_threshold
        self.rare_classes = rare_classes if rare_classes is not None else []
        self.oversample_factor = oversample_factor
        self.precompute_stats = precompute_stats
        self.max_precompute = max_precompute
        
        self.num_samples = len(dataset)
        self.high_change_indices = []
        self.rare_class_indices = []
        self.regular_indices = []
        
        if self.precompute_stats:
            self._precompute_sample_stats()
        else:
            # Without precomputation, treat all samples as regular
            self.regular_indices = list(range(self.num_samples))
    
    def _precompute_sample_stats(self):
        """Precompute which samples have high change ratio or rare classes."""
        print(f"[BalancedChangeSampler] Precomputing sample statistics...")
        
        max_samples = self.max_precompute if self.max_precompute is not None else self.num_samples
        max_samples = min(max_samples, self.num_samples)
        
        for idx in range(max_samples):
            try:
                sample = self.dataset[idx]
                
                # Extract labels
                if isinstance(sample, dict):
                    L1 = sample.get('L1')
                    L2 = sample.get('L2')
                else:
                    # Assume tuple/list format
                    L1, L2 = sample[2], sample[3]
                
                if L1 is None or L2 is None:
                    self.regular_indices.append(idx)
                    continue
                
                # Convert to numpy for faster computation
                if isinstance(L1, torch.Tensor):
                    L1 = L1.cpu().numpy()
                if isinstance(L2, torch.Tensor):
                    L2 = L2.cpu().numpy()
                
                # Compute change ratio
                changed_pixels = (L1 != L2).sum()
                total_pixels = L1.size if hasattr(L1, 'size') else L1.shape[0] * L1.shape[1]
                change_ratio = changed_pixels / max(total_pixels, 1)
                
                # Check for rare classes
                has_rare_class = False
                if self.rare_classes:
                    unique_classes = np.unique(np.concatenate([L1.flatten(), L2.flatten()]))
                    has_rare_class = any(cls in self.rare_classes for cls in unique_classes)
                
                # Categorize sample
                if change_ratio >= self.change_threshold:
                    self.high_change_indices.append(idx)
                elif has_rare_class:
                    self.rare_class_indices.append(idx)
                else:
                    self.regular_indices.append(idx)
                    
            except Exception as e:
                print(f"[BalancedChangeSampler] Warning: Failed to process sample {idx}: {e}")
                self.regular_indices.append(idx)
        
        print(f"[BalancedChangeSampler] Statistics:")
        print(f"  High change (>{self.change_threshold*100:.1f}%): {len(self.high_change_indices)} samples")
        print(f"  Rare classes: {len(self.rare_class_indices)} samples")
        print(f"  Regular: {len(self.regular_indices)} samples")
    
    def __iter__(self):
        """Generate indices with oversampling of high-change and rare-class patches."""
        # Base indices (all samples appear at least once)
        indices = list(range(self.num_samples))
        
        if self.precompute_stats:
            # Oversample high-change patches
            num_oversample_high = int(len(self.high_change_indices) * (self.oversample_factor - 1.0))
            if num_oversample_high > 0:
                oversample_high = np.random.choice(
                    self.high_change_indices,
                    size=num_oversample_high,
                    replace=True
                ).tolist()
                indices.extend(oversample_high)
            
            # Oversample rare-class patches
            num_oversample_rare = int(len(self.rare_class_indices) * (self.oversample_factor - 1.0))
            if num_oversample_rare > 0:
                oversample_rare = np.random.choice(
                    self.rare_class_indices,
                    size=num_oversample_rare,
                    replace=True
                ).tolist()
                indices.extend(oversample_rare)
        
        # Shuffle
        np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
        """Return effective dataset size after oversampling."""
        if not self.precompute_stats:
            return self.num_samples
        
        base_size = self.num_samples
        oversample_high = int(len(self.high_change_indices) * (self.oversample_factor - 1.0))
        oversample_rare = int(len(self.rare_class_indices) * (self.oversample_factor - 1.0))
        
        return base_size + oversample_high + oversample_rare


class WeightedRandomSamplerByChange(Sampler):
    """Alternative: Weighted random sampling based on change ratio and rare classes.
    
    This is simpler than BalancedChangeSampler but may be less effective.
    Each sample gets a weight based on its change ratio and presence of rare classes.
    """
    
    def __init__(
        self,
        dataset,
        change_threshold: float = 0.01,
        rare_classes: Optional[List[int]] = None,
        high_change_weight: float = 3.0,
        rare_class_weight: float = 2.0,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ):
        self.dataset = dataset
        self.change_threshold = change_threshold
        self.rare_classes = rare_classes if rare_classes is not None else []
        self.high_change_weight = high_change_weight
        self.rare_class_weight = rare_class_weight
        self.replacement = replacement
        
        self.dataset_size = len(dataset)
        self.num_samples = num_samples if num_samples is not None else self.dataset_size
        
        # Compute weights
        self.weights = self._compute_weights()
    
    def _compute_weights(self):
        """Compute sampling weight for each sample."""
        print(f"[WeightedRandomSamplerByChange] Computing sample weights...")
        
        weights = torch.ones(self.dataset_size, dtype=torch.float32)
        
        for idx in range(self.dataset_size):
            try:
                sample = self.dataset[idx]
                
                if isinstance(sample, dict):
                    L1 = sample.get('L1')
                    L2 = sample.get('L2')
                else:
                    L1, L2 = sample[2], sample[3]
                
                if L1 is None or L2 is None:
                    continue
                
                if isinstance(L1, torch.Tensor):
                    L1 = L1.cpu().numpy()
                if isinstance(L2, torch.Tensor):
                    L2 = L2.cpu().numpy()
                
                # Compute change ratio
                changed_pixels = (L1 != L2).sum()
                total_pixels = L1.size if hasattr(L1, 'size') else L1.shape[0] * L1.shape[1]
                change_ratio = changed_pixels / max(total_pixels, 1)
                
                # Boost weight for high change
                if change_ratio >= self.change_threshold:
                    weights[idx] *= self.high_change_weight
                
                # Boost weight for rare classes
                if self.rare_classes:
                    unique_classes = np.unique(np.concatenate([L1.flatten(), L2.flatten()]))
                    if any(cls in self.rare_classes for cls in unique_classes):
                        weights[idx] *= self.rare_class_weight
                        
            except Exception as e:
                print(f"[WeightedRandomSamplerByChange] Warning: Failed to process sample {idx}: {e}")
        
        print(f"[WeightedRandomSamplerByChange] Weight statistics:")
        print(f"  Mean: {weights.mean():.3f}, Std: {weights.std():.3f}")
        print(f"  Min: {weights.min():.3f}, Max: {weights.max():.3f}")
        
        return weights
    
    def __iter__(self):
        """Sample indices according to weights."""
        indices = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples
