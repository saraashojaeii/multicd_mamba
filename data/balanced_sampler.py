"""
Balanced patch sampler for change detection datasets.

Oversamples patches with:
- High change ratio (>τ, e.g., 1-2%)
- Rare classes/transitions (e.g., water, playground)
"""

import torch
import numpy as np
import json
import os
from torch.utils.data import Sampler
from typing import List, Optional


class BalancedChangeSampler(Sampler):
    """Sampler that oversamples patches with high change ratio and rare classes/transitions.
    
    Includes batch-level balance control and epoch-aware oversampling scheduling.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'L1', 'L2' (semantic labels)
        change_threshold: Minimum change ratio to be considered "high change" (e.g., 0.01 = 1%)
        rare_classes: List of class indices to oversample (e.g., [3, 5] for water, playground)
        oversample_factor: How much to oversample high-change/rare patches (e.g., 2.0 = 2x)
        precompute_stats: Whether to precompute change ratios (faster but uses more memory)
        max_precompute: Maximum number of samples to precompute (None = all)
        batch_balance_ratio: Target ratio of high-change to low-change samples (e.g., 0.5 = 50/50)
        use_batch_balance: Whether to enforce batch-level balance
        epoch_schedule_oversample: Whether to reduce oversampling over epochs
        oversample_warmup_epochs: Number of epochs to maintain high oversampling (default 10)
        min_oversample_factor: Minimum oversample factor after warmup (default 1.0 = no oversampling)
    """
    
    def __init__(
        self,
        dataset,
        change_threshold: float = 0.01,
        rare_classes: Optional[List[int]] = None,
        oversample_factor: float = 2.0,
        precompute_stats: bool = True,
        max_precompute: Optional[int] = None,
        stats_file: Optional[str] = None,
        batch_balance_ratio: float = 0.5,
        use_batch_balance: bool = True,
        epoch_schedule_oversample: bool = True,
        oversample_warmup_epochs: int = 10,
        min_oversample_factor: float = 1.0,
        current_epoch: int = 0,
    ):
        self.dataset = dataset
        self.change_threshold = change_threshold
        self.rare_classes = rare_classes if rare_classes is not None else []
        self.oversample_factor = oversample_factor
        self.precompute_stats = precompute_stats
        self.max_precompute = max_precompute
        self.stats_file = stats_file
        self.batch_balance_ratio = batch_balance_ratio
        self.use_batch_balance = use_batch_balance
        self.epoch_schedule_oversample = epoch_schedule_oversample
        self.oversample_warmup_epochs = oversample_warmup_epochs
        self.min_oversample_factor = min_oversample_factor
        self.current_epoch = current_epoch
        
        self.num_samples = len(dataset)
        self.high_change_indices = []
        self.low_change_indices = []  # Renamed from regular_indices for clarity
        self.rare_class_indices = []
        
        # Load from file if provided
        if self.stats_file is not None and os.path.exists(self.stats_file):
            print(f"[BalancedChangeSampler] Loading precomputed stats from: {self.stats_file}")
            self._load_stats_from_file()
        elif self.precompute_stats:
            self._precompute_sample_stats()
        else:
            # Without precomputation, treat all samples as low change
            self.low_change_indices = list(range(self.num_samples))
    
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
                
                # Categorize sample into high-change or low-change pools
                if change_ratio >= self.change_threshold:
                    self.high_change_indices.append(idx)
                else:
                    self.low_change_indices.append(idx)
                
                # Track rare classes separately (can overlap with high/low change)
                if has_rare_class:
                    self.rare_class_indices.append(idx)
                    
            except Exception as e:
                print(f"[BalancedChangeSampler] Warning: Failed to process sample {idx}: {e}")
                self.low_change_indices.append(idx)
        
        print(f"[BalancedChangeSampler] Statistics:")
        print(f"  High change (>{self.change_threshold*100:.1f}%): {len(self.high_change_indices)} samples")
        print(f"  Low change (<={self.change_threshold*100:.1f}%): {len(self.low_change_indices)} samples")
        print(f"  Rare classes: {len(self.rare_class_indices)} samples")
    
    def _load_stats_from_file(self):
        """Load precomputed statistics from JSON file."""
        try:
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
            
            # Validate stats
            if stats.get('num_samples', 0) != self.num_samples:
                print(f"Warning: Stats file has {stats.get('num_samples')} samples, dataset has {self.num_samples}")
                print("Falling back to precomputation...")
                self._precompute_sample_stats()
                return
            
            # Load indices (support both old 'regular_indices' and new 'low_change_indices')
            self.high_change_indices = stats.get('high_change_indices', [])
            self.rare_class_indices = stats.get('rare_class_indices', [])
            self.low_change_indices = stats.get('low_change_indices', stats.get('regular_indices', []))
            
            print(f"  Loaded statistics:")
            print(f"    High change: {len(self.high_change_indices)} samples")
            print(f"    Low change: {len(self.low_change_indices)} samples")
            print(f"    Rare classes: {len(self.rare_class_indices)} samples")
            
        except Exception as e:
            print(f"Error loading stats file: {e}")
            print("Falling back to precomputation...")
            self._precompute_sample_stats()
    
    def set_epoch(self, epoch: int):
        """Update current epoch for epoch-aware oversampling scheduling."""
        self.current_epoch = epoch
    
    def get_effective_oversample_factor(self) -> float:
        """Compute effective oversample factor based on current epoch."""
        if not self.epoch_schedule_oversample:
            return self.oversample_factor
        
        # Linear decay from oversample_factor to min_oversample_factor over warmup_epochs
        if self.current_epoch < self.oversample_warmup_epochs:
            # During warmup: maintain high oversampling
            return self.oversample_factor
        else:
            # After warmup: linearly decay to minimum
            decay_epochs = max(self.oversample_warmup_epochs, 1)
            progress = min(1.0, (self.current_epoch - self.oversample_warmup_epochs) / decay_epochs)
            return self.oversample_factor - progress * (self.oversample_factor - self.min_oversample_factor)
    
    def __iter__(self):
        """Generate indices with batch-level balance and epoch-aware oversampling."""
        if not self.precompute_stats:
            # Without precomputation, return all indices shuffled
            indices = list(range(self.num_samples))
            np.random.shuffle(indices)
            return iter(indices)
        
        # Get effective oversample factor for current epoch
        effective_oversample = self.get_effective_oversample_factor()
        
        if self.use_batch_balance:
            # Batch-level balance: sample from high-change and low-change pools with target ratio
            # Target: batch_balance_ratio of samples should be high-change
            
            # Calculate how many samples we want from each pool
            total_samples = self.num_samples
            target_high_change = int(total_samples * self.batch_balance_ratio)
            target_low_change = total_samples - target_high_change
            
            # Sample from high-change pool (with replacement if needed)
            if len(self.high_change_indices) > 0:
                high_change_samples = np.random.choice(
                    self.high_change_indices,
                    size=target_high_change,
                    replace=(target_high_change > len(self.high_change_indices))
                ).tolist()
            else:
                high_change_samples = []
            
            # Sample from low-change pool (with replacement if needed)
            if len(self.low_change_indices) > 0:
                low_change_samples = np.random.choice(
                    self.low_change_indices,
                    size=target_low_change,
                    replace=(target_low_change > len(self.low_change_indices))
                ).tolist()
            else:
                low_change_samples = []
            
            # Combine and shuffle
            indices = high_change_samples + low_change_samples
            
            # Add rare class oversampling on top (if factor > 1.0)
            if effective_oversample > 1.0 and len(self.rare_class_indices) > 0:
                num_oversample_rare = int(len(self.rare_class_indices) * (effective_oversample - 1.0))
                if num_oversample_rare > 0:
                    oversample_rare = np.random.choice(
                        self.rare_class_indices,
                        size=num_oversample_rare,
                        replace=True
                    ).tolist()
                    indices.extend(oversample_rare)
        else:
            # Original behavior: oversample high-change and rare-class patches
            indices = list(range(self.num_samples))
            
            # Oversample high-change patches
            num_oversample_high = int(len(self.high_change_indices) * (effective_oversample - 1.0))
            if num_oversample_high > 0:
                oversample_high = np.random.choice(
                    self.high_change_indices,
                    size=num_oversample_high,
                    replace=True
                ).tolist()
                indices.extend(oversample_high)
            
            # Oversample rare-class patches
            num_oversample_rare = int(len(self.rare_class_indices) * (effective_oversample - 1.0))
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
        
        effective_oversample = self.get_effective_oversample_factor()
        
        if self.use_batch_balance:
            # With batch balance, size is base + rare class oversampling
            base_size = self.num_samples
            oversample_rare = int(len(self.rare_class_indices) * (effective_oversample - 1.0))
            return base_size + oversample_rare
        else:
            # Original behavior
            base_size = self.num_samples
            oversample_high = int(len(self.high_change_indices) * (effective_oversample - 1.0))
            oversample_rare = int(len(self.rare_class_indices) * (effective_oversample - 1.0))
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
