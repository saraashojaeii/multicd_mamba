"""
Precompute balanced sampler statistics and save to disk.
Run this once before training to avoid slow startup.

Usage:
    python precompute_sampler_stats.py --config config/second_cdmamba/cdmamba_seg_cd_balanced.json
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as Data
from core.logger import parse as parse_cfg


def precompute_stats(dataset, change_threshold=0.01, rare_classes=None, max_samples=None):
    """Precompute sample statistics for balanced sampling.
    
    Args:
        dataset: Dataset to analyze
        change_threshold: Minimum change ratio to be considered "high change"
        rare_classes: List of class indices to oversample
        max_samples: Maximum number of samples to analyze (None = all)
    
    Returns:
        dict with keys:
            - high_change_indices: List of sample indices with high change ratio
            - rare_class_indices: List of sample indices containing rare classes
            - regular_indices: List of remaining sample indices
            - change_ratios: Dict mapping sample_idx -> change_ratio
            - class_presence: Dict mapping sample_idx -> set of present classes
    """
    rare_classes = rare_classes if rare_classes is not None else []
    num_samples = len(dataset)
    max_samples = min(max_samples, num_samples) if max_samples is not None else num_samples
    
    high_change_indices = []
    rare_class_indices = []
    regular_indices = []
    change_ratios = {}
    class_presence = {}
    
    print(f"Precomputing statistics for {max_samples}/{num_samples} samples...")
    
    for idx in tqdm(range(max_samples)):
        try:
            sample = dataset[idx]
            
            # Extract labels
            if isinstance(sample, dict):
                L1 = sample.get('L1')
                L2 = sample.get('L2')
            else:
                # Assume tuple/list format
                L1, L2 = sample[2], sample[3]
            
            if L1 is None or L2 is None:
                regular_indices.append(idx)
                continue
            
            # Convert to numpy
            if isinstance(L1, torch.Tensor):
                L1 = L1.cpu().numpy()
            if isinstance(L2, torch.Tensor):
                L2 = L2.cpu().numpy()
            
            # Compute change ratio
            changed_mask = (L1 != L2)
            change_ratio = changed_mask.sum() / changed_mask.size
            change_ratios[idx] = float(change_ratio)
            
            # Get unique classes present
            unique_classes = set(np.unique(L1).tolist() + np.unique(L2).tolist())
            class_presence[idx] = list(unique_classes)
            
            # Categorize sample
            has_high_change = change_ratio > change_threshold
            has_rare_class = any(cls in rare_classes for cls in unique_classes)
            
            if has_high_change:
                high_change_indices.append(idx)
            if has_rare_class:
                rare_class_indices.append(idx)
            if not (has_high_change or has_rare_class):
                regular_indices.append(idx)
        
        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            regular_indices.append(idx)
            continue
    
    stats = {
        'high_change_indices': high_change_indices,
        'rare_class_indices': rare_class_indices,
        'regular_indices': regular_indices,
        'change_ratios': change_ratios,
        'class_presence': class_presence,
        'num_samples': num_samples,
        'max_precomputed': max_samples,
        'change_threshold': change_threshold,
        'rare_classes': rare_classes
    }
    
    print(f"\nStatistics:")
    print(f"  High change samples: {len(high_change_indices)} ({len(high_change_indices)/max_samples*100:.1f}%)")
    print(f"  Rare class samples: {len(rare_class_indices)} ({len(rare_class_indices)/max_samples*100:.1f}%)")
    print(f"  Regular samples: {len(regular_indices)} ({len(regular_indices)/max_samples*100:.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Precompute balanced sampler statistics')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default=None, help='Output path for stats file (default: auto)')
    parser.add_argument('--phase', type=str, default='train', help='Dataset phase (train/val/test)')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    opt = parse_cfg(args.config)
    
    # Get dataset config
    dataset_opt = opt['datasets'][args.phase]
    
    # Get sampler config
    sampler_config = opt['train'].get('balanced_sampler', {})
    change_threshold = sampler_config.get('change_threshold', 0.01)
    rare_classes = sampler_config.get('rare_classes', [3, 5])
    max_precompute = sampler_config.get('max_precompute', None)
    
    print(f"\nSampler configuration:")
    print(f"  Change threshold: {change_threshold}")
    print(f"  Rare classes: {rare_classes}")
    print(f"  Max precompute: {max_precompute}")
    
    # Create dataset
    print(f"\nCreating {args.phase} dataset...")
    dataset = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=args.phase)
    print(f"Dataset size: {len(dataset)}")
    
    # Precompute statistics
    stats = precompute_stats(
        dataset,
        change_threshold=change_threshold,
        rare_classes=rare_classes,
        max_samples=max_precompute
    )
    
    # Determine output path
    if args.output is None:
        dataset_root = dataset_opt.get('datasetroot', 'unknown')
        dataset_name = os.path.basename(dataset_root)
        output_dir = os.path.join(os.path.dirname(args.config), 'sampler_stats')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{dataset_name}_{args.phase}_sampler_stats.json')
    else:
        output_path = args.output
    
    # Save statistics
    print(f"\nSaving statistics to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Done! Statistics saved to: {output_path}")
    print(f"\nTo use these statistics, add to your config:")
    print(f'  "balanced_sampler": {{')
    print(f'    "precompute_stats": false,')
    print(f'    "stats_file": "{output_path}"')
    print(f'  }}')


if __name__ == '__main__':
    main()
