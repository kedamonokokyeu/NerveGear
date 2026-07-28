"""
CFD Dataset — bridges .npz files and the PyTorch training loop.

[REWRITTEN 2026-06-07 — see CHANGELOG.md, items #1, #2, #3]
Changes vs. the original:
  * Every sample is CANONICALIZED to a left inlet (see Source/transforms.py).
  * Input grew from 5 -> 9 channels: + broadcast inlet velocity (x2) + CoordConv (x2).
    (The old 'edge one-hot' idea is unnecessary now: after canonicalization the
    inlet is always 'left', so an edge channel would be constant.)
  * Normalization stats are computed on the TRAIN SPLIT ONLY (no val leakage).
  * Optional vertical-mirror augmentation for the training set (keeps inlet on left).

Each sample returns:
    x : tensor (9, N, N)   — model input  (see transforms.build_input_channels)
    y : tensor (3, N, N)   — target: vx, vy, pressure  (canonical frame, normalized)
"""

import glob
import os

import numpy as np
import torch  # type: ignore
from torch.utils.data import Dataset, DataLoader  # type: ignore

from Source.transforms import (
    canon_k, rotate_scalar, rotate_vector_field, rotate_vector_scalar,
    build_input_channels, _NORM,
)


def _load_canonical(path):
    """Load one .npz and rotate everything into the canonical left-inlet frame."""
    d = np.load(path)
    edge = str(d['inlet_edge']) if 'inlet_edge' in d else 'left'
    k = canon_k(edge)

    sdf      = rotate_scalar(d['sdf'].astype(np.float32), k)
    mask     = rotate_scalar(d['mask'].astype(np.float32), k)
    pressure = rotate_scalar(d['pressure'].astype(np.float32), k)
    vx, vy   = rotate_vector_field(d['vx'].astype(np.float32),
                                   d['vy'].astype(np.float32), k)

    Re    = float(d['Re'])
    ux_in = float(d['ux_in'])
    uy_in = float(d['uy_in']) if 'uy_in' in d else 0.0
    ux_c, uy_c = rotate_vector_scalar(ux_in, uy_in, k)

    return dict(sdf=sdf, mask=mask, vx=vx, vy=vy, pressure=pressure,
                Re=Re, ux_c=ux_c, uy_c=uy_c)


def compute_stats(files):
    """Per-channel (mean, std) for vx, vy, pressure over the CANONICAL targets."""
    vx_all, vy_all, p_all = [], [], []
    for f in files:
        s = _load_canonical(f)
        vx_all.append(s['vx'].ravel())
        vy_all.append(s['vy'].ravel())
        p_all.append(s['pressure'].ravel())

    def ms(arrs):
        a = np.concatenate(arrs)
        return float(a.mean()), float(max(a.std(), 1e-8))

    return {'vx': ms(vx_all), 'vy': ms(vy_all), 'pressure': ms(p_all)}


class CFDDataset(Dataset):
    """
    files   : list of sample_*.npz paths
    stats   : {'vx':(mean,std), ...} for output normalization. If None, targets
              are returned un-normalized (used while computing stats).
    augment : if True, randomly vertical-mirror samples (train only).
    """

    def __init__(self, files, stats=None, augment=False):
        if len(files) == 0:
            raise FileNotFoundError("CFDDataset got an empty file list")
        self.files = files
        self.stats = stats
        self.augment = augment

    def set_stats(self, stats):
        self.stats = stats

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = _load_canonical(self.files[idx])
        sdf, mask = s['sdf'], s['mask']
        vx, vy, pressure = s['vx'], s['vy'], s['pressure']
        ux_c, uy_c = s['ux_c'], s['uy_c']

        # vertical mirror keeps the inlet on the left; rows flip, vy & uy_c flip sign
        if self.augment and np.random.rand() < 0.5:
            sdf      = np.flipud(sdf).copy()
            mask     = np.flipud(mask).copy()
            pressure = np.flipud(pressure).copy()
            vx       = np.flipud(vx).copy()
            vy       = (-np.flipud(vy)).copy()
            uy_c     = -uy_c

        x = build_input_channels(sdf, mask, s['Re'], ux_c, uy_c, _NORM)

        if self.stats is not None:
            vx       = (vx       - self.stats['vx'][0])       / self.stats['vx'][1]
            vy       = (vy       - self.stats['vy'][0])       / self.stats['vy'][1]
            pressure = (pressure - self.stats['pressure'][0]) / self.stats['pressure'][1]

        y = np.stack([vx, vy, pressure], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def make_loaders(data_dir: str, batch_size: int = 8, val_split: float = 0.15,
                 num_workers: int = 2, seed: int = 42, augment: bool = True):
    """
    Split into train/val, compute stats on TRAIN ONLY, build DataLoaders.

    Returns (train_loader, val_loader, stats).   <-- NOTE: now returns stats too.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No sample_*.npz files found in {data_dir}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_split))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_files = [files[i] for i in train_idx]
    val_files   = [files[i] for i in val_idx]

    stats = compute_stats(train_files)   # <-- train split only: no leakage

    train_set = CFDDataset(train_files, stats=stats, augment=augment)
    val_set   = CFDDataset(val_files,   stats=stats, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_files)} samples  |  Val: {len(val_files)} samples  "
          f"|  Batch: {batch_size}  |  augment={augment}")
    for kk, (mu, sd) in stats.items():
        print(f"  {kk:>10s}  mean={mu:.4f}  std={sd:.4f}")
    return train_loader, val_loader, stats


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    train_loader, val_loader, stats = make_loaders(data_dir, batch_size=4)
    x, y = next(iter(train_loader))
    print(f"\nInput  x : {x.shape}  dtype={x.dtype}  range=[{x.min():.3f}, {x.max():.3f}]")
    print(f"Target y : {y.shape}  dtype={y.dtype}  range=[{y.min():.3f}, {y.max():.3f}]")
    print("Dataset looks good.")
