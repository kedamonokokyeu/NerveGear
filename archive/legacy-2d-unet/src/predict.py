"""
Load the trained checkpoint and run single-sample inference.

The model is now trained in a CANONICAL left-inlet frame with 9 input channels.
So inference must:
    1. rotate the user's mask + inlet vector so the inlet becomes 'left'
    2. build the 9-channel input via Source.transforms.build_input_channels
    3. run the forward pass, denormalize
    4. rotate vx, vy, pressure BACK to the user's original frame
Steps 1 & 4 are handled with the shared helpers in Source/transforms.py.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Source.unet import UNet
from Source.dataset import CFDDataset, compute_stats
from Source.transforms import (
    canon_k, rotate_scalar, rotate_vector_scalar,
    inverse_rotate_scalar, inverse_rotate_vector_field,
    build_input_channels,
)


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    model = UNet().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def load_stats(checkpoint_path: str, data_dir: str) -> dict:
    stats_path = checkpoint_path.replace('.pth', '_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            raw = json.load(f)
        norm = raw.get('normalization', raw)
        return {k: tuple(v) for k, v in norm.items()}
    print("Stats file not found, recomputing from data...")
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
    return compute_stats(files)


def mask_to_sdf(mask: np.ndarray):
    N = mask.shape[0]
    pixel_size = 1.0 / N
    dist_outside = distance_transform_edt(1 - mask)
    dist_inside = distance_transform_edt(mask)
    return ((dist_outside - dist_inside) * pixel_size).astype(np.float32)


def build_input_tensor(mask: np.ndarray, Re: float, ux_in: float,
                       uy_in: float = 0.0, inlet_edge: str = 'left'):
    """
    Returns (tensor (1,9,N,N), k) where k is the rotation applied to reach the
    canonical left-inlet frame — keep it to rotate the prediction back.
    """
    k = canon_k(inlet_edge)
    mask_c = rotate_scalar(mask.astype(np.float32), k)          # mask in canonical frame
    sdf_c = mask_to_sdf(mask_c)                                 # SDF of the rotated mask
    ux_c, uy_c = rotate_vector_scalar(ux_in, uy_in, k)         # inlet vector -> canonical

    x = build_input_channels(sdf_c, mask_c, Re, ux_c, uy_c)
    return torch.from_numpy(x).unsqueeze(0), k


def predict(model, x, stats, device, k=0):
    """Forward pass, denormalize, then rotate outputs back to the original frame."""
    with torch.no_grad():
        pred = model(x.to(device))
    pred = pred.squeeze(0).cpu().numpy()

    mu_vx, std_vx = stats['vx']
    mu_vy, std_vy = stats['vy']
    mu_p,  std_p  = stats['pressure']

    vx = pred[0] * std_vx + mu_vx
    vy = pred[1] * std_vy + mu_vy
    p  = pred[2] * std_p  + mu_p

    # back to the user's frame
    vx, vy = inverse_rotate_vector_field(vx, vy, k)
    p = inverse_rotate_scalar(p, k)
    return {'vx': vx, 'vy': vy, 'pressure': p}


def main():
    parser = argparse.ArgumentParser(description="AXIOM CFD surrogate — single-sample inference")
    parser.add_argument('--mask', required=True,
                        help=".npy file containing a binary mask  (NxN, 1=solid 0=fluid)")
    parser.add_argument('--Re', type=float, default=200.0)
    parser.add_argument('--ux_in', type=float, default=0.10)
    parser.add_argument('--uy_in', type=float, default=0.05)
    parser.add_argument('--inlet_edge', default='left',
                        help="left | right | top | bottom")
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--data_dir', default='data/raw')
    parser.add_argument('--out', default='results/prediction.npz')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    stats = load_stats(args.checkpoint, args.data_dir)

    raw_mask = np.load(args.mask).astype(np.float32)
    x, k = build_input_tensor(raw_mask, args.Re, args.ux_in, args.uy_in, args.inlet_edge)
    print(f"Input tensor shape: {tuple(x.shape)}  |  Re={args.Re}  "
          f"ux_in={args.ux_in}  uy_in={args.uy_in}  inlet={args.inlet_edge}  (canon k={k})")

    result = predict(model, x, stats, device, k=k)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, **result)
    print(f"\nSaved -> {args.out}")
    for kk, v in result.items():
        print(f"  {kk:>10s}  shape={v.shape}  range=[{v.min():.5f}, {v.max():.5f}]")


if __name__ == '__main__':
    main()
