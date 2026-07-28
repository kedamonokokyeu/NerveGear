import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Source.unet import UNet
from Source.dataset import make_loaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# [MODIFIED 2026-06-07 — see CHANGELOG.md, items #1, #5, #7, #8]
#   * Adam -> AdamW (decoupled weight decay actually regularizes with Adam)
#   * loss = weighted MSE + divergence (mass-conservation) penalty + gradient (sharpness) loss
#   * checkpoint / early-stop on mean VELOCITY rel-L2 (the metric we actually care about),
#     not raw val MSE
#   * make_loaders now returns stats explicitly (train-split only)

"""Hyperparameters Block"""
DATA_DIR = 'data/raw'
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3                      # updated per-epoch by the scheduler
WEIGHT_DECAY = 1e-4
LOSS_WEIGHTS = [2.0, 2.0, 0.5]   # vx, vy, pressure  (emphasize velocity)
LAMBDA_DIV  = 0.05            # mass-conservation penalty weight  (TUNE: 0.01-0.5)
LAMBDA_GRAD = 0.10           # gradient/sharpness loss weight     (TUNE: 0.05-0.3)
EARLY_STOP_PATIENCE = 15     # epochs without vel-rel-L2 improvement before stopping


def make_criterion(stats):
    """Weighted MSE + divergence penalty + gradient loss.

    `stats` gives (mean,std) per channel so we can DENORMALIZE velocity before
    computing physical divergence (du/dx + dv/dy ~ 0 in incompressible flow).
    """
    mse = nn.MSELoss()
    vx_m, vx_s = stats['vx']
    vy_m, vy_s = stats['vy']

    def weighted_mse(pred, target):
        return sum(LOSS_WEIGHTS[c] * mse(pred[:, c:c+1], target[:, c:c+1]) for c in range(3))

    def divergence(pred, mask_ch):
        # denormalize velocity components back to physical (lattice) units
        vx = pred[:, 0] * vx_s + vx_m
        vy = pred[:, 1] * vy_s + vy_m
        # central differences (grid spacing = 1 cell): x is axis=-1, y is axis=-2
        dvx_dx = (vx[:, 1:-1, 2:] - vx[:, 1:-1, :-2]) * 0.5
        dvy_dy = (vy[:, 2:, 1:-1] - vy[:, :-2, 1:-1]) * 0.5
        div = dvx_dx + dvy_dy                       # (B, N-2, N-2)
        fluid = (mask_ch[:, 1:-1, 1:-1] < 0.5).float()
        return (div.pow(2) * fluid).sum() / (fluid.sum() + 1e-8)

    def gradient_loss(pred, target):
        # L1 on first differences of the velocity channels — sharpens wakes
        p, t = pred[:, :2], target[:, :2]
        pgx = p[:, :, :, 1:] - p[:, :, :, :-1]
        tgx = t[:, :, :, 1:] - t[:, :, :, :-1]
        pgy = p[:, :, 1:, :] - p[:, :, :-1, :]
        tgy = t[:, :, 1:, :] - t[:, :, :-1, :]
        return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()

    def criterion(pred, target, x):
        mask_ch = x[:, 1]                            # channel 1 == raw mask
        return (weighted_mse(pred, target)
                + LAMBDA_DIV * divergence(pred, mask_ch)
                + LAMBDA_GRAD * gradient_loss(pred, target))

    return criterion


if __name__ == '__main__':
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    model = UNet().to(device)                        # in_channels=9 by default now
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_loader, val_loader, dataset_stats = make_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    criterion = make_criterion(dataset_stats)

    best_vel_rel = float('inf')
    epochs_since_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_num = torch.zeros(3)     # per-channel sum of squared errors
        val_den = torch.zeros(3)     # per-channel sum of squared targets
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                prediction = model(x)
                val_loss += criterion(prediction, y, x).item()
                val_num += (prediction - y).pow(2).sum(dim=(0, 2, 3)).cpu()
                val_den += y.pow(2).sum(dim=(0, 2, 3)).cpu()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        rel_l2 = (val_num / val_den.clamp(min=1e-8)).sqrt()      # (3,) vx, vy, p
        vel_rel = 0.5 * (rel_l2[0].item() + rel_l2[1].item())    # mean velocity rel-L2

        print(f"Epoch {epoch+1:>3}/{EPOCHS}  |  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"|  rel-L2  vx={rel_l2[0]:.3f}  vy={rel_l2[1]:.3f}  p={rel_l2[2]:.3f}  "
              f"|  vel-relL2={vel_rel:.3f}")

        if vel_rel < best_vel_rel:
            best_vel_rel = vel_rel
            epochs_since_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            stats_path = CHECKPOINT_PATH.replace('.pth', '_stats.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'normalization': {k: list(v) for k, v in dataset_stats.items()},
                    'val_loss': val_loss,
                    'rel_l2': {'vx': rel_l2[0].item(), 'vy': rel_l2[1].item(),
                               'pressure': rel_l2[2].item()},
                    'in_channels': 9,
                    'canonical_inlet': 'left',
                }, f, indent=2)
            print(f"  saved checkpoint  (vel-relL2={vel_rel:.3f}  "
                  f"vx={rel_l2[0]:.3f}  vy={rel_l2[1]:.3f}  p={rel_l2[2]:.3f})")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= EARLY_STOP_PATIENCE:
                print(f"  early stop: no vel-relL2 improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs (best={best_vel_rel:.3f})")
                break
