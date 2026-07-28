# Archive — the 2-D LBM + U-Net prototype

Frozen reference, not part of the product. Kept because this is where the
U-Net approach was learned; nothing in `packages/` imports it and nothing
ever will.

What's here:

- `src/` — the original pipeline: `lbm_solver.py` (lattice-Boltzmann ground
  truth), `generate_data.py` + `dataset.py` + `transforms.py` (training data),
  `unet.py` + `train.py` (the model), `predict.py`, `sdf_generator.py`.
- `checkpoints/` — the trained weights (119 MB). Inference still possible
  with `src/predict.py` + `requirements.txt`; the 298 MB training dataset was
  deleted in the 2026-07 deep-clean, so retraining means regenerating data
  with `generate_data.py` first.
- `SAO 2D Prototype.pdf` — the write-up from that era.

Everything else from the old `legacy-2d/` tree (2-D designer frontend,
physics-demo layer, training data) was deleted in the same clean.
