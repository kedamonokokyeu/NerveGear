"""
Training Data Generator

Pick a random shape (circle, rectangle, blob, naca, polygon)
Pick random Re and ux_in within stable ranges
Run LBM to steady state
Save everything as a compressed .npz file

Output files go into data/raw/ — one .npz per sample.
Each .npz contains:
    sdf      (N, N) float32 — signed distance field of the shape
    mask     (N, N) float32 — 1=solid, 0=fluid
    vx       (N, N) float32 — x-velocity at steady state
    vy       (N, N) float32 — y-velocity at steady state
    pressure (N, N) float32 — pressure at steady state
    Re       scalar float   — Reynolds number used
    ux_in    scalar float   — inlet velocity used

    python generate_data.py               # generates 200 samples
    python generate_data.py --n 1000      # generates 1000 samples
    python generate_data.py --n 50 --res 64   # 64x64 grid (faster, for testing)
"""

import argparse
import glob
import os
import sys
import time
import multiprocessing
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Source.sdf_generator import make_grid, random_shape
from Source.lbm_solver import run_lbm

# range for reynold numbers
RE_MIN,   RE_MAX   = 50,  400

# inlet speed range (total magnitude) — must stay below 0.2 for lattice Mach stability
UXIN_MIN, UXIN_MAX = 0.05, 0.15

# random edges to define random points for inlet flow 
EDGES = ['left', 'right', 'top', 'bottom']
OPPOSITE_EDGE = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}

# interprets direction of inlet velocity, e.g. left inlet flow results in a [1 0] vector 
EDGE_NORMAL   = {'left': (1., 0.), 'right': (-1., 0.), 'top': (0., -1.), 'bottom': (0., 1.)} 

# angular deviation max value to prevent parallel wind flows
ANGLE_PERTURB = np.pi / 6

N_STEPS = 3000


def _run_one(job):
    i          = job['i']
    out_dir    = job['out_dir']
    sdf        = job['sdf']
    mask       = job['mask']
    meta       = job['meta']
    Re         = job['Re']
    ux_in      = job['ux_in']
    uy_in      = job['uy_in']
    inlet_edge = job['inlet_edge']
    n_steps    = job['n_steps']

    try:
        vx, vy, pressure = run_lbm(mask, ux_in=ux_in, uy_in=uy_in,
                                    inlet_edge=inlet_edge, Re=Re, n_steps=n_steps)
    except Exception as e:
        return {'ok': False, 'i': i, 'reason': str(e)}

    max_v = float(np.sqrt(vx**2 + vy**2).max())
    if np.isnan(max_v) or max_v > 0.5:
        return {'ok': False, 'i': i, 'reason': f'unstable max_v={max_v:.3f}'}

    fname = os.path.join(out_dir, f"sample_{i:05d}.npz")
    np.savez_compressed(
        fname,
        sdf        = sdf,
        mask       = mask,
        vx         = vx,
        vy         = vy,
        pressure   = pressure,
        Re         = np.float32(Re),
        ux_in      = np.float32(ux_in),
        uy_in      = np.float32(uy_in),
        inlet_edge = np.str_(inlet_edge),
    )

    return {
        'ok':        True,
        'i':         i,
        'shape':     meta['shape_type'],
        'Re':        Re,
        'ux_in':     ux_in,
        'uy_in':     uy_in,
        'inlet':     inlet_edge,
        'max_v':     max_v,
    }

def generate(n_samples: int, grid_res: int, out_dir: str, seed: int, n_workers: int):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    gx, gy = make_grid(N=grid_res)

    print(f"Generating {n_samples} samples  |  grid={grid_res}x{grid_res}  "
          f"|  workers={n_workers}  |  seed={seed}")
    print(f"Output dir: {out_dir}\n")

    jobs = []
    skipped_edge = 0

    for i in range(n_samples):
        Re         = float(rng.uniform(RE_MIN, RE_MAX))
        u_in       = float(rng.uniform(UXIN_MIN, UXIN_MAX))
        inlet_edge = str(rng.choice(EDGES))
        outlet_edge = OPPOSITE_EDGE[inlet_edge]

        # ±30° keeps the flow from being nearly parallel to the wall (which would be unstable)
        delta = float(rng.uniform(-ANGLE_PERTURB, ANGLE_PERTURB))
        nx, ny = EDGE_NORMAL[inlet_edge]
        ux_in = float(u_in * (nx * np.cos(delta) - ny * np.sin(delta))) # changes the inlet velocity flow according to the angle
        uy_in = float(u_in * (nx * np.sin(delta) + ny * np.cos(delta)))

        sdf, mask, meta = random_shape(gx, gy, rng=rng)

        # drop shapes that touch the SDF
        def _touches(edge):
            if edge == 'left':   return bool(mask[:, 0].any())
            if edge == 'right':  return bool(mask[:, -1].any())
            if edge == 'top':    return bool(mask[-1, :].any())
            if edge == 'bottom': return bool(mask[0, :].any())
        if _touches(inlet_edge) or _touches(outlet_edge):
            skipped_edge += 1
            continue

        jobs.append({
            'i':          i,
            'out_dir':    out_dir,
            'sdf':        sdf,
            'mask':       mask,
            'meta':       meta,
            'Re':         Re,
            'ux_in':      ux_in,
            'uy_in':      uy_in,
            'inlet_edge': inlet_edge,
            'n_steps':    N_STEPS,
        })

    print(f"Built {len(jobs)} jobs  ({skipped_edge} edge-touching shapes skipped)\n")

    # Pool.imap_unordered sends one job to each free worker and yields results as they finish --- we don't have to wait for the whole batch.
    # Allows for live printing in parallel processing

    counts   = {s: 0 for s in ['circle', 'rectangle', 'blob', 'naca', 'polygon']}
    failures = 0
    done     = 0
    t0       = time.time()

    with multiprocessing.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_run_one, jobs):
            done += 1

            if result['ok']:
                counts[result['shape']] += 1
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(jobs) - done) / max(rate, 1e-6)
                print(f"  [{done:>5}/{len(jobs)}]  {result['shape']:<12}  "
                      f"inlet={result['inlet']:<6}  Re={result['Re']:>6.1f}  "
                      f"ux={result['ux_in']:+.3f}  uy={result['uy_in']:+.3f}  "
                      f"max|u|={result['max_v']:.4f}  ETA {eta/60:.1f}min")
            else:
                failures += 1
                print(f"  [FAIL] sample {result['i']}  —  {result['reason']}")

    total_saved = sum(counts.values())
    print(f"\nFinished. {total_saved} samples saved, {failures} failed, "
          f"{skipped_edge} edge-skipped.")
    print(f"Shape breakdown: {counts}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Files in: {out_dir}")



if __name__ == '__main__':
    # The 'if __name__ == "__main__"' guard is REQUIRED on Windows for
    # multiprocessing — without it, every worker would try to spawn more
    # workers recursively and crash.

    parser = argparse.ArgumentParser(description='Generate LBM training data')
    parser.add_argument('--n',       type=int, default=1500, help='Number of samples') # upped sample number to 1500 for more training data
    parser.add_argument('--res',     type=int, default=128, help='Grid resolution (NxN)')
    parser.add_argument('--out',     type=str, default='data/raw', help='Output directory')
    parser.add_argument('--seed',    type=int, default=42,  help='Random seed')
    parser.add_argument('--workers', type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help='Number of parallel workers (default: all cores minus 1)')
    parser.add_argument('--clean', action='store_true',
                        help='Delete all existing sample_*.npz files in the output directory before generating')
    args = parser.parse_args()

    if args.clean and os.path.isdir(args.out):
        existing = glob.glob(os.path.join(args.out, 'sample_*.npz'))
        if existing:
            print(f"--clean: removing {len(existing)} existing files from {args.out}")
            for f in existing:
                os.remove(f)

    generate(
        n_samples = args.n,
        grid_res  = args.res,
        out_dir   = args.out,
        seed      = args.seed,
        n_workers = args.workers,
    )
