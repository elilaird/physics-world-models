# Physics World Models

Comparing neural network architectures for learning physics dynamics as world models. Models are trained on simulated physics environments and evaluated on how well different inductive biases (discrete jumps, Newtonian mechanics, Hamiltonian structure) capture the true dynamics — including energy conservation and temporal generalization across timesteps.

## Setup

```bash
conda activate dino_wm
pip install hydra-core omegaconf
```

**Core dependencies:** PyTorch, torchdiffeq, hydra-core, omegaconf, matplotlib, opencv-python

## Quick Start

```bash
# Train with defaults (oscillator + jump model)
python train.py

# Train a specific env + model combo
python train.py env=pendulum model=port_hamiltonian

# Override training hyperparams
python train.py training.epochs=80 training.lr=5e-4 data.n_seqs=500

# Sweep across models
python train.py --multirun model=jump,lstm,newtonian,velocity,port_hamiltonian

# Evaluate a trained checkpoint
python evaluate.py checkpoint=outputs/<date>/<time>/best_model.pt
```

Checkpoints, logs, and plots are saved to `outputs/<date>/<time>/<model_name>/` by Hydra.

## Dataset Generation

By default, `train.py` generates data on the fly each run. For reproducibility and efficiency — especially when sweeping across models — you can pre-generate a dataset once and reuse it:

```bash
# Generate a dataset (saved to datasets/<env_name>/<timestamp>/)
python generate_dataset.py
python generate_dataset.py env=pendulum data.n_seqs=1000 data.seq_len=200

# Custom split ratios (default: 80/10/10 train/val/test)
python generate_dataset.py env=spaceship data.n_seqs=2000 data.val_split=0.15 data.test_split=0.15
```

This produces `train.pt`, `val.pt`, `test.pt`, and `metadata.pt` in the output directory. Each file contains pre-stacked tensors ready for loading.

To train on a saved dataset, pass `dataset_path`:

```bash
# Single model
python train.py dataset_path=datasets/oscillator/<timestamp>

# Sweep models — all share the exact same data
python train.py --multirun model=jump,lstm,newtonian,velocity,port_hamiltonian \
    dataset_path=datasets/oscillator/<timestamp>
```

Short paths are resolved under `datasets/` automatically. Without `dataset_path`, training generates data on the fly as before.

## Environments

| Environment | State | Actions | Config |
|---|---|---|---|
| Forced Oscillator | `[x, v]` (2D) | 3 (force left/none/right) | `env=oscillator` |
| Forced Pendulum | `[theta, omega]` (2D) | 3 (torque) | `env=pendulum` |
| 2-Body Spaceship | `[qx, qy, vx, vy]` (4D) | 9 (8 thrusters + idle) | `env=spaceship` |
| 3-Body Gravity | `[pos, vel]` (12D) | 9 (thrust on body 0) | `env=three_body` |

All environments use discrete action spaces and integrate ground-truth dynamics with `torchdiffeq.odeint` (except three-body which uses symplectic Euler). Variable parameters (e.g. damping) are randomized per sequence to force generalization.

## Models

| Model | Type | Inductive Bias | Config |
|---|---|---|---|
| JumpModel | Discrete | Residual MLP: `x_{t+1} = x_t + f(x_t, a_t)` | `model=jump` |
| LSTMModel | Discrete | Recurrent with residual connection | `model=lstm` |
| FirstOrderODENet | ODE | Learns `dx/dt = NN(x, a)` directly | `model=first_order_ode` |
| NewtonianDynamicsModel | ODE | `d/dt[x,v] = [v, f(x,v,a)]` with learned damping | `model=newtonian` |
| VelocityDynamicsModel | ODE | Learns velocity, shares integrator interface | `model=velocity` |
| PortHamiltonianModel | ODE | Learns H(q,p), symplectic dynamics via autograd | `model=port_hamiltonian` |
| VisualWorldModel | Visual | VQ-VAE + latent predictor for pixel observations | `model=visual_world_model` |

ODE models are automatically wrapped with `TrajectoryMatchingModel` for integration via `torchdiffeq`. This means they naturally handle variable timesteps — no retraining needed to evaluate at different dt values.

The **VisualWorldModel** operates on pixel observations rather than state vectors. It uses a VQ-VAE (encoder → vector quantizer → decoder) for image reconstruction, paired with a residual-MLP latent predictor that forecasts the next quantized latent from a context window of past latents + action. The predictor is gradient-isolated from the autoencoder via detach, so each component trains on its own objective with a single optimizer.

## Evaluation

`evaluate.py` loads a checkpoint and runs three analyses:

- **Open-loop rollout** — feeds model's own predictions back recursively, plots trajectory vs ground truth
- **Energy conservation** — tracks total energy over time, reports absolute and relative drift
- **dt generalization** — tests prediction accuracy across different integration timesteps

```bash
python evaluate.py checkpoint=outputs/<date>/<time>/<model>/best_model.pt eval.horizon=100
```

Outputs `rollout.png`, `energy.png`, `dt_generalization.png`, and `eval_metrics.pt`.

### Multi-Model Comparison

`report.py` compares multiple checkpoints (trained on the same env) side-by-side:

```bash
# Scan a training run directory (short path auto-resolves under outputs/ or multirun/)
python report.py report_checkpoint_dir=<date>/<time>

# Or use full paths
python report.py report_checkpoint_dir=outputs/<date>/<time>
```

Produces comparison tables (console + CSV) and combined plots in `reports/<timestamp>/`:
- `summary.csv` — train loss, open-loop MSE, energy drift per model
- `dt_generalization.csv` — MSE at each dt per model
- `rollout_comparison.png` — all models' trajectories overlaid with ground truth
- `energy_comparison.png` — energy traces overlaid
- `dt_generalization.png` — grouped bar chart across dt values
- `mse_over_horizon.png` — per-timestep MSE curves

## Visual Observations

Environments can render states as images for pixel-based learning. Currently supported: oscillator and pendulum (custom renderer), plus a dm_control pendulum wrapper for MuJoCo rendering.

```bash
# Preview rendering
python scripts/visualize_env.py --env oscillator --n_frames 50
python scripts/visualize_env.py --env pendulum --save_gif pendulum_demo.gif --img_size 128
python scripts/visualize_env.py --dataset datasets/oscillator_visual/<timestamp>


# Train visual world model
python train.py env=oscillator_visual model=visual_world_model
python train.py env=pendulum_visual model=visual_world_model

# Override visual model hyperparams
python train.py env=oscillator_visual model=visual_world_model model.latent_dim=64 model.context_length=2 model.n_codebook=256
```

Visual env configs (`oscillator_visual`, `pendulum_visual`) inherit physics parameters from their base configs and add `observation_mode: pixels`. The `visual` section in `config.yaml` controls `img_size`, `color`, and `render_quality`.

The visual world model config (`model=visual_world_model`) exposes: `latent_dim` (codebook vector size), `n_codebook` (number of codebook entries), `context_length` (how many past latents the predictor sees), `commitment_beta` (VQ commitment loss weight), and `predictor_weight` (latent prediction loss weight).

For the dm_control wrapper: `pip install gymnasium shimmy[dm_control] dm_control`, then use `env=pendulum_dmcontrol`.

## Project Structure

```
configs/
  config.yaml              # Hydra defaults + training params
  env/                     # Per-environment configs
  model/                   # Per-model configs
src/
  models/                  # Model implementations + registry
  envs/                    # Environment implementations + registry
  data/                    # Dataset generation + loading
  eval/                    # Metrics and rollout evaluation
scripts/
  visualize_env.py         # Validate environment rendering (argparse)
generate_dataset.py        # Pre-generate and save train/val/test splits
train.py                   # Unified training entry point
evaluate.py                # Single-model evaluation with plots + metrics
report.py                  # Multi-model comparison report
datasets/                  # Saved datasets (gitignored)
environments/              # HGN pixel-rendering environments (separate system)
experiments/               # Original Jupyter notebooks (archived)
```

## References

- Hamiltonian Generative Networks: [arxiv.org/abs/1909.13789](https://arxiv.org/abs/1909.13789)
- For gymnasium with dm_control envs: [setup guide](https://medium.com/@kaige.yang0110/run-dm-control-with-gymnasium-framestack-and-resize-pixel-obsservation-34c1b8ff4764)
