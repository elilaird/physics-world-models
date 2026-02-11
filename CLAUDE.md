# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing neural network architectures for learning physics dynamics as world models. Trains models on simulated physics environments (oscillators, pendulums) and evaluates how well different inductive biases (discrete jumps, Newtonian mechanics, Hamiltonian structure) capture the true dynamics.

## Key Dependencies

- PyTorch, torchdiffeq, hydra-core, omegaconf
- OpenCV (cv2), matplotlib (visualization/rendering)
- Conda environment: `dino_wm`
- Install extras: `pip install hydra-core omegaconf`

## Running Experiments

```bash
# Train (defaults: oscillator env + jump model)
python train.py

# Train specific env + model combination
python train.py env=pendulum model=port_hamiltonian

# Override training params
python train.py env=spaceship model=newtonian training.epochs=80 training.lr=5e-4

# Sweep models
python train.py --multirun model=jump,lstm,newtonian,port_hamiltonian

# Generate a dataset (saved to datasets/<env_name>/<timestamp>/)
python generate_dataset.py
python generate_dataset.py env=pendulum data.n_seqs=1000

# Train on a pre-generated dataset
python train.py dataset_name=oscillator/<timestamp>
python train.py --multirun model=jump,lstm,newtonian dataset_name=oscillator/<timestamp>

# Train visual world model on pixel observations
python train.py env=oscillator_visual model=visual_world_model
python train.py env=pendulum_visual model=visual_world_model model.latent_dim=64

# Evaluate a checkpoint
python evaluate.py checkpoint=outputs/<date>/<time>/<model>/best_model.pt

# Evaluate with custom settings
python evaluate.py checkpoint=path/to/best_model.pt eval.horizon=100 eval.dt_values=[0.05,0.1,0.2,0.5]

# Compare models from a training run (short path auto-resolves under outputs/ or multirun/)
python report.py report_checkpoint_dir=<date>/<time>

# Or use full paths
python report.py report_checkpoint_dir=outputs/<date>/<time>

# Override eval settings for report
python report.py report_checkpoint_dir=<date>/<time> eval.horizon=100 eval.dt_values=[0.05,0.1,0.2,0.5]
```

Hydra outputs (checkpoints, logs, plots) go to `outputs/<date>/<time>/<model_name>/`.

## Architecture

### Config system (`configs/`)
Hydra with composable groups. `configs/config.yaml` sets defaults and training params. Override with `env=<name>` and `model=<name>`.
- **env configs**: oscillator, pendulum, spaceship, three_body — each defines state_dim, action_dim, physics params, variable_params ranges, init_state_range
- **model configs**: jump, lstm, first_order_ode, newtonian, velocity, port_hamiltonian, visual_world_model — each defines type (discrete/ode/visual), hidden_dim, integration_method

### Models (`src/models/`)
All models use `nn.Embedding` for discrete action spaces. Registry in `src/models/__init__.py`.
- **Discrete** (`discrete.py`): `JumpModel` (residual MLP), `LSTMModel` — forward: `(state, action) → next_state`
- **ODE** (`ode.py`): `FirstOrderODENet`, `NewtonianDynamicsModel`, `VelocityDynamicsModel` — forward: `(t, state, action_emb)` for `torchdiffeq.odeint`
- **Hamiltonian** (`hamiltonian.py`): `PortHamiltonianModel` — learns H(q,p), derives dynamics via `torch.autograd.grad`
- **Wrappers** (`wrappers.py`): `TrajectoryMatchingModel` (single-step ODE integration), `TrajectoryShootingModel` (multi-step rollout). ODE models are auto-wrapped by `train.py` when `model.type == "ode"`.
- **Visual** (`visual.py`): `VisualWorldModel` — beta-VAE (VisionEncoder → reparameterize → VisionDecoder) + swappable `LatentPredictor` (residual MLP, via `PREDICTOR_REGISTRY`). Encoder outputs `(mu, logvar)`, KL divergence uses per-dimension free bits clamping to prevent posterior collapse. Predictor operates on detached sampled latents with context window. Separate optimizers for encoder, decoder, and predictor. `train.py` routes to `visual_train_step`/`visual_eval_step` when `model.type == "visual"`.

### Environments (`src/envs/`)
`PhysicsControlEnv` base class with discrete action maps. Registry in `src/envs/__init__.py`.
- `ForcedOscillator`: 2D state [x, v], 3 actions
- `ForcedPendulum`: 2D state [theta, omega], 3 actions
- `ForcedTwoBodySpaceship`: 4D state [qx, qy, vx, vy], 9 directional thrusters
- `ThreeBodyEnv`: 12D state, 9 actions, symplectic Euler integration

### Data (`src/data/`)
`SequenceDataset` generates (states, actions, targets) sequences from any env with randomized variable params per sequence. `generate_dataset.py` pre-computes and saves train/val/test splits as stacked tensors under `data_root/<env>/` (defaults to `datasets/<env>/`). `PrecomputedDataset` loads these for training via `dataset_name` config.

### Evaluation (`src/eval/`)
- `utils.py`: `load_checkpoint()`, `rebuild_model()`, `rebuild_env()` — shared by `evaluate.py` and `report.py`
- `metrics.py`: `mse_over_horizon()`, `energy_drift()`
- `rollout.py`: `open_loop_rollout()`, `dt_generalization_test()`

### Visual Observations (`src/envs/rendering.py`, `src/data/visual_dataset.py`)
Pixel-based observations for environments with `render_state()`. Oscillator and Pendulum have custom renderers ported from `environments/`. `dm_control` pendulum wrapper provides MuJoCo-rendered alternative (requires `gymnasium shimmy[dm_control] dm_control`).

- **Visual env configs** (`oscillator_visual`, `pendulum_visual`): inherit physics from base, set `observation_mode: pixels`
- **`VisualSequenceDataset`**: generates `(images, actions, target_images)` tuples in `(T, C, H, W)` format, includes vector states for validation
- **`visual` config section**: `img_size`, `channels`, `color`, `render_quality`
- **`VisualWorldModel`** (`model=visual_world_model`): beta-VAE + swappable latent predictor. Config exposes `latent_dim`, `beta` (KL weight), `free_bits` (per-dimension KL floor), `context_length`, `predictor_weight`, `predictor` (registry key, default `latent_mlp`). Training uses `visual_train_step` which reconstructs all frames via ELBO (recon + beta * KL) and predicts next-latent from a context window of past sampled latents. Gradient isolation via `.detach()` separates autoencoder and predictor objectives. Eval uses posterior mean (no sampling) for deterministic evaluation.
- **Pretrained checkpoint loading**: `pretrained_checkpoint` config loads weights from a prior checkpoint via `strict=False` (allows architecture changes like swapping predictors). `training.train_encoder`, `training.train_decoder`, `training.train_predictor` (all default `true`) control which components are trainable — set to `false` to freeze. Only trainable components get optimizers.
- Visual rollout evaluation is not yet implemented — `evaluate.py` raises `NotImplementedError` for visual checkpoints

```bash
# Visualize rendering
python scripts/visualize_env.py --env oscillator --n_frames 50
python scripts/visualize_env.py --env pendulum --save_gif pendulum_demo.gif

# Train visual world model
python train.py env=oscillator_visual model=visual_world_model
python train.py env=pendulum_visual model=visual_world_model

# Override visual model hyperparams
python train.py env=oscillator_visual model=visual_world_model model.latent_dim=64 model.context_length=2
python train.py env=oscillator_visual model=visual_world_model model.beta=1.0 model.free_bits=0.25

# Two-phase training: train VAE first, then freeze and train predictor only
# Phase 1: train encoder/decoder only (disable predictor training and loss)
python train.py env=oscillator_visual model=visual_world_model \
    training.train_predictor=false model.predictor_weight=0.0

# Phase 2: load pretrained VAE, freeze encoder/decoder, train predictor
python train.py env=oscillator_visual model=visual_world_model \
    pretrained_checkpoint=/path/to/best_model.pt \
    training.train_encoder=false training.train_decoder=false

# Or fine-tune all components together from a pretrained checkpoint
python train.py env=oscillator_visual model=visual_world_model \
    pretrained_checkpoint=/path/to/best_model.pt
```

### Legacy systems (kept for reference)
- `models.py`, `envs.py`, `datasets.py` — original flat-file versions, superseded by `src/`
- `environments/` — separate HGN pixel-rendering system (Pendulum, Spring, NObjectGravity, ChaoticPendulum) based on arxiv 1909.13789
- `experiments/` — original Jupyter notebooks (archived)
