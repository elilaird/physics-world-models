# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing neural network architectures for learning physics dynamics as world models. Trains models on simulated physics environments (oscillators, pendulums) and evaluates how well different inductive biases (discrete jumps, Newtonian mechanics, Hamiltonian structure) capture the true dynamics.

## Key Dependencies

- PyTorch, torchdiffeq, hydra-core, omegaconf
- OpenCV (cv2), matplotlib (visualization/rendering)
- Conda environment: `world_models`
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

# Train visual world model (spatial latents + convolutional predictors)
python train_visual.py
python train_visual.py predictor=spatial_hamiltonian
python train_visual.py predictor=spatial_jump training.lr=1.5e-4

# Sweep spatial predictors
python train_visual.py --multirun predictor=spatial_jump,spatial_lstm,spatial_ode,spatial_newtonian,spatial_hamiltonian

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
- **predictor configs** (`configs/predictor/`): `spatial_jump` (default), `spatial_lstm`, `spatial_ode`, `spatial_newtonian`, `spatial_hamiltonian` — spatial convolutional predictors for the visual world model. Legacy flat predictors (`latent_mlp`, `latent_lstm`, etc.) are kept for backward compat with vector-state models.

### Models (`src/models/`)
All models use `nn.Embedding` for discrete action spaces. Registry in `src/models/__init__.py`.
- **Discrete** (`discrete.py`): `JumpModel` (residual MLP), `LSTMModel` — forward: `(state, action) → next_state`
- **ODE** (`ode.py`): `FirstOrderODENet`, `NewtonianDynamicsModel`, `VelocityDynamicsModel` — forward: `(t, state, action_emb)` for `torchdiffeq.odeint`
- **Hamiltonian** (`hamiltonian.py`): `PortHamiltonianModel` — learns H(q,p), derives dynamics via `torch.autograd.grad`
- **Wrappers** (`wrappers.py`): `TrajectoryMatchingModel` (single-step ODE integration), `TrajectoryShootingModel` (multi-step rollout). ODE models are auto-wrapped by `train.py` when `model.type == "ode"`.
- **Visual** (`visual.py`): `VisualWorldModel` — beta-VAE with **spatial latents** + swappable convolutional predictor. See "Visual World Model" section below.
- **Predictors** (`predictors.py`): All latent-space predictors (flat and spatial). See "Predictors" section below.

### Predictors (`src/models/predictors.py`)

Two families of predictors, registered in `PREDICTOR_REGISTRY`:

**Spatial (convolutional) predictors** — operate on `(B, T, C, H, W)` spatial latents:
- `SpatialJumpPredictor` (`spatial_jump`): residual ConvNet, `z + Conv(cat(z, action_broadcast))`
- `SpatialConvLSTMPredictor` (`spatial_lstm`): ConvLSTM cell loops over time, residual output
- `SpatialODEPredictor` (`spatial_ode`): `dz/dt = ConvNet(z, a)` integrated with `torchdiffeq.odeint`
- `SpatialNewtonianPredictor` (`spatial_newtonian`): `dq/dt = p, dp/dt = ConvNet(q,p,a) - γp`
- `SpatialHamiltonianPredictor` (`spatial_hamiltonian`): separable `H(q,p) = T(p) + V(q)` with ConvNet energy networks (Conv→Softplus→Conv→Softplus→AdaptiveAvgPool→Linear→scalar), autograd gives spatial force fields; supports leapfrog integration. `.energy(z)` method for monitoring.

**Flat (vector) predictors** — operate on `(B, T, D)` flat latents (legacy, for vector-state models):
- `LatentPredictor` (`latent_mlp`), `LatentLSTMPredictor` (`latent_lstm`), `LatentODEPredictor` (`latent_ode`), `LatentNewtonianPredictor` (`latent_newtonian`), `LatentHamiltonianPredictor` (`latent_hamiltonian`)

Action handling: `_broadcast_action()` embeds discrete actions via `nn.Embedding` and tiles to `(B*T, emb_dim, H, W)` for spatial concatenation with latents.

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
- `metrics.py`: `mse_over_horizon()`, `energy_drift()`, `compute_visual_metrics()`
- `rollout.py`: `open_loop_rollout()`, `dt_generalization_test()`, `visual_open_loop_rollout()`, `visual_dt_generalization_test()`

### Visual World Model (`src/models/visual.py`, `train_visual.py`)

HGN-inspired architecture using **spatial latents** — the encoder outputs `(B, C, 8, 8)` feature maps instead of flat vectors, preserving spatial structure through the entire pipeline.

**Latent shape reference:**

| Component | Shape |
|-----------|-------|
| Encoder output (mu, logvar) | `(B, 32, 8, 8)` |
| State transform (encoder transformer) | 3-layer ConvNet, `(B, 32, 8, 8)` → `(B, 32, 8, 8)` |
| Position q (for decoding) | `(B, 16, 8, 8)` — first half of channels |
| Momentum p (for dynamics) | `(B, 16, 8, 8)` — second half of channels |
| Decoder input | `(B, 16, 8, 8)` → 1×1 conv expand → ResBlock chain → `(B, C, 64, 64)` |
| Predictor context | `(B, T, 32, 8, 8)` |
| Action broadcast | `nn.Embedding → (B*T, emb, 1, 1) → expand to (B*T, emb, 8, 8)` |

**Key config parameters** (`configs/model/visual_world_model.yaml`):
- `latent_channels: 32` — total latent channels (split into 16 position + 16 momentum)
- `spatial_size: 8` — spatial resolution of latent maps (8×8 for 64×64 images = 8× compression)
- `hidden_channels: 64` — hidden channels in predictor ConvNets
- `beta: 0.001` — KL weight in ELBO
- `free_bits: 0.0` — per-element KL floor (prevents posterior collapse)
- `encoder_frames: 2` — number of frames channel-concatenated for velocity estimation
- `predictor_weight: 1.0` — weight of predictor loss (detached mode only)

**Components:**
- **VisionEncoder**: 8-layer ConvNet (3 downsample + 5 depth), outputs spatial mu/logvar via 1×1 conv
- **VisionDecoder**: 1×1 conv channel expand → ResBlock+Upsample ×3 (8→64). Input is position half of spatial latent.
- **State transform**: 3-layer ConvNet (Conv→Tanh→Conv→Tanh→Conv) mapping variational latent z to phase-space state s=(q,p). Analogous to HGN paper's "encoder transformer".
- **KL divergence**: Per-element free bits clamping, sums over all spatial-channel dims `(C×H×W)`. With 32×8×8=2048 elements, effective KL is larger than flat — controlled by `beta`.

**Training modes** (`train_visual.py`):
- **HGN mode** (default, `training.training_mode=hgn`): Encode first K frames → single z₀ → autoregressive rollout → decode ALL frames → single ELBO. End-to-end gradients. Single optimizer.
- **Detached mode** (`training.training_mode=detached`): Per-frame encoding with detached predictor loss. Separate encoder/decoder/predictor optimizers.

**Reconstruction loss**: L2 (MSE), matching the HGN paper. Changed from L1 in the flat-latent version.

**Pretrained checkpoint loading**: `pretrained_checkpoint` config loads weights via `strict=False` (allows architecture changes like swapping predictors). `training.train_encoder`, `training.train_decoder`, `training.train_predictor` control which components are trainable.

```bash
# Visualize rendering
python scripts/visualize_env.py --env oscillator --n_frames 50
python scripts/visualize_env.py --env pendulum --save_gif pendulum_demo.gif

# Train visual world model with spatial latents (default: spatial_jump predictor)
python train_visual.py
python train_visual.py predictor=spatial_hamiltonian
python train_visual.py predictor=spatial_jump training.lr=1.5e-4

# Override spatial model hyperparams
python train_visual.py model.latent_channels=64 model.context_length=2
python train_visual.py model.beta=1.0 model.free_bits=0.25

# Sweep all spatial predictors
python train_visual.py --multirun predictor=spatial_jump,spatial_lstm,spatial_ode,spatial_newtonian,spatial_hamiltonian

# Two-phase training: train VAE first, then freeze and train predictor only
# Phase 1: train encoder/decoder only (disable predictor training and loss)
python train_visual.py training.train_predictor=false model.predictor_weight=0.0

# Phase 2: load pretrained VAE, freeze encoder/decoder, train predictor
python train_visual.py \
    pretrained_checkpoint=/path/to/best_model.pt \
    training.train_encoder=false training.train_decoder=false

# Or fine-tune all components together from a pretrained checkpoint
python train_visual.py pretrained_checkpoint=/path/to/best_model.pt

# Evaluate a visual checkpoint
python evaluate.py checkpoint=path/to/best_model.pt eval.n_rollouts=8
```

### Legacy systems (kept for reference)
- `models.py`, `envs.py`, `datasets.py` — original flat-file versions, superseded by `src/`
- `environments/` — separate HGN pixel-rendering system (Pendulum, Spring, NObjectGravity, ChaoticPendulum) based on arxiv 1909.13789
- `experiments/` — original Jupyter notebooks (archived)
- Flat latent predictors (`latent_mlp`, `latent_lstm`, etc.) and `train.py` visual steps — superseded by spatial predictors and `train_visual.py`
