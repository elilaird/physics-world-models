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

# Evaluate a checkpoint
python evaluate.py checkpoint=outputs/<date>/<time>/best_model.pt

# Evaluate with custom settings
python evaluate.py checkpoint=path/to/best_model.pt eval.horizon=100 eval.dt_values=[0.05,0.1,0.2,0.5]

# Compare multiple models (must share same env)
python report.py checkpoints=[path/to/jump/best_model.pt,path/to/lstm/best_model.pt]

# Or scan a multirun output directory
python report.py checkpoint_dir=multirun/<date>/<time>

# Override eval settings for report
python report.py checkpoint_dir=multirun/<date>/<time> eval.horizon=100 eval.dt_values=[0.05,0.1,0.2,0.5]
```

Hydra outputs (checkpoints, logs, plots) go to `outputs/<date>/<time>/`.

## Architecture

### Config system (`configs/`)
Hydra with composable groups. `configs/config.yaml` sets defaults and training params. Override with `env=<name>` and `model=<name>`.
- **env configs**: oscillator, pendulum, spaceship, three_body — each defines state_dim, action_dim, physics params, variable_params ranges, init_state_range
- **model configs**: jump, lstm, first_order_ode, newtonian, velocity, port_hamiltonian — each defines type (discrete/ode), hidden_dim, integration_method

### Models (`src/models/`)
All models use `nn.Embedding` for discrete action spaces. Registry in `src/models/__init__.py`.
- **Discrete** (`discrete.py`): `JumpModel` (residual MLP), `LSTMModel` — forward: `(state, action) → next_state`
- **ODE** (`ode.py`): `FirstOrderODENet`, `NewtonianDynamicsModel`, `VelocityDynamicsModel` — forward: `(t, state, action_emb)` for `torchdiffeq.odeint`
- **Hamiltonian** (`hamiltonian.py`): `PortHamiltonianModel` — learns H(q,p), derives dynamics via `torch.autograd.grad`
- **Wrappers** (`wrappers.py`): `TrajectoryMatchingModel` (single-step ODE integration), `TrajectoryShootingModel` (multi-step rollout). ODE models are auto-wrapped by `train.py` when `model.type == "ode"`.

### Environments (`src/envs/`)
`PhysicsControlEnv` base class with discrete action maps. Registry in `src/envs/__init__.py`.
- `ForcedOscillator`: 2D state [x, v], 3 actions
- `ForcedPendulum`: 2D state [theta, omega], 3 actions
- `ForcedTwoBodySpaceship`: 4D state [qx, qy, vx, vy], 9 directional thrusters
- `ThreeBodyEnv`: 12D state, 9 actions, symplectic Euler integration

### Data (`src/data/`)
`SequenceDataset` generates (states, actions, targets) sequences from any env with randomized variable params per sequence.

### Evaluation (`src/eval/`)
- `utils.py`: `load_checkpoint()`, `rebuild_model()`, `rebuild_env()` — shared by `evaluate.py` and `report.py`
- `metrics.py`: `mse_over_horizon()`, `energy_drift()`
- `rollout.py`: `open_loop_rollout()`, `dt_generalization_test()`

### Legacy systems (kept for reference)
- `models.py`, `envs.py`, `datasets.py` — original flat-file versions, superseded by `src/`
- `environments/` — separate HGN pixel-rendering system (Pendulum, Spring, NObjectGravity, ChaoticPendulum) based on arxiv 1909.13789
- `experiments/` — original Jupyter notebooks (archived)
