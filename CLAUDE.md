# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules

- Do NOT run any Python commands. The user will test everything themselves.

## Epistemic Standards

### Never make strong causal claims without proof
- Do NOT say "the problem IS X" or "this IS caused by Y" unless you have directly verified it (e.g., ran the code, read the traceback, checked the value).
- Instead, say "this is LIKELY caused by X because [reasoning]" and then propose a verification step.

### Diagnosis workflow
When debugging or diagnosing issues:
1. **State hypotheses ranked by likelihood** — "Most likely: ..., Less likely: ..., Unlikely but possible: ..."
2. **Propose concrete verification steps** for each hypothesis — a command to run, a value to print, a file to check.
3. **Run the check yourself when possible** before claiming a root cause.
4. **After verification, state what you confirmed** — "Confirmed: the tensor shape at line 42 is [3,64] not [3,128] as expected."

### Language rules
- ✅ "Based on the traceback, this is caused by X" (you have evidence)
- ✅ "This is likely X because [reason]. To confirm, try: `print(tensor.shape)`"
- ❌ "The problem is X" (without having checked)
- ❌ "This happens because Y" (without evidence)

### When you don't know
- Say so. "I'm not sure what's causing this. Here are three things I'd check: ..."
- Never fabricate explanations to sound confident.

## Project Overview

Research project testing whether a **port-Hamiltonian predictor + JEPA training** learns continuous-time dynamics from pixels with dt-generalization. Trains visual world models on simulated physics environments (oscillators, pendulums) and evaluates how well different inductive biases capture the true dynamics.

Core experiment: compare **Hamiltonian** (physics-informed, dt-aware) against **MLP** and **LSTM** baselines (fixed-step, dt-agnostic) under JEPA training.

## Key Dependencies

- PyTorch, hydra-core, omegaconf
- OpenCV (cv2), matplotlib (visualization/rendering)
- lpips (perceptual metrics)
- Conda environment: `world_models`

## Running Experiments

```bash
# Generate a dataset
python generate_dataset.py
python generate_dataset.py dataset=oscillator_visual_60k

# Train with each predictor (JEPA training, always)
python train_visual.py predictor=hamiltonian
python train_visual.py predictor=mlp
python train_visual.py predictor=lstm

# Sweep all predictors
python train_visual.py --multirun predictor=mlp,lstm,hamiltonian

# Override training params
python train_visual.py training.lr=1e-4 training.sigreg_lambda=0.05

# Train on pendulum
python train_visual.py env=pendulum_visual predictor=hamiltonian

# Override model hyperparams
python train_visual.py model.latent_channels=64 model.context_length=3

# Tune SIGReg lambda (the key JEPA hyperparameter)
python train_visual.py --multirun training.sigreg_lambda=0.01,0.05,0.1,0.5,1.0

# Hybrid mode (JEPA + lightweight reconstruction)
python train_visual.py training.hybrid_recon_weight=0.1

# Two-phase training: train VAE first, then freeze and train predictor only
# Phase 1: train encoder/decoder only (disable predictor training)
python train_visual.py training.train_predictor=false

# Phase 2: load pretrained VAE, freeze encoder/decoder, train predictor
python train_visual.py \
    pretrained_checkpoint=/path/to/best_model.pt \
    training.train_encoder=false training.train_decoder=false

# Evaluate a checkpoint (includes dt generalization)
python evaluate.py checkpoint=path/to/best_model.pt
python evaluate.py checkpoint=path/to/best_model.pt eval.n_rollouts=8 eval.dt_values=[0.05,0.1,0.2,0.5]
```

Hydra outputs (checkpoints, logs, plots) go to `outputs/<date>/<time>/<model_name>/`.

## Architecture

### Config system (`configs/`)
Hydra with composable groups. `configs/config.yaml` sets defaults and training params. Override with `env=<name>` and `predictor=<name>`.
- **env configs**: `oscillator_visual`, `pendulum_visual` — each defines state_dim, action_dim, physics params, variable_params ranges, init_state_range, rendering settings
- **model config**: `visual_world_model` — latent_channels, hidden_channels, context_length, pred_length, encoder_frames
- **predictor configs**: `hamiltonian` (default), `mlp`, `lstm`
- **dataset configs**: `oscillator_visual_testing`, `oscillator_visual_60k`, etc.

### Predictors (`src/models/predictors.py`)

Three predictors registered in `PREDICTOR_REGISTRY`:

- **MLPPredictor** (`mlp`): per-frame residual MLP, `z_{t+1} = z_t + f(z_t, a_t)`. Fixed-step, ignores dt.
- **LSTMPredictor** (`lstm`): LSTM over context + residual output. Fixed-step, ignores dt.
- **HamiltonianPredictor** (`hamiltonian`): non-separable `H(z)` — a single scalar energy network over the full latent. Hamilton's equations derived via one autograd call (∂H/∂z) sliced into ∂H/∂q and ∂H/∂p, with port-Hamiltonian dissipation `γ·∂H/∂p` and per-frame action force `G(a)` on momentum. Forward Euler integration. **dt-aware**: accepts dt parameter for temporal generalization. `.energy(z)` method for monitoring.

### Visual World Model (`src/models/visual.py`)

JEPA-only architecture with flat latent space:
- **VisionEncoder**: 8-layer ConvNet → MLP → flat latent `z ∈ (B, D)`. BatchNorm projector for SIGReg compatibility.
- **VisionDecoder**: MLP → spatial → ConvNet upsample. Receives the **full latent** `z ∈ (B, D)` — symmetric in q and p.
- **VisualWorldModel**: encoder + decoder + swappable predictor. No state_transform (encoder output IS the state). The Hamiltonian predictor splits `z = [q, p]` internally for its dynamics, but the decoder is blind to the split — this removes the q/p misalignment problem documented in `takeaways/02`.

**Key config parameters** (`configs/model/visual_world_model.yaml`):
- `latent_channels: 64` — total latent dims (predictor splits into 32 q + 32 p internally; decoder sees all 64)
- `hidden_channels: 512` — hidden dim in encoder/decoder MLPs
- `encoder_frames: 2` — number of frames channel-concatenated for velocity estimation
- `context_length: 3` — number of latent frames the predictor sees
- `pred_length: 5` — prediction window for sliding-window training

### Training (`train_visual.py`)

JEPA-only training (LeWorldModel-style):
- **Loss** = latent prediction (MSE) + SIGReg (Gaussian regularization) + decoder probe (detached recon)
- Encoder learns via co-adaptation with predictor (targets NOT detached by default)
- SIGReg prevents collapse (replaces KL divergence from beta-VAE)
- Single optimizer (Adam)
- Optional hybrid mode: set `training.hybrid_recon_weight > 0` for reconstruction supervision

Key JEPA config parameters:
- `training.sigreg_lambda`: SIGReg weight (0.01–1.0, THE key hyperparameter)
- `training.hybrid_recon_weight`: 0 = pure JEPA, >0 = hybrid
- `training.detach_jepa_targets`: if true, encoder doesn't receive predictor gradients

### SIGReg (`src/models/sigreg.py`)

Sketched-Isotropic-Gaussian Regularizer from the LeWorldModel paper. Projects embeddings onto random directions, applies Epps-Pulley normality test, averages. Zero loss = perfectly isotropic Gaussian embeddings.

**Reference paper**: `references/LeWorldModel- Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.pdf`

### Environments (`src/envs/`)
`PhysicsControlEnv` base class with discrete action maps. Two environments:
- `ForcedOscillator`: 2D state [x, v], 3 actions
- `ForcedPendulum`: 2D state [theta, omega], 3 actions

### Data (`src/data/`)
`PrecomputedDataset` loads pre-generated train/val/test splits from `.npz` files. `generate_dataset.py` generates visual trajectories and saves as stacked tensors under `datasets/<env>/`.

### Evaluation (`src/eval/`)
- `utils.py`: `load_checkpoint()`, `rebuild_model()`, `rebuild_env()`
- `metrics.py`: `compute_visual_metrics()` (MAE, PSNR, SSIM, LPIPS)
- `rollout.py`: `visual_open_loop_rollout()`, `visual_dt_generalization_test()`

Key metrics:
- **dt generalization**: Hamiltonian should adapt to different dt values; MLP/LSTM baselines should degrade
- **energy_monotone**: fraction of timesteps with non-increasing Hamiltonian (should be high for dissipative systems)
- **SIGReg loss**: should converge to near-zero (embeddings are Gaussian)
