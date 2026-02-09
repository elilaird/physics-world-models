# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing neural network architectures for learning physics dynamics as world models. Trains models on simulated physics environments (oscillators, pendulums) and evaluates how well different inductive biases (discrete jumps, Newtonian mechanics, Hamiltonian structure) capture the true dynamics.

## Key Dependencies

- PyTorch, torchdiffeq (ODE integration via `odeint` with dopri5 solver)
- OpenCV (cv2), matplotlib (visualization/rendering)
- No package manager config exists — install dependencies manually (`pip install torch torchdiffeq opencv-python matplotlib`)

## Architecture

There are two parallel environment systems:

### 1. Control environments (`envs.py`)
`PhysicsControlEnv` base class with `ForcedOscillator` and `ForcedPendulum`. These use discrete action spaces (action maps like `{0: -1.0, 1: 0.0, 2: 1.0}`), integrate dynamics with `torchdiffeq.odeint`, and support variable parameters (mass, damping, stiffness) randomized per sequence for generalization. Used with `datasets.py:SequenceDataset` which generates (states, actions, targets) tuples.

### 2. Hamiltonian Generative Network environments (`environments/`)
Based on [arxiv.org/abs/1909.13789](https://arxiv.org/abs/1909.13789). `Environment` ABC in `environment.py` with concrete implementations: `Pendulum`, `Spring`, `NObjectGravity`, `ChaoticPendulum`. These use generalized coordinates (q, p), render pixel observations via anti-aliased circle rendering, and are accessed through `EnvFactory`. Used with `environments/datasets.py:EnvironmentSampler`.

### Models (`models.py`)
All models predict next state from (state, action) with discrete action embeddings:
- **JumpModel**: Residual MLP, learns `x_{t+1} = x_t + f(x_t, a_t)`
- **LSTMModel**: LSTM with residual connection for sequential prediction
- **NewtonianDynamicsModel**: Physics prior separating position/velocity, learns acceleration with learned damping. Takes `(t, state, action_emb)` for ODE integration.
- **VelocityDynamicsModel**: Learns velocity directly, zeros for dvdt to share integrator interface
- **PortHamiltonianModel**: Learns Hamiltonian H(q,p) with autograd for symplectic structure, includes dissipation and input port G(u)

The ODE-based models (Newtonian, Velocity, PortHamiltonian) use `forward(t, state, action_emb)` signature compatible with `torchdiffeq.odeint`. The discrete models (Jump, LSTM) use `forward(state, action)`.

## Experiments

Jupyter notebooks in `experiments/` — run directly (not via a test runner). Key notebook: `experiments/pendulum/pendulum.ipynb`.

## Running Code

```bash
# Environment factory demo (from environments/ directory)
python environments/environment_factory.py

# Individual environment demos
python environments/pendulum.py
```
