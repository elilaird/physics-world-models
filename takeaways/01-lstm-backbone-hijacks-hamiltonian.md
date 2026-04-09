# Takeaway 1: Unconstrained Temporal Backbones Bypass Hamiltonian Structure

**Experiment:** HamiltonianPredictor (separable H = V(q) + T(p), Euler integration) with LSTM temporal backbone (2-layer, 128 hidden), JEPA training on oscillator_visual, 60k sequences, dt=0.2, 50 epochs.

## Observation

When a port-Hamiltonian predictor is augmented with an LSTM backbone that conditions the action force G(a), the backbone absorbs the full dynamics and the Hamiltonian energy networks collapse to near-constant functions. The model achieves low teacher-forced prediction error but fails at autoregressive rollout and dt generalization — the defining properties that the Hamiltonian structure was designed to provide.

## Evidence

### Energy landscape collapse

The energy networks V(q) and T(p) failed to learn meaningful structure:

| Metric | Observed | Expected (healthy) |
|--------|----------|--------------------|
| energy_time_var | 5e-6 | O(0.01–0.1) |
| energy_std | 0.0024 | O(0.1–1.0) |
| energy_monotone | 0.567 | >0.8 (damped system) |

An energy_monotone of 0.567 is barely above the 0.5 baseline expected from random ordering, indicating the Hamiltonian imposes no meaningful energy constraint on the learned dynamics.

### Dynamics bypass

With near-zero energy gradients (dV/dq ≈ 0, dT/dp ≈ 0), the Euler update reduces to:

    q_{t+1} = q_t + dt * dT/dp ≈ q_t
    p_{t+1} = p_t + dt * (-dV/dq - gamma * dT/dp + G(a)) ≈ p_t + dt * G(a)

The position coordinate receives no update from the Hamiltonian flow. All state evolution is driven by the LSTM-conditioned action force G(a), which acts as an unconstrained dynamics model.

### Teacher-forcing gap

| Setting | Latent MSE |
|---------|------------|
| Teacher-forced (train) | 7.1e-5 |
| Autoregressive rollout (val) | 1.6e-3 |
| Ratio | 22x |

The 22x gap between teacher-forced and autoregressive evaluation indicates the predictor cannot recover from its own errors. A functioning Hamiltonian would constrain predictions to a physically plausible manifold, reducing error accumulation; the collapsed Hamiltonian provides no such regularization.

### dt generalization failure

| dt | Latent MSE | PSNR |
|----|------------|------|
| 0.05 | 0.006 | 19.8 |
| 0.1 | 0.003 | 20.5 |
| 0.2 (train) | 0.005 | 19.3 |
| 0.5 | 13.34 | 9.4 |

At dt=0.5 (2.5x the training timestep), the predicted latents diverge so far from the training distribution that the decoder produces high-frequency noise artifacts (PSNR 9.4, LPIPS 0.72). This is because G(a) was calibrated for dt=0.2; the linear scaling dt * G(a) produces catastrophically large updates at larger timesteps.

Notably, dt=0.1 outperforms the training dt=0.2. This is not genuine dt generalization but reflects the reduced difficulty of predicting smaller inter-frame state changes.

## Analysis

The LSTM backbone (2-layer, hidden_dim=128) is a powerful sequence model that can learn arbitrary mappings from (state, action) histories to per-frame forces. From the optimizer's perspective, routing dynamics through the backbone is strictly easier than through the Hamiltonian path, which requires:

1. V_net and T_net to develop non-trivial energy landscapes
2. Meaningful autograd gradients (dV/dq, dT/dp) to emerge via second-order optimization
3. The symplectic structure (dq/dt = dT/dp, dp/dt = -dV/dq) to correctly represent the physics

Since the JEPA objective only measures prediction accuracy — not *how* predictions are made — the optimizer follows the path of least resistance. The backbone is a universal approximator that satisfies the prediction loss without requiring the energy networks to participate.

This failure mode is particularly insidious because the model appears to train successfully: teacher-forced prediction loss decreases steadily, reconstructions are sharp, and single-step accuracy is high. The failure only manifests during autoregressive rollout and dt generalization — precisely the settings where the Hamiltonian inductive bias is supposed to provide its advantage.

## Design Principle

**The action conditioning pathway in a physics-informed predictor must be capacity-limited relative to the physics pathway.** If an unconstrained function approximator (LSTM, Transformer, or large MLP) can access the full state history and produce arbitrary forces, it will subsume the physics structure entirely.

For port-Hamiltonian predictors, the action force G(a) should be a weak correction to the conservative dynamics, not a substitute. This can be achieved by:

1. **Removing temporal backbones entirely** — using only per-frame action embeddings
2. **Bottlenecking G_net capacity** — small hidden dimensions or norm constraints
3. **Scaling G_net output** — learned scaling initialized near zero, so the Hamiltonian must establish dynamics first

The integration itself provides temporal coupling: each step's output becomes the next step's input. External temporal context (via LSTM/Transformer) is redundant and counterproductive.

## Implications for Training

This finding suggests a broader principle for physics-informed latent dynamics models: **the physics structure must be the path of least resistance for satisfying the training objective.** If a non-physics pathway exists with greater capacity, the optimizer will prefer it regardless of the physics pathway's theoretical advantages (energy conservation, symplecticity, dt generalization).

This motivates architectural constraints rather than regularization losses. While one could add explicit energy regularization terms, architectural bottlenecking of the non-physics pathway is more robust — it eliminates the failure mode rather than penalizing it.
