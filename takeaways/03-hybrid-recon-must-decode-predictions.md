# Takeaway 3: Hybrid Reconstruction Must Decode Predicted States, Not Encoded States

**Experiment:** HamiltonianPredictor (separable H = V(q) + T(p), Euler, per-frame action embedding), JEPA training with hybrid reconstruction on oscillator_visual, 60k sequences, dt=0.2, 30 epochs.

## Observation

Adding a lightweight reconstruction loss on *encoded* states (the original hybrid design) briefly activates the Hamiltonian energy landscape but collapses within a few epochs. The reconstruction signal reaches the encoder but not the predictor — so the encoder learns to place position information in q, but the predictor has no incentive to preserve it across integration steps. The fix is to decode *predicted* states, creating a gradient path through the predictor that forces its q output to be visually correct.

## Evidence

### Encoded-state reconstruction: transient activation followed by collapse

Run 405175 (`hybrid_recon_weight=0.1`, decoding encoded states):

| Epoch | energy_time_var | val/recon_loss | val/rollout_lpips |
|-------|-----------------|----------------|-------------------|
| 9     | 3.3e-4          | 0.0021         | 0.099             |
| 16    | 2.3e-7          | 0.0020         | 0.097             |

Between epochs 9 and 16, `energy_time_var` collapsed by a factor of 1,400 — from a promising value back to the flat-energy regime observed in pure JEPA (Takeaway 2). Meanwhile the encoder's reconstruction quality (`recon_loss`) remained stable, confirming the encoder continued to place position information in q. The Hamiltonian predictor simply learned to copy q unchanged, never developing the ∂T/∂p gradients needed for position evolution.

### Why the gradient path is broken

In the encoded-state design, the reconstruction loss is computed as:

    L_recon = || decode(encode(x_t)) - x_t ||²

The gradient flows through the decoder and encoder but *not* through the predictor. The encoder receives the signal "q must decode to the correct image," which it satisfies. But the predictor — the component that must evolve q via dq/dt = ∂T/∂p — receives no reconstruction gradient. Its only training signal remains the latent prediction loss, which is satisfied by the identity mapping on q (Takeaway 2).

### The predicted-state design

Replacing the reconstruction target with predicted states creates the necessary gradient path:

    L_pred_recon = || decode(predict(z_{t-1}, a_{t-1})) - x_t ||²

Now the gradient flows: L_pred_recon → decoder → predicted q → predictor → T_net. The predictor must produce q values that decode to the correct next frame. The only mechanism available to it for evolving q is dq/dt = ∂T/∂p, so T_net must develop a non-flat energy landscape with meaningful gradients.

## Analysis

### The encoder-predictor gradient disconnect

The original hybrid design was motivated by the correct diagnosis: the encoder needs a signal to place position information in the q half of the latent space (Takeaway 2). A reconstruction loss on encoded states achieves this — the encoder does learn the correct q/p assignment. However, the diagnosis was incomplete: correct assignment is necessary but not sufficient. The predictor must also *use* the assignment, evolving q through the Hamiltonian integration rather than copying it.

This reveals a general principle about physics-informed latent dynamics: the structural inductive bias (Hamiltonian integration) is only active when the training objective creates gradient flow *through* the physics pathway. A loss that supervises the encoder's output but bypasses the predictor leaves the physics structure inert.

### Two-stage failure mode

The transient activation at epoch 9 suggests a plausible training trajectory:

1. **Early training (epochs 1-9):** The reconstruction signal pushes position information into q. The predictor has not yet converged, and the latent prediction loss gradient passes through T_net. Energy variation increases as T_net begins to learn.

2. **Late training (epochs 9-16):** The predictor converges to the identity-on-q solution, which is a lower-loss basin than the Hamiltonian integration path. Once the predictor learns to copy q, the latent prediction loss gradient through T_net vanishes (the Jacobian ∂pred_q/∂T_net → 0). Energy variation collapses.

This explains why the collapse is delayed rather than immediate: the Hamiltonian pathway is initially competitive, but the identity shortcut eventually wins because it has strictly lower capacity requirements.

### Why a higher weight (0.3) doesn't help

Run 405200 (`hybrid_recon_weight=0.3`, decoding encoded states, epoch 5):

| Metric | 0.3 weight (epoch 5) | 0.1 weight (epoch 16) |
|--------|----------------------|------------------------|
| energy_time_var | 6.5e-4 | 2.3e-7 |
| val/rollout_lpips | 0.452 | 0.097 |
| val/recon_loss | 0.150 | 0.002 |

At epoch 5 the 0.3 run had higher energy variation but far worse reconstruction and rollout quality — it had not converged. The fundamental issue is not the weight magnitude but the gradient topology: no amount of encoded-state reconstruction loss can create gradients through the predictor.

## Design Principle

**Reconstruction losses in physics-informed latent dynamics must decode the predictor's output, not the encoder's output.** When the inductive bias (Hamiltonian, Lagrangian, or any structured integration) governs how latent states evolve, the reconstruction signal must flow through the integration pathway. Otherwise the predictor can satisfy the latent prediction loss with a trivial mapping (identity, linear shift) that bypasses the physics structure entirely.

For Hamiltonian predictors specifically: `L = || decode(predict(z, a)) - x_{t+1} ||²` creates the gradient chain decode → predict → ∂T/∂p, forcing the kinetic energy network to develop meaningful gradients for position evolution.

## Relationship to Previous Takeaways

The three takeaways form a progression of failure modes for the port-Hamiltonian predictor under JEPA training, each requiring a more targeted fix:

| # | Failure mode | Root cause | Fix |
|---|-------------|------------|-----|
| 1 | LSTM backbone hijacks Hamiltonian | Backbone has higher capacity than physics pathway | Remove backbone; use per-frame action embedding |
| 2 | Encoder q/p misalignment | JEPA treats all latent dims symmetrically | Add reconstruction signal on position half |
| 3 | Encoded-state recon doesn't reach predictor | Gradient path bypasses physics integration | Decode predicted states, not encoded states |

Each fix addresses a different point in the encoder → predictor → decoder chain where the physics structure can be short-circuited. The complete solution requires all three: (1) capacity-limited action conditioning, (2) reconstruction-based q/p alignment, and (3) gradient flow through the integration pathway.
