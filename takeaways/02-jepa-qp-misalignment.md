# Takeaway 2: Pure JEPA Cannot Enforce the Hamiltonian Position-Momentum Split

**Experiment:** HamiltonianPredictor (separable H = V(q) + T(p), Euler, per-frame action embedding, no LSTM backbone), JEPA training on oscillator_visual, 60k sequences, dt=0.2, 30 epochs.

## Observation

After removing the LSTM temporal backbone (Takeaway 1), the port-Hamiltonian predictor achieves flat dt generalization in latent space — a 100,000x improvement at dt=0.5 — but decoded images show a near-static ball. The position coordinate q does not evolve during integration because the kinetic energy network T(p) remains flat, yielding dq/dt = dT/dp ≈ 0. The encoder, lacking any signal that q should encode spatial position, distributes dynamically-varying information into the momentum half p, which the decoder never sees.

## Evidence

### dt generalization: latent space vs. image space

The latent-space prediction is accurate and dt-invariant:

| dt | Latent MSE | PSNR |
|----|------------|------|
| 0.05 | 0.000663 | 20.0 |
| 0.1 | 0.000312 | 22.0 |
| 0.2 (train) | 0.000214 | 20.2 |
| 0.5 | 0.000128 | 20.2 |

Compare to the LSTM-backbone run (Takeaway 1), where dt=0.5 had Latent MSE = 13.34 and PSNR = 9.4 with catastrophic decoder artifacts. The integration now scales correctly with dt.

However, decoded rollout images show the ball at a near-constant position across all timesteps and all dt values. The predictor produces accurate full-state predictions (low latent MSE) but the position half q — the only half the decoder receives — is near-constant.

### Energy landscape remains flat

| Metric | Value | Expected (healthy) |
|--------|-------|--------------------|
| energy_time_var | 2.5e-5 | O(0.01-0.1) |
| energy_monotone | 0.533 | >0.8 |
| energy_mean | 14.09 | varies meaningfully |

V_net and T_net converged to a high constant (~14) but with negligible variation across states. The Hamiltonian imposes no dynamical structure.

### Predictor dynamics reduce to identity + action shift

With ∂T/∂p ≈ 0 and ∂V/∂q ≈ 0, the Euler integration simplifies to:

    q_{t+1} = q_t + dt · ∂T/∂p  ≈  q_t           (identity copy)
    p_{t+1} = p_t + dt · G(a)                       (action shift only)

The model correctly predicts p evolution (where G(a) acts) and trivially copies q. The training loss, computed over all 64 latent dimensions equally, is satisfied because p predictions are accurate and q predictions are trivially correct (near-identity matches near-constant ground truth).

### Reconstruction is unaffected

The decoder, trained as a detached probe on encoded latents, produces sharp reconstructions (recon_loss = 0.0038). This confirms the encoder does place frame-specific information in q — but the variation between consecutive frames is small enough that the predictor can ignore it.

## Analysis

### The q/p assignment problem

The Hamiltonian predictor imposes a structural assumption: the first half of the latent state is position q (decoded, evolves via dq/dt = ∂T/∂p) and the second half is momentum p (not decoded, evolves via dp/dt = -∂V/∂q + G(a)). This creates an asymmetry: q must encode visually meaningful information that changes between frames, while p carries the dynamical complement.

In pure JEPA training, neither the encoder nor the predictor receives a signal that enforces this assignment:

1. **SIGReg** regularizes all latent dimensions equally toward an isotropic Gaussian. It has no concept of a position-momentum split.

2. **The latent prediction loss** evaluates accuracy over the full state vector. A predictor that correctly predicts p (32 dynamic dims) and copies q (32 near-static dims) achieves low loss without the Hamiltonian contributing.

3. **The decoder** is a detached probe — its reconstruction loss does not flow back to the encoder. The encoder never learns that q specifically must carry decodable position information.

Without external pressure to align the encoder's latent decomposition with the Hamiltonian's structural assumptions, the encoder distributes information in whatever arrangement minimizes the joint prediction + SIGReg objective. This arrangement is not aligned with the physics.

### Why the Hamiltonian predictor doesn't need history

A common intuition is that the predictor needs multi-frame context to infer dynamics. The Hamiltonian predictor processes each frame independently: it reshapes (B, T, D) to (B*T, D) and integrates each state in parallel. This is correct by design — in Hamiltonian mechanics, the phase-space state (q, p) is Markovian. Knowing position and momentum at a single instant uniquely determines the future trajectory. No history is needed.

The encoder already has access to velocity information through channel-concatenated frame pairs (encoder_frames = 2). It can, in principle, compute both position and momentum from consecutive images and place them in the correct halves. The failure is not one of insufficient information but of insufficient incentive: the training objective does not reward correct placement.

### The hybrid reconstruction signal

Setting hybrid_recon_weight > 0 closes the feedback loop that pure JEPA leaves open:

1. Reconstruction loss flows through the encoder, creating the gradient signal: "q must decode to the correct image."
2. The encoder learns to place position information in q, causing q to vary between frames as the ball moves.
3. The latent prediction loss now requires the predictor to predict non-trivial q changes.
4. The only mechanism available to the Hamiltonian predictor for evolving q is dq/dt = ∂T/∂p.
5. Therefore T_net must develop a non-flat energy landscape with meaningful gradients.

This chain cannot activate in pure JEPA mode because step 1 is absent. The hybrid weight need not be large — a small value (0.1) suffices to establish the alignment signal without dominating the JEPA objective.

## Design Principle

**Physics-informed latent dynamics models require the encoder's latent decomposition to match the predictor's structural assumptions.** When the predictor imposes asymmetric roles on latent subspaces (e.g., position vs. momentum, decoded vs. not decoded), the training objective must include a signal that aligns the encoder's representation with these roles. Self-supervised objectives like JEPA that treat all latent dimensions symmetrically cannot enforce this alignment.

For Hamiltonian predictors specifically, a lightweight reconstruction signal on the position half is the minimal intervention: it establishes which latent subspace corresponds to observable configuration and lets the physics structure handle the rest.

## Relationship to Takeaway 1

Takeaway 1 showed that an unconstrained temporal backbone bypasses the Hamiltonian entirely. Takeaway 2 shows that even after removing the backbone, a second failure mode exists: the encoder's latent decomposition can be misaligned with the Hamiltonian's structural assumptions. Both failure modes result in a flat energy landscape, but for different reasons:

- **Takeaway 1**: The backbone provides an easier optimization path that subsumes the Hamiltonian.
- **Takeaway 2**: The encoder distributes information in a way that makes the Hamiltonian's structural split vacuous.

Fixing both requires (1) capacity-limiting the non-physics pathway and (2) providing an alignment signal between the encoder's latent structure and the predictor's physics structure.
