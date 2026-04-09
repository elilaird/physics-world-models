# Takeaway 5: Information-Theoretic Limits of dt-Generalization Under Finite Temporal Coverage

**Context:** The Hamiltonian predictor requires velocity information to make the q/p split well-posed (Takeaway 4), but encoding velocity from frame pairs (`encoder_frames=2`) couples the representation to the training timestep. This analysis characterizes the residual dt-dependence in the proposed resolution — single-frame encoding with latent finite-difference velocity inference — and derives the information-theoretic bounds on generalization.

## The Velocity–Generalization Conflict

Two confirmed requirements are in tension:

1. **Velocity is necessary for Hamiltonian dynamics.** A single-frame encoder produces position-only latents, making the q/p split geometrically meaningless. The predictor collapses to mean-prediction within ~10 epochs (Takeaway 4, runs 405817, 405855, 405875, 406063).

2. **dt-generalization requires dt-independent representations.** When `encoder_frames=2`, the encoder processes channel-concatenated frame pairs `[I_t, I_{t+dt}]`. The pixel displacement between frames scales as `v_true * dt`, so the learned latent velocity is proportional to dt_train. At evaluation with dt_eval != dt_train, the encoded velocity is systematically miscalibrated by a factor of dt_eval / dt_train — a bias the predictor cannot correct.

These two requirements are not jointly satisfiable by the encoder alone. Any architecture that embeds velocity into the encoder output at a fixed dt sacrifices generalization; any architecture that omits velocity sacrifices the Hamiltonian's structural inductive bias.

## Proposed Resolution: Separate Position and Velocity Pathways

The resolution is to split the two types of information into architecturally distinct pathways:

- **Position q = encoder output from a single frame.** dt-independent by construction — a single image contains no temporal information.
- **Velocity/momentum p = latent finite difference, smoothed by GRU.** Computed as `v_t = (q_{t+1} - q_t) / dt`, where the division by dt is hardcoded (not learned). The GRU processes a context window of (q_t, v_t, a_t) tuples and outputs a smoothed momentum estimate p and system-identification parameters theta.

The key property: the per-step GRU input `v_t = (q_{t+1} - q_t) / dt` is dt-independent in expectation. At any observation rate, the same physical velocity produces the same normalized input:

    dt = 0.2:  Δq = v_true × 0.2,   v = Δq / 0.2 = v_true
    dt = 0.05: Δq = v_true × 0.05,  v = Δq / 0.05 = v_true

However, the GRU processes a fixed number of frames (infer_context_length = N), so the total temporal coverage T_cover = (N - 1) * dt varies with dt. This is the residual dt-dependence, and it has information-theoretic consequences.

## Three Information Channels with Distinct dt-Sensitivity

The N context frames provide three channels of information to the GRU, each with qualitatively different sensitivity to the observation timestep:

**Channel 1: Position sequence {q_0, ..., q_{N-1}}.** The position at each sample point is a property of the continuous trajectory evaluated at that instant — it does not depend on dt. However, the temporal span of the sampled positions is T_cover = (N-1) * dt, which determines how much of the trajectory's geometry is visible.

**Channel 2: Velocity sequence {v_0, ..., v_{N-2}}.** Each velocity estimate v_t = (q_{t+1} - q_t) / dt converges to the true instantaneous velocity as dt -> 0, but the estimation noise diverges. If the encoder output has additive noise with variance sigma^2, the velocity noise variance is:

    Var(v_t) = 2 * sigma^2 / dt^2

The velocity signal-to-noise ratio scales linearly with dt: SNR_v = |v_true| * dt / (sqrt(2) * sigma). At small dt, individual velocity estimates are noisy but the GRU can smooth across N-2 estimates, improving the effective SNR by a factor of sqrt(N-2).

At large dt, the finite difference approximates the chord velocity (mean over the interval) rather than the instantaneous velocity. For a trajectory with curvature kappa, the bias is O(kappa * dt^2) — negligible for smooth dynamics at moderate dt but significant when the trajectory curves substantially within one step.

**Channel 3: Trajectory curvature (implicit).** The curvature of the sampled trajectory — how the position sequence bends — carries information about system parameters (frequency omega, damping gamma). This information is not a direct input but an emergent property that the GRU must extract from the pattern of positions and velocities. It has the strongest dt-dependence because curvature is only visible when the temporal coverage spans a meaningful fraction of the system's characteristic timescale.

## Momentum Inference (p): Local, Mild dt-Sensitivity

Inferring the momentum at the last context frame is a local operation. The GRU needs to estimate the instantaneous velocity at time t_{N-1}, for which the most recent 2-3 velocity estimates (v_{N-3}, v_{N-2}) provide direct information. The GRU's role is to smooth these noisy finite differences.

The noise-bias tradeoff for momentum inference:

- **Small dt**: higher noise per velocity estimate (Var(v) proportional to 1/dt^2), but more estimates of the same local quantity. The GRU can average, yielding effective SNR proportional to dt * sqrt(N).
- **Large dt**: lower noise per estimate, but the finite-difference bias O(kappa * dt^2) introduces a systematic error. For oscillatory dynamics with frequency omega, the bias becomes non-negligible when omega * dt approaches 1.

In both regimes, the GRU has sufficient local information for robust momentum estimation. Momentum inference should generalize well across a wide range of dt values.

## Parameter Inference (theta): Global, Strong dt-Sensitivity

System parameters — frequency omega, damping gamma, and other trajectory-level invariants — are global properties of the dynamics. They cannot be identified from a local patch of the trajectory; the observer must see the system evolve over a meaningful fraction of its characteristic timescale.

### Cramer-Rao bound for frequency estimation

For N uniformly-spaced position samples of a sinusoidal signal x(t_i) = A * cos(omega * t_i + phi) + noise (variance sigma^2), the Fisher information for omega is:

    I(omega) = (A^2 / sigma^2) * sum_i  t_i^2 * sin^2(omega * t_i + phi)

For uniform sampling over total time T_cover, this simplifies to:

    I(omega) ~ (A^2 * N * T_cover^2) / (3 * sigma^2)

Since T_cover = (N-1) * dt, the Fisher information scales as dt^2 at fixed N. The Cramer-Rao lower bound on the variance of any unbiased frequency estimator is:

    Var(omega_hat) >= 1 / I(omega)  proportional to  1 / dt^2

**Halving the observation timestep does not merely halve the frequency information — it quarters it.** This is the fundamental information-theoretic limit: with a fixed number of context frames, smaller dt means less temporal coverage, which means quadratically less information about periodic structure.

### Concrete scaling for the oscillator

For a damped oscillator with natural frequency omega ~ 1 rad/s (period T_period ~ 6.3 time units), the GRU's context window of N = 8 frames provides:

| dt    | T_cover = 7 * dt | Coverage (T_cover / T_period) | Relative Fisher info | Qualitative regime           |
|-------|------------------|-------------------------------|----------------------|------------------------------|
| 0.05  | 0.35             | 5.6%                          | 1/16                 | Near-linear; omega unidentifiable |
| 0.1   | 0.7              | 11%                           | 1/4                  | Slight curvature visible     |
| 0.2   | 1.4              | 22%                           | 1 (reference)        | Clear arc; omega estimable   |
| 0.4   | 2.8              | 45%                           | 4                    | Half-period; well-constrained |
| 0.5   | 3.5              | 56%                           | 6.25                 | Over half a period           |

At dt = 0.05, the GRU sees 5.6% of one oscillation — a nearly straight line segment. From a straight line, position and velocity are estimable but frequency is almost completely unidentifiable: the observed trajectory is consistent with any oscillator that passes through that region with the observed velocity. At dt = 0.5, the GRU sees over half a period — enough to observe the trajectory curve, reverse direction, and begin returning, strongly constraining the frequency.

### Damping estimation is even harder

Damping gamma manifests as envelope decay: A(t) = A_0 * exp(-gamma * t). To estimate gamma, the observer must detect a measurable amplitude change:

    Delta_A / A_0 = 1 - exp(-gamma * T_cover) ~ gamma * T_cover   (for small gamma * T_cover)

For typical light damping (gamma ~ 0.1):

| dt   | T_cover | Amplitude change (gamma * T_cover) | Regime                    |
|------|---------|------------------------------------|---------------------------|
| 0.05 | 0.35    | 3.5%                               | Below noise floor         |
| 0.2  | 1.4     | 14%                                | Marginal; detectable with low noise |
| 0.5  | 3.5     | 35%                                | Clearly detectable        |

Even at the training dt = 0.2, damping estimation from 8 frames is marginal — the amplitude changes by only 14% over the context window. At dt = 0.05, the 3.5% change is likely below the encoder's noise floor. This implies that for lightly-damped systems, the GRU must rely on velocity-change patterns (acceleration) rather than amplitude decay to infer damping, which requires even more temporal coverage.

## Nyquist Constraint: Hard Upper Bound on dt

Sampling theory imposes a hard ceiling independent of architecture. The Nyquist-Shannon theorem requires:

    dt < pi / omega_max

where omega_max is the highest frequency component in the trajectory. For the oscillator with omega ~ 1, the Nyquist limit is dt < pi ~ 3.14 — permissive for the natural dynamics. However, discrete action forcing introduces impulse-like inputs that broaden the effective bandwidth, lowering the aliasing threshold.

At dt > pi / omega_max, the sampled trajectory is aliased: multiple distinct continuous trajectories produce identical discrete samples. This is a hard information-theoretic wall — no model, regardless of architecture or training, can disambiguate aliased observations.

## Asymmetric Generalization Prediction

The analysis predicts qualitatively different generalization behavior in two regimes:

### Regime 1: dt_eval > dt_train (increased temporal coverage)

More temporal coverage provides *more* Fisher information for theta than the GRU saw during training. Velocity estimates have lower noise (Var(v) proportional to 1/dt^2) but higher finite-difference bias (O(dt^2)). The net effect should be improved or maintained prediction quality, up to the point where finite-difference bias or Nyquist aliasing degrades the velocity estimates.

**Prediction:** dt-generalization to larger timesteps should succeed, potentially *improving* over training-dt performance, until approximately dt_eval ~ 5 * dt_train (where the finite-difference velocity deviates significantly from instantaneous velocity for oscillatory dynamics).

### Regime 2: dt_eval < dt_train (reduced temporal coverage)

Less temporal coverage provides *less* Fisher information for theta, degrading quadratically with dt. Momentum inference remains robust (local operation with sufficient samples). The predictor runs dynamics with correctly-estimated momentum but imprecisely-estimated system parameters.

**Prediction:** dt-generalization to smaller timesteps degrades gracefully. The expected failure signature is *phase drift* — predictions that initially move in the correct direction but accumulate frequency/damping error over the rollout horizon. This is qualitatively different from the catastrophic failure of encoder_frames=2 at non-training dt, which produces immediate velocity miscalibration and divergence.

At dt_eval ~ dt_train / 4, the Fisher information for frequency is 1/16 of its training-dt value, placing a practical floor on useful generalization.

### Complementary failure modes

The two architectures fail in opposite directions:

| Property             | encoder_frames=2                       | Finite-difference velocity inference    |
|----------------------|----------------------------------------|-----------------------------------------|
| dt-dependent quantity | Encoder output (velocity proportional to dt) | GRU observability window (T_cover = N * dt) |
| Failure direction    | dt_eval > dt_train (velocity overestimated) | dt_eval < dt_train (theta underidentified) |
| Failure mechanism    | Systematic bias in every latent frame  | Information loss in system identification |
| Failure symptom      | Immediate divergence (wrong velocity)  | Phase drift over rollout (wrong frequency) |
| Degradation          | Abrupt (bias is proportional to dt ratio) | Graceful (Fisher info degrades as dt^2) |
| Estimated useful range | ~1.5x around dt_train               | ~20x range (dt_train/4 to 5*dt_train)  |

This complementarity is not coincidental. It reflects a fundamental tradeoff: encoding velocity at the pixel level embeds the observation rate into the representation (biasing large-dt evaluation), while deferring velocity to post-hoc computation preserves the representation but limits the observation window (degrading small-dt system identification).

## The Irreducible Residual: Finite-Window System Identification

The temporal coverage dependence is not an architectural limitation that can be engineered away — it is an intrinsic property of finite-window system identification. No matter how the velocity and parameter inference are structured, a fixed number of observations at a given dt provides a fixed amount of information about the underlying dynamical system. The Fisher information bound I(omega) proportional to N * T_cover^2 / sigma^2 applies regardless of whether the inference is performed by a GRU, attention mechanism, or optimal statistical estimator.

What architecture *can* control is where the dt-dependence enters:

1. **Per-pixel (encoder_frames=2):** dt is embedded in the raw representation. Every downstream component inherits the bias. No post-hoc correction is possible.

2. **Per-step learned pathway (GRU on dt-normalized velocity):** dt enters only through the GRU's learned weights being calibrated to a training-dt velocity distribution. The dependence is weak (velocity values are dt-independent; only the noise statistics vary).

3. **Sequence-level coverage only (GRU on positions, hardcoded velocity):** dt enters only through how much of the trajectory is visible. The learned model never processes dt-dependent quantities. This is the minimum achievable dt-dependence for any architecture that performs temporal inference from a fixed number of frames.

4. **No temporal inference (set-based theta, hardcoded p):** Formally dt-free in the learned pathway, but sacrifices temporal ordering information, severely limiting theta estimation quality.

The proposed architecture (latent finite-difference velocity + GRU smoothing) operates between levels 2 and 3. The GRU processes dt-normalized velocity (level 2 per-step content) but its sequence-level behavior is governed by temporal coverage (level 3). The practical question is whether the GRU generalizes to temporal coverages outside its training distribution — an empirical question that the information-theoretic analysis bounds but does not fully resolve.

### Mitigation: multi-dt training augmentation

Training with varying dt per sequence (sampling dt uniformly from a range at dataset generation time) does not change the information-theoretic limits but teaches the GRU to:

1. Recognize how much information is available about theta at a given temporal coverage.
2. Fall back to a reasonable prior (population-mean theta) when coverage is insufficient for reliable estimation.
3. Weight local velocity signals against global curvature signals adaptively.

This is a training-time intervention, fully compatible with the proposed architecture, and is the natural way to push the model toward the theoretical bounds across dt values.

## Design Principle

**dt-generalization under finite temporal coverage is bounded by the Fisher information for system identification, which scales quadratically with the observation window.** A fixed context length of N frames provides Fisher information I(omega) proportional to N * dt^2 for frequency estimation. This places a fundamental floor at approximately dt_train / 4 below which system parameters become unidentifiable, regardless of architecture. The practical strategy is to (a) confine dt-dependence to the system-identification pathway (keeping position and momentum inference dt-free), and (b) expand the GRU's training distribution across dt values so it learns to operate gracefully at varying temporal coverages.

## Relationship to Previous Takeaways

| #  | Finding | Information pathway affected | Resolution |
|----|---------|------------------------------|------------|
| 1  | LSTM backbone hijacks Hamiltonian | Predictor: unconstrained pathway bypasses physics | Remove backbone |
| 2  | Encoder q/p misalignment | Decoder: asymmetric view collapses q/p structure | Decoder sees full z |
| 3  | Encoded-state recon doesn't reach predictor | Training: gradient path bypasses integration | Decode predicted states |
| 4  | Single-frame encoder can't produce q/p | Encoder: no velocity in single frame | encoder_frames >= 2 |
| 5  | Temporal coverage limits dt-generalization | Inference: finite window bounds system ID | Separate velocity pathways; multi-dt training |

Takeaways 4 and 5 are directly linked: #4 establishes that velocity information is structurally necessary for Hamiltonian dynamics, and #5 characterizes the cost of providing that velocity through post-hoc computation rather than encoder-level feature extraction. The progression from #4 to #5 shifts the bottleneck from a representational failure (no velocity at all) to an information-theoretic limit (insufficient temporal coverage for system identification) — a fundamentally gentler constraint that admits graceful degradation rather than catastrophic collapse.
