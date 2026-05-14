# Takeaway 6: Context-Length Identifiability is an Information-Theoretic Prerequisite, Not an Architectural Choice

**Context:** Takeaway 5 derives the Cramer-Rao bound on per-trajectory parameter identifiability from a finite-window state inferrer: `I(omega) ~ N * T_cover^2 / sigma^2`, with `T_cover = (N-1) * dt`. This takeaway records the conclusion that no per-step architectural augmentation of the predictor can pay down that bound, and formalizes the practical requirement on `infer_context_length` that follows from it.

## The Question

The Latent-* predictors (`LatentHamiltonianPredictor`, `LatentNeuralODEPredictor`) compute per-trajectory parameters theta once per rollout, from the first `infer_context_length` encoded latents (`predictors.py:625`, `predictors.py:460`). For environments whose slowest period is long relative to `infer_context_length * observation_dt`, the initial state estimate is provably underdetermined — most directly visible as phase drift in long-horizon rollouts at small dt (takeaway 5, Regime 2). A natural reflex is to look for an architectural fix that lets the model continue refining theta as the rollout proceeds.

Two architectural ideas were considered and rejected:

### Idea A: Sliding-window state inference

Maintain a FIFO of the last `T = infer_context_length` predicted q values during the rollout. Re-run the existing GRU on the buffer at every step (using the same `(q, v=Δq/dt, a)` convention) and produce a fresh theta. The Hamiltonian integrator continues to own `(q, p)` evolution; theta refreshes continuously from the GRU's view of the recent predicted trajectory.

### Idea B: Sequence-model Hamiltonian

Replace `H_net(q, p, theta)` with `H_net(q_{t-K..t}, p_t, theta)` — let the energy functional itself consume a window of recent phase coordinates, so non-Markovian effects can enter the dynamics directly without going through theta.

## Why Neither Idea Beats the Floor

### The information is either in the buffer or it isn't

The Cramer-Rao bound applies to *any* unbiased estimator of theta from the available samples, including a sliding-GRU estimator and a sequence-H estimator. Both ideas can only rearrange how the buffer's information is consumed; they cannot manufacture information that isn't there. For a buffer that spans `T_cover = (T-1) * dt`, the Fisher information for omega is bounded above by `I(omega) ~ T * T_cover^2 / sigma^2` regardless of whether the parameters live in an explicit `theta_dim` vector, in a GRU hidden state, or in a sequence-H's window inputs.

The diagnostic case: if the original 8-frame context at `dt = 0.05` (5.6% of one oscillation, takeaway 5) is insufficient for omega-identification, then the next 8 *predicted* frames at the same dt — produced by a model whose own theta is wrong — are no more informative than the originals. The buffer slides, but the temporal-coverage fraction does not.

### Sliding-window inference introduces a collapse risk

Sliding-window theta refresh re-introduces a recurrent backbone (GRU) into the per-step dynamics loop. This is structurally similar to the failure mode in takeaway 1, where an LSTM in the dynamics absorbed the per-step prediction signal and collapsed H_net to a constant. The sliding-theta variant has a `theta_dim` bottleneck (default 8) that the LSTM hijack did not, which reduces but does not eliminate the absorption surface: a fast-varying theta can still encode trajectory-local information into the energy landscape via H_net's theta-input pathway, defeating the "theta is slow, system-identifying" interpretation.

Mitigations exist (Δtheta regularizer, EMA on theta, frozen-then-thawed training curriculum), but each introduces hyperparameters and risk surface. None of them eliminate the Cramer-Rao floor — they only protect the Hamiltonian from being absorbed *given* that the floor allows useful theta updates at all.

### Sequence-H abandons the Hamiltonian structure

Letting `H_net` consume a window of recent `(q, p)` makes the dynamics non-Markovian by construction. The defining property of a Hamiltonian system — that the full future evolution depends only on the current phase point — is gone. Hamilton's equations `dq/dt = ∂H/∂p`, `dp/dt = -∂H/∂q` are derived assuming H is a state function; with H a trajectory functional, the autograd-computed `∂H/∂q_t`, `∂H/∂p_t` no longer correspond to canonical conjugate momenta. Symplecticity in the conservative limit, which motivates the semi-implicit Euler and implicit-midpoint integrators in `predictors.py:335` and `predictors.py:705`, no longer holds.

Sequence-H also has more absorption capacity than sliding-theta: it removes the `theta_dim` bottleneck and gives the dynamics block direct access to the history. The collapse mode would look like "H_net learns to read the next q out of the recent window and emits a flat energy with a sharp ridge wherever needed to reproduce it" — H_net itself becomes a sequence-to-sequence model with energy as a thin output.

### dt-generalization is preserved by neither architecture beyond what the current design already preserves

A sliding GRU on `(q, v=Δq/dt, a)` inherits exactly the dt-dependence of the original one-shot GRU: dt-normalized velocities are dt-independent in expectation, but the temporal coverage `T_cover = (T-1) * dt` still varies with dt. A sequence-H over `K` frames has the same property: `K * dt` represents different physical time at different dt. Both designs would still need multi-dt training (takeaway 5, "Mitigation") to generalize across observation rates. Neither outperforms the current one-shot inferrer on dt-generalization in any way that is not bottlenecked by the Cramer-Rao bound.

## The Practical Requirement

The honest mitigation for the underdetermination problem is to ensure the prerequisite is satisfied at the data + config level, not to add machinery that compensates for its absence.

> **Context-length identifiability prerequisite.** For the per-trajectory parameter vector theta produced by a Latent-* predictor's one-shot `infer()` to be reliably identifiable, the context window must satisfy
>
>     (infer_context_length - 1) * observation_dt  >=  T_slowest
>
> where `T_slowest` is the slowest physical period in the environment (e.g., `2 * pi / omega_min` for an oscillator with minimum natural frequency `omega_min`, or the period at the largest plausible pendulum amplitude). When this is violated, theta is underdetermined and rollouts will exhibit phase drift that no per-step architectural addition to the predictor can correct.

Equivalently: the design knob is `infer_context_length * observation_dt`, the *physical time covered by the context*, not `infer_context_length` alone.

### Concrete numbers for current environments

For the forced oscillator with natural frequency `omega ~ 1 rad/s` (period `T_period ~ 6.3`):

| infer_context_length | observation_dt | T_cover | Coverage | Identifiability regime |
|----------------------|----------------|---------|----------|------------------------|
| 8                    | 0.05           | 0.35    | 5.6%     | Underdetermined (current dt-gen failure mode) |
| 8                    | 0.1            | 0.7     | 11%      | Marginal |
| 8                    | 0.2            | 1.4     | 22%      | Estimable |
| 8                    | 0.5            | 3.5     | 56%      | Well-constrained |
| 16                   | 0.05           | 0.75    | 12%      | Marginal |
| 32                   | 0.05           | 1.55    | 25%      | Estimable |

To preserve identifiability at `dt = 0.05` without lowering the training dt, `infer_context_length` would need to roughly quadruple (from 8 to ~32) to recover the `T_cover ~ 1.4` of the `dt = 0.2` reference row. This is the unavoidable cost of operating at smaller dt with a fixed-window inferrer.

For the forced pendulum at large amplitude, `T_period` lengthens by up to ~1.5x relative to the small-amplitude harmonic limit (the well-known amplitude-period dependence of a nonlinear pendulum). The context-length requirement scales with the *largest* expected period over the training distribution of `init_state_range`, not the small-amplitude approximation.

## Design Principle

**For Latent-* predictors, set `infer_context_length` from the environment's slowest physical period and the chosen `observation_dt`, not from heuristics or compute budget.** A predictor with a too-short context will exhibit phase drift that looks like a model failure but is in fact an information-theoretic prerequisite violation. Architectural augmentation (sliding-window inference, sequence-H, learned online filters) cannot remediate this: it can only redistribute the available information among the predictor's internal variables, and it does so at the cost of structural properties (Hamiltonian discipline, theta-as-system-identity, symplecticity) that the original design depends on.

## Relationship to Previous Takeaways

| #  | Finding | Information pathway affected | Resolution |
|----|---------|------------------------------|------------|
| 1  | LSTM backbone hijacks Hamiltonian | Predictor: unconstrained pathway bypasses physics | Remove backbone |
| 2  | Encoder q/p misalignment | Decoder: asymmetric view collapses q/p structure | Decoder sees full z |
| 3  | Encoded-state recon doesn't reach predictor | Training: gradient path bypasses integration | Decode predicted states |
| 4  | Single-frame encoder can't produce q/p | Encoder: no velocity in single frame | encoder_frames >= 2 |
| 5  | Temporal coverage limits dt-generalization | Inference: finite window bounds system ID | Separate velocity pathways; multi-dt training |
| 6  | Context-length is a Cramer-Rao floor, not a knob | Inference: identifiability prerequisite | Set (N-1)*dt >= T_slowest at config time |

Takeaway 6 is the corollary of takeaway 5 applied to architecture choices. Takeaway 5 establishes the bound; takeaway 6 records the conclusion that the predictor's per-step block cannot be made to evade it, and locates the design lever at the configuration of `infer_context_length` and `observation_dt` rather than at the predictor head. Takeaway 1 provides the cautionary counterexample: introducing a recurrent backbone into the per-step dynamics (the natural sliding-window-inference instinct) re-creates the absorption surface that motivated the original "infer once, integrate many" discipline.
