# Takeaway 4: Single-Frame Encoding Makes the Hamiltonian Q/P Split Ill-Posed

**Experiment:** LatentHamiltonianPredictor (non-separable H(z, theta), semi-implicit Euler, GRU theta inference, per-frame action embedding), JEPA training with hybrid reconstruction on oscillator_visual, 60k sequences, dt=0.2, encoder_frames=1 vs encoder_frames=2.

## Observation

When `encoder_frames=1`, the LatentHamiltonianPredictor's dynamics collapse to mean-prediction: the predictor outputs the population-mean latent at every rollout step, which decodes to the ball at its time-averaged position. Reconstructions are near-perfect (the encoder captures appearance from single frames), but the predictor cannot track dynamics. Training eventually terminates in a NaN cliff, regardless of unroll length, H_net activation choice, or gradient clipping. When `encoder_frames=2` with the same architecture, the model trains to completion with PSNR >33 and tracks oscillatory dynamics correctly.

## Evidence

### Controlled comparison

All runs use the same architecture (non-separable Softplus H_net, GRU theta inference, infer_context_length=8, hybrid_recon_weight=0.1, SIGReg lambda=0.1):

| Run | encoder_frames | pred_length | H_net | Cliff epoch | Peak PSNR | Dynamics |
|-----|---------------|-------------|-------|-------------|-----------|----------|
| 405968 | 2 | 5 | Softplus | none | >33 | Correct |
| 405817 | 1 | 5 | Softplus | 14 | 31.4 | (not imaged) |
| 405855 | 1 | 5 | LayerNorm+SiLU | 11 | 30.7 | (not imaged) |
| 405875 | 1 | 1 | LayerNorm+SiLU | 24 | (not recorded) | (not imaged) |
| 406063 | 1 | 3 | Softplus | 13 | (not recorded) | Mean-prediction |

The only variable that separates healthy training (405968) from collapse (all others) is `encoder_frames`.

### Rollout grid at epoch 8 (run 406063)

The rollout grid captures the dynamics failure visually:

- **GT row:** sharp yellow dots oscillating sinusoidally across the frame sequence.
- **PRED row:** context reconstructions (first ~10 frames) are sharp and correctly positioned. Predicted frames progressively blur and drift toward a fixed central position. By the end of the prediction horizon, the output is a diffuse blob at the ball's mean location.
- **ERROR row:** low error in the context region, growing halos during prediction, converging to a steady-state error at the mean position.

This is the signature of a predictor that outputs the population-mean latent at every step — the safest prediction when the dynamics model has learned nothing about temporal evolution.

### Reconstruction quality at epoch 8 (run 406063)

Reconstructions (encode-then-decode, no prediction) are near-perfect at the same epoch where rollouts show mean-prediction collapse. The encoder captures single-frame appearance with high fidelity. The failure is purely in the predictor's dynamics, not in the encoder's representation of individual frames.

### Timing: "immediately after perfect reconstruction"

The NaN cliff follows closely after reconstructions converge. Once reconstruction loss approaches zero, the encoder's output manifold stabilizes and the only remaining gradient signal to the encoder comes from the latent prediction loss, which flows through H_net's `autograd.grad(..., create_graph=True)`. But the predictor is stuck in the mean-prediction regime, so its gradients push the encoder toward constant (easily-predicted) latents. This directly conflicts with SIGReg's diversity objective, creating an unstable gradient equilibrium that the second-order autograd amplifies into NaN.

### Unroll length delays but does not prevent the cliff

Reducing `pred_length` from 5 to 1 delayed the cliff from epoch 11 to epoch 24 (a ~2x delay, not elimination). This rules out BPTT-through-unroll as the primary mechanism — the per-step second-order autograd stiffness is sufficient to cause the cliff on its own. Longer unrolls amplify the problem by a factor of ~2x but are not the root cause.

### H_net activation choice does not prevent the cliff

Replacing Softplus with LayerNorm + SiLU (bounded second derivatives, input normalization) moved the cliff from epoch 14 to epoch 11 — earlier, not later. The cliff is not primarily a function of H_net tail behavior or second-derivative smoothness. It is driven by the encoder-side ill-posedness.

## Analysis

### Why single-frame encoding breaks Hamilton's equations

Hamilton's equations evolve the state via:

    dq/dt = +dH/dp    (velocity from momentum gradient)
    dp/dt = -dH/dq    (force from position gradient)

This requires q and p to have physically distinct roles: q is position (configuration), p is momentum (velocity-like). The distinction is not a labeling convention — it determines the *direction* of information flow in the dynamics.

With `encoder_frames=2`, the encoder receives two channel-concatenated frames `[I_t, I_{t+1}]`. The temporal difference between frames provides a direct signal for velocity. The encoder can — and empirically does — place position information in q and velocity information in p, because these are the two distinguishable types of information available in a frame pair.

With `encoder_frames=1`, the encoder receives a single frame. A single frame contains position information (where the ball is) but **no velocity information** (velocity is a temporal derivative; it does not exist in a single image). Both q and p must be populated from position-only features. The resulting q/p split is geometrically arbitrary — both halves encode appearance, neither encodes velocity.

### Why the Hamiltonian compensates by becoming stiff

When q and p are both position-like, H_net must learn a contorted scalar field to produce dynamics that match the data. The correct Hamiltonian for a harmonic oscillator is a smooth quadratic H = (1/2)(q^2 + p^2), with bounded and constant second derivatives. But this only works when q and p have their canonical meaning.

When q and p are both arbitrary projections of appearance features, H_net must learn a field whose gradients dH/dq and dH/dp happen to produce correct trajectories despite the canonical structure being violated. This field necessarily has sharper local features (higher curvature) than the physically-correct Hamiltonian, because it must compensate for the encoder's inability to provide the right inputs.

As training progresses, this curvature sharpens further (the encoder concentrates on a narrow latent manifold, and H_net sharpens its ridge to match). Eventually, a rare outlier batch encounters a high-curvature region, producing a large but finite gradient. The optimizer step displaces weights into a pathological basin, and the next forward pass goes non-finite.

### Why this is a representational failure, not a numerical one

The previous diagnosis framed this as "stiff second-order autograd" — a numerical issue amenable to activation-function or gradient-clipping fixes. The visual evidence from 406063 contradicts this framing. The rollout grid shows the predictor failing at dynamics (mean-prediction collapse) *before* any NaN. The cliff is a downstream consequence of the representational failure, not the primary problem.

No amount of activation-function engineering (SiLU, Softplus, tanh clamping) or gradient infrastructure (clipping, skip guards, checkpoint reloading) can fix a predictor that receives canonically-meaningless inputs. These interventions address the symptom (NaN) but not the cause (the encoder cannot produce a q/p split from a single frame).

### Relationship to theta inference

The LatentHamiltonianPredictor's GRU infers per-trajectory parameters theta from `infer_context_length=8` latent frames. With `encoder_frames=1`, each context frame carries no velocity information, so the GRU must infer trajectory parameters (damping, frequency, amplitude) purely from the temporal pattern of position-only latents. This is noisier and less stable than with `encoder_frames=2`, where each frame already carries velocity. The noisier theta contributes additional variance to H_net's input distribution, increasing the probability of encountering high-curvature regions per batch.

## Design Principle

**Hamilton's equations require canonically-structured latent variables. A single-frame visual encoder cannot provide them.** The q/p split in a Hamiltonian predictor is not merely a notational convention for partitioning latent dimensions — it is a structural requirement that determines the direction of information flow in the dynamics. When the encoder cannot distinguish position from velocity (because velocity is absent from a single frame), the split is vacuous and the Hamiltonian's inductive bias becomes a liability rather than an asset.

The minimum encoder temporal context for a Hamiltonian predictor is `encoder_frames=2` (consecutive frame pairs), which provides a direct signal for velocity via the temporal difference. This is the simplest encoder architecture that makes the q/p split well-posed. A more principled alternative would be an encoder with an explicit velocity-inference head, but the frame-pair approach achieves the same result implicitly and has been empirically validated.

## Relationship to Previous Takeaways

| # | Failure mode | Where structure breaks | Fix |
|---|-------------|----------------------|-----|
| 1 | LSTM backbone hijacks Hamiltonian | Predictor: unconstrained pathway bypasses physics | Remove backbone |
| 2 | Encoder q/p misalignment (decoder side) | Decoder: only sees q, so encoder puts dynamics in p | Decoder sees full z |
| 3 | Encoded-state recon doesn't reach predictor | Training: gradient path bypasses integration | Decode predicted states |
| 4 | Single-frame encoder can't produce q/p (encoder side) | Encoder: no velocity in single frame | encoder_frames >= 2 |

Takeaways 2 and 4 are complementary: #2 diagnosed misalignment caused by the decoder's asymmetric view of q and p (fixed by letting the decoder see full z); #4 diagnoses misalignment caused by the encoder's inability to distinguish position from velocity (fixed by giving the encoder frame pairs). Together they bracket the representation problem: the q/p split must be well-posed at both ends of the encoder-decoder chain.

The progression reveals that physics-informed latent dynamics require alignment at every point in the pipeline: (1) the non-physics pathway must be capacity-limited so the physics pathway is used, (2) the decoder must see the full state so the encoder distributes information correctly, (3) the reconstruction gradient must flow through the predictor so the physics is activated, and (4) the encoder must have sufficient temporal context to produce canonically-structured latents. Violating any one of these conditions collapses the Hamiltonian to a trivial or degenerate function.
