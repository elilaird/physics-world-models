# Forcing the Physics Inductive Bias

The core problem across all three failure modes (Takeaways 1-3) is the same: the optimizer finds a shortcut that bypasses the Hamiltonian integration. The fixes so far close specific shortcuts — backbone capacity (T1), q/p alignment (T2), gradient topology (T3). The approaches below attack the problem from other angles: some make the physics path easier to optimize, others make the shortcut harder.

## Currently Running

### Predicted-state reconstruction (Takeaway 3 fix)

Decode the predictor's output rather than the encoder's output for the hybrid reconstruction loss:

    L_pred_recon = || decode(predict(z, a)) - x_{t+1} ||^2

Creates the gradient chain: L → decoder → predicted q → predictor → T_net. Forces T_net to develop meaningful dq/dt = dT/dp gradients.

**Status:** Running with `hybrid_recon_weight=0.1`. This is the minimal fix for the gradient topology gap.

---

## Candidate Approaches

### 1. Multi-step autoregressive prediction loss

**Idea:** Train with k-step autoregressive rollouts in latent space — feed `pred_z` back as input for the next step instead of using teacher-forced ground truth.

**Why it helps:** The identity-on-q shortcut works perfectly for one step because the q change between consecutive frames is tiny. Over multiple steps, copying q accumulates drift while correct Hamiltonian integration stays accurate. This makes the shortcut more expensive.

**Implementation:** In the sliding window loop, replace teacher-forced `w_states[:, :-1]` with autoregressive unrolling:

```python
z = w_states[:, 0]  # initial state
for t in range(n_pred):
    z = model.predictor(z.unsqueeze(1), w_actions[:, t:t+1]).squeeze(1)
    latent_pred_loss += ((z - w_states[:, t+1]) ** 2).mean() / (n_pred * num_windows)
```

**Tradeoffs:**
- Deeper computation graphs, more memory usage
- Potential training instability from error compounding
- Start with k=2 or k=3, not full sequence length
- Could use scheduled increase: teacher-forced early, autoregressive later

**Priority:** High — try next. Attacks the problem from a fundamentally different angle than reconstruction. The identity shortcut fails over time rather than us forcing gradients through the physics path.

---

### 2. Energy variance regularization

**Idea:** Directly penalize flat energy landscapes by adding a variance term:

    L_energy = -lambda_e * Var_t[H(z_t)]

This pushes V_net and T_net away from constant-energy solutions.

**Variants:**
- **Temporal variance:** Variance of H across timesteps within a trajectory (energy should change as the system evolves)
- **Batch variance:** Variance of H across different states in a batch (different configurations should have different energies)
- **Contrastive:** Sample random latent pairs, encourage H to assign different energies to different states

**Tradeoffs:**
- Does not guarantee the energy is physically meaningful — just non-trivial
- The dynamics loss must still shape what H actually represents
- Risk of the energy network learning arbitrary high-variance functions that don't correspond to physics
- Lambda scheduling may be needed: strong early to escape the flat basin, weaker later so dynamics loss takes over

**Priority:** Medium — cheapest direct fix if energy stays flat after predicted-state recon. Good insurance policy.

---

### 3. Velocity / finite-difference consistency

**Idea:** The encoder sees frame pairs (`encoder_frames=2`), so it has access to velocity information. Explicitly supervise Hamilton's equations using the encoder's own outputs:

    L_vel = || (q_{t+1} - q_t) / dt - dT/dp(p_t) ||^2

This directly tells T_net: "your gradient at p_t must match the observed position change."

**Analogously for momentum:**

    L_vel_p = || (p_{t+1} - p_t) / dt - (-dV/dq(q_t) - gamma * dT/dp(p_t) + G(a_t)) ||^2

This supervises the full port-Hamiltonian equations.

**Tradeoffs:**
- Requires the encoder to already place position/momentum correctly — works best combined with predicted-state reconstruction (Takeaway 3)
- Finite differences are noisy (amplify encoder noise by 1/dt)
- Assumes the encoder's frame-to-frame changes are smooth enough for finite differencing
- Elegant: directly supervises Hamilton's equations rather than indirectly through prediction

**Priority:** Medium — research-interesting and principled. Best applied after the q/p alignment is confirmed working (i.e., after predicted-state recon shows position evolving in q).

---

### 4. Symplectic regularization on the flow map

**Idea:** For a true Hamiltonian system, the Jacobian of the flow map (predictor) should be symplectic. Penalize deviations:

    L_symp = || J^T * Omega * J - Omega ||_F

where Omega is the canonical symplectic matrix `[[0, I], [-I, 0]]` and J = d(predictor)/dz is the predictor's Jacobian.

This doesn't require knowing H explicitly — it constrains the predictor's dynamics to be Hamiltonian-like regardless of what H looks like.

**Tradeoffs:**
- Computing the full Jacobian is expensive: D^2 autograd calls (D=64 → 4096 backward passes per sample)
- Stochastic approximation possible: project Jacobian onto random directions, check symplecticity in projection
- Only valid for the conservative part of the dynamics; the port-Hamiltonian dissipation and forcing terms break symplecticity by design
- Would need to apply only to the Hamiltonian flow component, not the dissipation/forcing

**Priority:** Low — expensive and conceptually awkward for port-Hamiltonian (which is deliberately non-symplectic). More suited for conservative systems.

---

### 5. Stop-gradient on the identity component

**Idea:** In the Euler step `q_{t+1} = q_t + dt * dT/dp`, detach `q_t` from the residual/copy path:

```python
q_next = q_t.detach() + dt * self._dT_dp(p)
```

The only gradient into T_net now comes from the `dt * dT/dp` term. The predictor cannot improve its loss by better copying — it must improve the physics term.

**Tradeoffs:**
- Aggressive: removes the predictor's ability to learn any q-residual correction
- Could destabilize training if T_net isn't ready to carry the full prediction load early on
- Might need a warmup schedule: full gradients early, stop-gradient later
- Similarly for p: `p_next = p_t.detach() + dt * (-dV/dq + G(a))`

**Priority:** Low — nuclear option. Try if predicted-state recon + multi-step still shows identity-on-q behavior. The stop-gradient is a guarantee that the physics path is the only gradient path, but it's harsh.

---

## Suggested Experiment Sequence

| Order | Experiment | Config change | What to watch |
|-------|-----------|---------------|---------------|
| 1 | Predicted-state recon | `hybrid_recon_weight=0.1` (already running) | energy_time_var should stay high (not collapse like epoch 9→16) |
| 2 | + Multi-step rollout (k=3) | New training loop option | Rollout PSNR/LPIPS should improve; identity shortcut penalized |
| 3 | + Energy variance reg | `energy_var_lambda=0.01` | Insurance if energy still flat after (1)+(2) |
| 4 | Velocity consistency | `velocity_consistency_lambda=0.1` | dT/dp matches observed finite differences |
| 5 | Stop-gradient (if needed) | Code change in predictor | Last resort; confirms whether identity path is the sole problem |

The first two are likely sufficient. Predicted-state recon fixes the gradient topology; multi-step rollout penalizes the identity shortcut over time. Together they close the two main failure channels. Energy variance reg is cheap insurance. Velocity consistency and stop-gradient are research-interesting but unlikely to be needed if (1)+(2) work.
