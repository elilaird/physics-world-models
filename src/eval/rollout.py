import numpy as np
import torch


@torch.no_grad()
def visual_open_loop_rollout(model, images, actions, dt=None):
    """Open-loop rollout for visual world models with flat latents.

    Encodes all frames with channel-concatenated overlapping windows, then
    calls ``predictor.infer`` ONCE on the first ``infer_context_length``
    latents to produce an initial state dict (carrying z, optional q/p, and
    optional static theta), and then ``predictor.unroll`` to roll out over
    the remaining horizon.

    Infer-once-per-rollout is the architectural discipline that keeps GRU
    state inference out of the per-step dynamics loop (takeaways/01).

    Args:
        model: VisualWorldModel (encoder, decoder, predictor).
        images: (B, T+1, C, H, W) ground-truth image sequence.
        actions: (B, T) discrete action indices.
        dt: optional timestep override for ODE-based predictors.
            When None, uses the predictor's training dt (self.dt).

    Returns:
        dict with:
            pred_latents: (B, horizon, D) predicted latents
            true_latents: (B, N_latents, D) encoded ground-truth latents
            pred_images: (B, horizon, C, H, W) decoded predicted frames
        where N_latents = N - encoder_frames + 1, horizon = N_latents - infer_ctx
    """
    B, N, C, H, W = images.shape
    infer_ctx = getattr(model, "infer_context_length", model.context_length)
    K = model.encoder_frames

    # Encode all ground-truth frames → latent states (encoder output IS the state)
    mu_all = model.encode_sequence(images)  # (B, N_latents, D)
    N_latents = mu_all.shape[1]
    D = mu_all.shape[2]

    true_latents = mu_all
    horizon = N_latents - infer_ctx

    # Transition actions: action[K-1+i] drives latent i → i+1, so the actions
    # aligned with the latent sequence start at index K-1.
    transition_actions = actions[:, K - 1:]  # (B, N_latents - 1)

    # Context = first infer_ctx latents. Context-internal actions drive those
    # first latents forward and are consumed by the GRU to infer theta.
    context = true_latents[:, :infer_ctx]
    context_actions = transition_actions[:, : infer_ctx - 1].long()

    # Unroll actions: the (infer_ctx - 1)-th transition bridges context to
    # the first predicted step; then we need `horizon` more actions.
    unroll_actions = transition_actions[
        :, infer_ctx - 1 : infer_ctx - 1 + horizon
    ].long()

    # Infer initial state (runs GRU once for Latent-* predictors). The
    # @torch.no_grad() decorator on this function disables grad globally,
    # but LatentHamiltonianPredictor.step has @torch.enable_grad() locally
    # so its autograd-based ∂H/∂z computation still works at eval time.
    state = model.predictor.infer(
        context, context_actions=context_actions,
        dt=dt or model.observation_dt,
    )
    pred_latents = model.predictor.unroll(state, unroll_actions, horizon, dt=dt)

    pred_images = model.decode(
        pred_latents.reshape(B * horizon, D)
    ).reshape(B, horizon, C, H, W)

    return {
        "pred_latents": pred_latents,
        "true_latents": true_latents,
        "pred_images": pred_images,
    }


@torch.no_grad()
def generate_visual_trajectory(env, init_state, actions, dt, render_opts):
    """Roll out an environment and render each state to an image.

    Args:
        env: PhysicsControlEnv with render_state().
        init_state: (state_dim,) tensor.
        actions: (T,) tensor of discrete action indices.
        dt: timestep for env.step().
        render_opts: dict passed to env.render_state() (img_size, color, etc.).

    Returns:
        images: (T+1, C, H, W) float tensor in [0, 1].
        states: (T+1, state_dim) tensor.
    """
    states = [init_state]
    state = init_state.clone()
    for t in range(len(actions)):
        state = env.step(state, int(actions[t].item()), dt)
        states.append(state)

    images = []
    for s in states:
        img = env.render_state(s, **render_opts)  # (H, W, C) in [0, 1]
        images.append(img.permute(2, 0, 1))  # (C, H, W)

    return torch.stack(images).float(), torch.stack(states).float()


@torch.no_grad()
def visual_dt_generalization_test(
    model, env, dt_values, cfg, n_seqs=8, seq_len=None,
    energy_radius_range_override=None,
    fixed_init_state=None,
    eval_dataset_dir=None,
    band_label=None,
):
    """Test visual model across different dt values.

    For each dt, generates fresh trajectories from the environment, runs the
    visual open-loop rollout, and compares predicted vs ground-truth frames.

    Args:
        model: VisualWorldModel.
        env: PhysicsControlEnv with render_state().
        dt_values: list of dt values to test.
        cfg: Hydra config (for env render settings and init_state_range).
        n_seqs: number of trajectories to generate per dt.
        seq_len: number of action steps per trajectory.
            Defaults to context_length + 10.
        energy_radius_range_override: optional (r_min, r_max) tuple/list that
            overrides the env-config-derived energy_radius_range AND forces
            sampling_mode="energy_radius". Used by visual_energy_stratified_test
            to run rollouts against a specific energy band. Default None
            preserves all existing behavior.
        fixed_init_state: optional state tensor that, when set, is used as
            the init for EVERY rollout in this dt-gen call (skipping
            env.sample_initial_state). Used by visual_fixed_init_stratified_test
            to collapse within-band init heterogeneity so only action-sequence
            variance remains. Default None preserves all existing behavior.
        eval_dataset_dir: optional path to a canonical eval dataset directory
            (produced by generate_eval_dataset.py). When set, the per-rollout
            loop loads pre-rendered (images, actions) from
            <eval_dataset_dir>/<band_label>/dt={dt}.npz instead of sampling
            at runtime. Used by paired cross-predictor comparison so every
            eval consumes identical trajectories. Default None preserves
            existing runtime-sampling behavior.
        band_label: required when eval_dataset_dir is set; specifies which
            band sub-directory to read ("low"/"med"/"high", or "all" for
            un-stratified). Ignored when eval_dataset_dir is None.

    Returns:
        dict mapping dt -> {
            'pred_images':   (n_seqs, horizon, C, H, W),
            'true_images':   (n_seqs, horizon, C, H, W),
            'metrics':       dict from compute_visual_metrics,
            'latent_mse':    float (mean across all seqs/steps; back-compat scalar),
            'latent_curves': dict of (n_seqs, horizon) CPU tensors with keys
                latent_mse, latent_cosine, latent_norm_l2,
                persistence_mse, persistence_cosine, persistence_norm_l2,
            'qp_curves':     dict of (n_seqs, horizon) CPU tensors with keys
                q_mse, p_mse, persistence_q_mse, persistence_p_mse, OR
                None when D is odd (no q/p split).
            'rollout_grid':  (C, H_grid, W_grid) CPU tensor, GT|Pred|Error grid.
        }
    """
    from omegaconf import OmegaConf
    from src.eval.metrics import (
        compute_visual_metrics,
        compute_latent_divergence_metrics,
        compute_qp_divergence_metrics,
    )

    # Grid layout and horizon bookkeeping use infer_context_length because
    # visual_open_loop_rollout seeds from the first infer_ctx latents.
    ctx_len = getattr(model, "infer_context_length", model.context_length)
    if seq_len is None:
        seq_len = ctx_len + 10

    # Pull the env/sampling config. Prefer cfg.dataset.env when present: that
    # is the block describing how the training data was generated (it carries
    # init_sampling / energy_radius_range). cfg.env is the default env config
    # and may not have the sampling keys. Callers without a dataset block
    # fall back to cfg.env for backwards compatibility.
    if "dataset" in cfg and "env" in cfg.dataset:
        env_cfg = cfg.dataset.env
    else:
        env_cfg = cfg.env

    render_opts = {
        "img_size": env_cfg.get("img_size", 64),
        "color": env_cfg.get("color", True),
        "render_quality": env_cfg.get("render_quality", "medium"),
    }
    for k in ("ball_color", "bg_color", "ball_radius"):
        v = env_cfg.get(k, None)
        if v is not None:
            render_opts[k] = list(v) if hasattr(v, "__iter__") else v

    # Init state sampling — route through env.sample_initial_state so the eval
    # distribution matches generate_dataset.py. When the dataset was generated
    # with init_sampling: energy_radius, eval trajectories start on the same
    # energy contours, not from a uniform box that biases toward low-energy /
    # near-equilibrium states.
    sampling_mode = env_cfg.get("init_sampling", "uniform_box")
    if energy_radius_range_override is not None:
        # Caller (e.g., visual_energy_stratified_test) is providing a band-specific
        # sub-range. Force energy_radius sampling and use the override.
        energy_radius_range = list(energy_radius_range_override)
        sampling_mode = "energy_radius"
    else:
        energy_radius_range = env_cfg.get("energy_radius_range", None)
        if energy_radius_range is not None:
            energy_radius_range = list(energy_radius_range)
    init_state_range = (
        OmegaConf.to_container(env_cfg.init_state_range, resolve=True)
        if "init_state_range" in env_cfg else None
    )

    device = next(model.parameters()).device
    results = {}

    # Resolve where image/action batches come from for each dt:
    #   - eval_dataset_dir set: load pre-rendered (images, actions) from disk.
    #     band_label decides which band sub-directory to read from.
    #   - otherwise: sample trajectories at runtime (existing back-compat path).
    use_dataset = eval_dataset_dir is not None

    if use_dataset:
        # Lazy import to keep the existing back-compat path free of new deps.
        from src.eval.eval_dataset_io import load_band_dt_npz
        if band_label is None:
            raise ValueError(
                "eval_dataset_dir requires band_label (one of 'low', 'med', 'high'). "
                "Pass band_label='all' to load the 'med' band sub-directory as "
                "a pooled un-stratified view (see load_band_dt_npz for the remap)."
            )

    for dt in dt_values:
        if use_dataset:
            loaded = load_band_dt_npz(eval_dataset_dir, band=band_label, dt=float(dt))
            # Sanity-check that the dataset's batch size matches what the caller
            # asked for. Slice down rather than crash; if the dataset has FEWER
            # sequences than n_seqs the downstream metrics will be silently
            # weaker — flag that loudly.
            n_loaded = loaded["images"].shape[0]
            if n_loaded < n_seqs:
                raise ValueError(
                    f"Eval dataset at {eval_dataset_dir} has only {n_loaded} "
                    f"sequences in band={band_label} dt={dt}, but n_seqs={n_seqs} "
                    f"was requested. Regenerate the dataset with eval_dataset.n_seqs"
                    f"={n_seqs} or pass eval.n_rollouts={n_loaded}."
                )
            images_batch = torch.from_numpy(loaded["images"][:n_seqs]).to(device)
            actions_batch = torch.from_numpy(loaded["actions"][:n_seqs]).to(device)
        else:
            all_images = []
            all_actions = []
            for _ in range(n_seqs):
                if fixed_init_state is not None:
                    # Caller (e.g., visual_fixed_init_stratified_test) wants every
                    # rollout to start from the SAME init. Clone so each rollout
                    # gets its own tensor object (defensive — generate_visual_trajectory
                    # treats init_state as immutable, but downstream callers shouldn't
                    # be coupled to that contract).
                    init_state = fixed_init_state.clone() if hasattr(fixed_init_state, "clone") else fixed_init_state
                else:
                    init_state = env.sample_initial_state(
                        sampling_mode=sampling_mode,
                        init_state_range=init_state_range,
                        energy_radius_range=energy_radius_range,
                        variable_params=None,
                    )

                actions = torch.randint(0, env.action_dim, (seq_len,))
                imgs, _ = generate_visual_trajectory(env, init_state, actions, dt, render_opts)
                all_images.append(imgs)
                all_actions.append(actions)

            images_batch = torch.stack(all_images).to(device)
            actions_batch = torch.stack(all_actions).to(device)

        # Run visual rollout (pass dt so ODE-based predictors integrate correctly)
        K = model.encoder_frames
        rollout = visual_open_loop_rollout(model, images_batch, actions_batch, dt=dt)
        pred_images = rollout["pred_images"]       # (n_seqs, horizon, C, H, W)
        true_latents = rollout["true_latents"]     # (n_seqs, N_latents, D)
        pred_latents = rollout["pred_latents"]     # (n_seqs, horizon, D)

        horizon = pred_images.shape[1]
        gt_images = images_batch[:, K - 1 + ctx_len:]  # (n_seqs, horizon, C, H, W)
        gt_latents = true_latents[:, ctx_len:]          # (n_seqs, horizon, D)

        latent_mse = ((pred_latents - gt_latents) ** 2).mean().item()

        # Per-step latent divergence (dynamics metrics) + persistence baseline.
        # The last context frame is index ctx_len-1 in true_latents (the latent
        # the predictor saw last before starting to predict).
        z_context_last = true_latents[:, ctx_len - 1]   # (n_seqs, D)
        latent_curves = compute_latent_divergence_metrics(
            pred_latents, gt_latents, z_context_last
        )
        # q/p split is Hamiltonian-only. We can't introspect predictor type
        # here without circular imports, so always compute it when D is even
        # and let the caller decide whether to use it.
        D = pred_latents.shape[-1]
        if D % 2 == 0:
            qp_curves = compute_qp_divergence_metrics(
                pred_latents, gt_latents, z_context_last
            )
        else:
            qp_curves = None

        vis_metrics = compute_visual_metrics(pred_images, gt_images)

        # Build rollout grid (GT | Pred | Error) for a few samples
        N = images_batch.shape[1]  # total raw frames
        C, H, W = images_batch.shape[2], images_batch.shape[3], images_batch.shape[4]
        n_show = min(4, n_seqs)
        blank = torch.zeros(C, H, W, device=device)

        # Encode context
        ctx_images = images_batch[:n_show, :ctx_len + K - 1]
        ctx_mu = model.encode_sequence(ctx_images)  # (n_show, ctx_len, D)
        D_enc = ctx_mu.shape[2]
        ctx_recon = model.decode(ctx_mu.reshape(n_show * ctx_len, D_enc)).reshape(n_show, ctx_len, C, H, W)

        rows = []
        for i in range(n_show):
            gt_row = torch.cat([images_batch[i, t] for t in range(N)], dim=-1)
            lead_blanks = [blank] * (K - 1)
            recon_frames = [ctx_recon[i, t] for t in range(ctx_len)]
            pred_frames = [pred_images[i, t] for t in range(horizon)]
            pred_row = torch.cat(lead_blanks + recon_frames + pred_frames, dim=-1)
            err_blanks = [blank] * (K - 1 + ctx_len)
            err_frames = [(pred_images[i, t] - gt_images[i, t]).abs() for t in range(horizon)]
            err_row = torch.cat(err_blanks + err_frames, dim=-1)
            rows.extend([gt_row, pred_row, err_row])

        grid = torch.cat(rows, dim=-2).clamp(0, 1).cpu()

        results[dt] = {
            "pred_images": pred_images,
            "true_images": gt_images,
            "metrics": vis_metrics,
            "latent_mse": latent_mse,
            "latent_curves": {k: v.cpu() for k, v in latent_curves.items()},
            "qp_curves": (
                {k: v.cpu() for k, v in qp_curves.items()}
                if qp_curves is not None else None
            ),
            "rollout_grid": grid,
        }

    return results


@torch.no_grad()
def visual_energy_stratified_test(
    model, env, dt_values, cfg, energy_radius_range,
    n_seqs=8, seq_len=None,
    eval_dataset_dir=None,
):
    """Run visual_dt_generalization_test once per energy band.

    Slices the supplied energy_radius_range into three equal sub-ranges
    (low / med / high) and calls visual_dt_generalization_test for each
    band with energy_radius_range_override set to the band's sub-range.

    Note: for envs whose energy-vs-radius mapping is non-linear
    (oscillator: energy ∝ r^2), the radius-spaced sub-ranges correspond
    to non-uniform energy intervals. The "low/med/high" labels describe
    radius bands, not equal energy partitions.

    Args:
        model:               VisualWorldModel.
        env:                 PhysicsControlEnv with render_state() and
                             _sample_energy_radius_state() implemented.
        dt_values:           list of dt values to test (per band).
        cfg:                 Hydra config (forwarded to dt-gen test).
        energy_radius_range: (r_min, r_max) tuple/list — the full
                             eval distribution to be split into bands.
        n_seqs:              n_rollouts per band (NOT divided across bands).
        seq_len:             optional seq_len override for dt-gen test.

    Returns:
        dict {band: dt_results_dict} where band is "low" / "med" / "high"
        and dt_results_dict has the same schema as
        visual_dt_generalization_test's return value (one entry per dt).
    """
    import numpy as np
    r_min, r_max = float(energy_radius_range[0]), float(energy_radius_range[1])
    edges = np.linspace(r_min, r_max, 4)
    bands = {
        "low":  (float(edges[0]), float(edges[1])),
        "med":  (float(edges[1]), float(edges[2])),
        "high": (float(edges[2]), float(edges[3])),
    }

    results = {}
    for band_name, band_range in bands.items():
        results[band_name] = visual_dt_generalization_test(
            model, env, dt_values, cfg,
            n_seqs=n_seqs, seq_len=seq_len,
            energy_radius_range_override=band_range,
            eval_dataset_dir=eval_dataset_dir,
            band_label=band_name if eval_dataset_dir is not None else None,
        )
    return results


@torch.no_grad()
def visual_fixed_init_stratified_test(
    model, env, dt_values, cfg, energy_radius_range,
    n_seqs=8, seq_len=None,
    eval_dataset_dir=None,
):
    """Per-band fixed-init eval.

    For each of three energy bands (low / med / high), sample ONE init
    state from the band's radius sub-range and run n_seqs trajectories
    from that fixed init with only action sequences varying. variable_params
    are already implicitly fixed via env construction (env.step / env.sample_initial_state
    use the env's instance attributes when no per-call params are supplied).

    Args:
        model:               VisualWorldModel.
        env:                 PhysicsControlEnv with sample_initial_state() and
                             _sample_energy_radius_state() implemented.
        dt_values:           list of dt values to test (per band).
        cfg:                 Hydra config (forwarded to dt-gen test).
        energy_radius_range: (r_min, r_max) — full eval distribution to split
                             into 3 equal sub-ranges for the per-band sampling.
        n_seqs:              n_rollouts per band.
        seq_len:             optional seq_len override.

    Returns:
        dict {band: {"init_state": <state tensor>, "results": dt_results_dict}}
        where dt_results_dict has the same schema as visual_dt_generalization_test's
        return value (one entry per dt). init_state is the band-representative
        state used for all rollouts in that band.
    """
    import numpy as np
    r_min, r_max = float(energy_radius_range[0]), float(energy_radius_range[1])
    edges = np.linspace(r_min, r_max, 4)
    bands = {
        "low":  (float(edges[0]), float(edges[1])),
        "med":  (float(edges[1]), float(edges[2])),
        "high": (float(edges[2]), float(edges[3])),
    }

    # When eval_dataset_dir is set, take the canonical "fixed init" as the
    # first sequence's init_state from metadata.json. This makes the fixed-
    # init eval paired across evals (same canonical init per band).
    canonical_inits = {}
    if eval_dataset_dir is not None:
        from src.eval.eval_dataset_io import load_metadata
        md = load_metadata(eval_dataset_dir)
        for band_name in bands.keys():
            anchor = md["anchors"][band_name]["0"]  # JSON keys are strings
            canonical_inits[band_name] = torch.as_tensor(
                anchor["init_state"]
            ).float()

    results = {}
    for band_name, band_range in bands.items():
        if eval_dataset_dir is not None:
            init_state = canonical_inits[band_name]
        else:
            # Sample ONE init state from this band's sub-range.
            init_state = env.sample_initial_state(
                sampling_mode="energy_radius",
                init_state_range=None,
                energy_radius_range=list(band_range),
                variable_params=None,
            )
        # Run n_seqs trajectories from that fixed init (only actions vary).
        band_results = visual_dt_generalization_test(
            model, env, dt_values, cfg,
            n_seqs=n_seqs, seq_len=seq_len,
            fixed_init_state=init_state,
            eval_dataset_dir=eval_dataset_dir,
            band_label=band_name if eval_dataset_dir is not None else None,
        )
        results[band_name] = {
            "init_state": init_state,
            "results": band_results,
        }
    return results
