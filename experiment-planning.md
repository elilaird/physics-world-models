# Temporal Resolution Generalization in Learned Dynamics Models

## Experiment Overview

This project investigates how the temporal resolution (sampling rate) at which a dynamics model is trained affects its ability to generalize to different temporal resolutions at test time. This is a fundamental question for world models, model-based RL, and video prediction systems.

### Research Questions

1. **Does training resolution affect deployment performance?** If we train a dynamics predictor at 20 Hz, can it accurately predict at 10 Hz or 50 Hz?

2. **How do learned dynamics depend on Δt?** What implicit assumptions about velocities and temporal granularity does the model learn?

3. **What approaches enable temporal generalization?**
   - Δt conditioning (explicit temporal input)
   - Multi-resolution training (train on mixed frequencies)
   - Recurrent rollouts (chain predictions)
   - Neural ODEs (continuous-time modeling)

4. **Which environments are most sensitive to temporal mismatch?** Do contact-rich tasks suffer more than smooth dynamics?

### Hypothesis

**Null hypothesis**: Dynamics models trained at one temporal resolution will generalize poorly to other resolutions without explicit architectural support for temporal abstraction.

**Alternative hypothesis**: Approaches like Δt conditioning or multi-resolution training will enable robust generalization across temporal scales.

---

## Experimental Design

### Phase 1: Data Collection

**Environments**:
- `Pendulum-v1` (continuous control, smooth dynamics)
- `CartPole-v1` (unstable equilibrium, requires fast control)

**Control Frequencies**:
- 10 Hz (coarse, 0.1s between actions)
- 20 Hz (medium, 0.05s between actions)  
- 50 Hz (fine, 0.02s between actions)

**Collection Strategy**:
- Train separate PPO agents at each control frequency
- Collect data during training (ensures high-reward state coverage)
- 500k timesteps per frequency per environment
- 4 parallel environments for efficiency
- Episode-based train/val/test splits (70/15/15)

**Why this approach?**
- PPO naturally explores and improves, providing diverse + task-relevant data
- Separate agents per frequency avoids conflating policy optimization with temporal effects
- Training data includes both exploratory and high-performing trajectories

### Phase 2: Dynamics Model Training

**Model Architectures** (see `models.py`):

All models use discrete action embeddings via `nn.Embedding(action_dim, embedding_dim)` and share `hidden_dim=64`, `action_embedding_dim=8` defaults.

1. **JumpModel** (Discrete residual MLP):
   - Learns `x_{t+1} = x_t + f(x_t, a_t)` as a discrete map
   - Forward: `(state, action) → next_state`

2. **LSTMModel** (Recurrent baseline):
   - LSTM with residual connection for sequential prediction
   - Forward: `(state, action) → next_state`

3. **NewtonianDynamicsModel** (2nd-order ODE with physics prior):
   - Separates position/velocity: `d/dt[x, v] = [v, f(x, v, a)]`
   - Learns acceleration with learned damping (`log_damping` parameter)
   - Forward: `(t, state, action_emb)` — compatible with `torchdiffeq.odeint`

4. **VelocityDynamicsModel** (1st-order ODE):
   - Learns velocity directly: `d/dt[x] = NN(x, a)`, zeros for dvdt
   - Shares ODE integrator interface with NewtonianDynamicsModel
   - Forward: `(t, state, action_emb)`

5. **PortHamiltonianModel** (Energy-based, symplectic structure):
   - Learns Hamiltonian `H(q, p)` via neural net with Softplus activations
   - Computes symplectic dynamics via `torch.autograd.grad` of H
   - Includes dissipation (learned damping) and input port `G(u)` for external forces
   - Forward: `(t, state, action_emb)`

Note: Models 1–2 are discrete-time (single-step prediction). Models 3–5 are continuous-time ODE functions integrated via `torchdiffeq.odeint` with dopri5/rk4 solvers.

**Training Variants**:

1. **Baseline (No Δt conditioning)**
   - Input: `[state, action]`
   - Output: `next_state`
   - Train separately on each frequency's data
   - Test on all frequencies

2. **Δt Conditioning**
   - Input: `[state, action, delta_t]`
   - Output: `next_state`
   - Train on single frequency but with Δt as input
   - Test on all frequencies (Δt varies)

3. **Multi-Resolution Training**
   - Input: `[state, action, delta_t]`
   - Train on mixed data from all frequencies
   - Test on all frequencies

4. **Recurrent Rollout Baseline**
   - Use baseline model trained at finest resolution (50 Hz)
   - For coarser predictions: autoregressively apply multiple times
   - E.g., for 10 Hz: apply 50 Hz model 5 times in sequence

5. **ODE Integration** (Models 3–5 only)
   - Naturally handle variable Δt by changing the integration interval
   - No retraining needed — just integrate from `t=0` to `t=Δt`
   - Test whether continuous-time structure gives inherent temporal generalization

**Training Configuration**:
- Optimizer: Adam
- Learning rate: 3e-4 with cosine decay
- Batch size: 256
- Epochs: 100 (early stopping on validation loss)
- Loss: MSE between predicted and actual next state
- Normalization: Standardize states and actions using training statistics

**Model Sizes to Test**:
- Small: [256, 256] hidden layers
- Medium: [512, 512, 512] hidden layers
- Large: [1024, 1024, 1024, 1024] hidden layers

### Phase 3: Evaluation

**Metrics**:

1. **One-step prediction error**:
   - MSE between predicted and actual next state
   - Per-dimension MSE (to identify which state components fail)
   - Test at each frequency: 10, 20, 50 Hz

2. **Multi-step rollout error**:
   - Open-loop prediction for 1 second (10, 20, or 50 steps depending on frequency)
   - Track error accumulation over time
   - Visualize trajectories

3. **Control performance**:
   - Use learned dynamics model for model-predictive control (MPC)
   - Planning horizon: 1 second
   - Evaluate episode return on actual environment
   - Test at each control frequency

4. **Dynamics consistency**:
   - Given same initial state, do predictions at different Δt converge?
   - E.g., does 5 steps at 50 Hz ≈ 1 step at 10 Hz?

**Test Protocol**:
```
For each (environment, train_freq, test_freq, method) combination:
  1. Load trained model
  2. Load test dataset at test_freq
  3. Compute one-step prediction error
  4. Run open-loop rollouts (50 episodes)
  5. Run closed-loop control (50 episodes)
  6. Save metrics and visualizations
```

**Visualization**:
- Prediction error heatmaps (train_freq × test_freq)
- Trajectory rollouts (predicted vs actual)
- Error vs. rollout length curves
- State-space coverage colored by error

---

## Repository Structure
```
temporal-dynamics-generalization/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (alternative)
├── .gitignore
│
├── configs/                           # Configuration files
│   ├── environments.yaml             # Environment settings
│   ├── collection.yaml               # Data collection hyperparameters
│   ├── training.yaml                 # Model training hyperparameters
│   └── evaluation.yaml               # Evaluation settings
│
├── data/                             # Data storage (gitignored, too large)
│   ├── raw/                          # Raw collected data
│   │   ├── Pendulum_v1/
│   │   │   ├── 10hz/
│   │   │   │   ├── final_dataset_10hz.h5
│   │   │   │   ├── final_dataset_10hz_train.h5
│   │   │   │   ├── final_dataset_10hz_val.h5
│   │   │   │   ├── final_dataset_10hz_test.h5
│   │   │   │   ├── final_model.zip
│   │   │   │   └── tensorboard/
│   │   │   ├── 20hz/
│   │   │   └── 50hz/
│   │   └── CartPole_v1/
│   │       ├── 10hz/
│   │       ├── 20hz/
│   │       └── 50hz/
│   └── processed/                    # Normalized/preprocessed data
│       └── normalization_stats.pkl
│
├── models/                           # Trained models (gitignored)
│   ├── pendulum/
│   │   ├── baseline_10hz/
│   │   │   ├── model.pt
│   │   │   ├── config.yaml
│   │   │   └── training_curves.png
│   │   ├── baseline_20hz/
│   │   ├── baseline_50hz/
│   │   ├── dt_conditioned_20hz/
│   │   ├── multi_resolution/
│   │   └── recurrent_50hz/
│   └── cartpole/
│       └── ...
│
├── results/                          # Experiment results
│   ├── metrics/                      # Numerical results
│   │   ├── one_step_errors.csv
│   │   ├── rollout_errors.csv
│   │   ├── control_performance.csv
│   │   └── summary_statistics.json
│   ├── figures/                      # Plots and visualizations
│   │   ├── error_heatmaps/
│   │   ├── trajectory_rollouts/
│   │   ├── error_vs_horizon/
│   │   └── paper_figures/
│   └── videos/                       # Rollout videos
│       ├── pendulum_baseline_10hz_on_50hz.mp4
│       └── ...
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_paper_figures.ipynb
│
├── scripts/                          # Executable scripts
│   ├── collect_data.py              # Data collection entry point
│   ├── train_models.py              # Model training entry point
│   ├── evaluate_models.py           # Evaluation entry point
│   ├── run_full_pipeline.sh         # Run entire experiment
│   └── generate_paper_figures.py    # Create publication-ready figures
│
├── src/                              # Source code modules
│   ├── __init__.py
│   │
│   ├── data/                         # Data handling
│   │   ├── __init__.py
│   │   ├── collection.py            # PPO training + data collection
│   │   ├── preprocessing.py         # Normalization, augmentation
│   │   ├── dataset.py               # PyTorch Dataset classes
│   │   └── loaders.py               # DataLoader utilities
│   │
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── dynamics_mlp.py          # MLP dynamics model
│   │   ├── dt_conditioned.py        # Δt-conditioned variant
│   │   ├── neural_ode.py            # Neural ODE implementation
│   │   └── ensemble.py              # Ensemble dynamics model
│   │
│   ├── training/                     # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py               # Main training loop
│   │   ├── losses.py                # Loss functions
│   │   └── callbacks.py             # Training callbacks
│   │
│   ├── evaluation/                   # Evaluation logic
│   │   ├── __init__.py
│   │   ├── metrics.py               # Metric computation
│   │   ├── rollouts.py              # Open-loop rollout evaluation
│   │   ├── control.py               # MPC-based control evaluation
│   │   └── visualization.py         # Plotting utilities
│   │
│   ├── environments/                 # Environment wrappers
│   │   ├── __init__.py
│   │   ├── frequency_wrapper.py     # Fixed frequency wrapper
│   │   └── utils.py                 # Environment utilities
│   │
│   └── utils/                        # General utilities
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── logging.py               # Logging setup
│       ├── seed.py                  # Random seed management
│       └── paths.py                 # Path management
│
├── tests/                            # Unit tests
│   ├── test_data/
│   ├── test_models/
│   ├── test_training/
│   └── test_evaluation/
│
└── docs/                             # Additional documentation
    ├── data_format.md               # Dataset format specification
    ├── model_architecture.md        # Model architecture details
    ├── experimental_protocol.md     # Detailed protocol
    └── results_interpretation.md    # How to interpret results
```

---

## Implementation Phases

### Phase 0: Setup (Day 1)

**Goal**: Set up repository structure and environment

**Tasks**:
1. Create repository structure
2. Set up Python environment:
```bash
   conda create -n temporal-dynamics python=3.10
   conda activate temporal-dynamics
   pip install -r requirements.txt
```
3. Write configuration files
4. Implement basic utilities (logging, config loading, path management)
5. Write unit tests for utilities

**Dependencies** (requirements.txt):
```
# Core
numpy>=1.24.0
torch>=2.0.0
gymnasium>=0.28.0
stable-baselines3>=2.0.0

# Data handling
h5py>=3.8.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Experiment tracking
tensorboard>=2.12.0
wandb>=0.15.0  # optional

# Testing
pytest>=7.3.0
pytest-cov>=4.0.0

# Optional: Neural ODE
torchdiffeq>=0.2.3

# Optional: dm_control for CartPole
dm-control>=1.0.0
```

**Deliverable**: Runnable repository skeleton with passing tests

---

### Phase 1: Data Collection (Days 2-3)

**Goal**: Collect datasets at multiple control frequencies

**Tasks**:
1. Implement `FixedFrequencyWrapper` for environments
2. Implement `MultiFrequencyDataCollector` callback
3. Implement data collection pipeline
4. Run collection for Pendulum at 10, 20, 50 Hz
5. Run collection for CartPole at 10, 20, 50 Hz
6. Create train/val/test splits
7. Verify data quality:
   - Check episode returns improve during PPO training
   - Visualize state coverage
   - Verify no NaNs or outliers

**Commands**:
```bash
# Collect all data
python scripts/collect_data.py --config configs/collection.yaml

# Verify data
python scripts/verify_data.py --data_dir data/raw/
```

**Deliverable**: 
- 6 datasets (2 envs × 3 frequencies)
- ~500k transitions per dataset
- Train/val/test splits
- Data quality report

**Time estimate**: 
- Pendulum: 2-3 hours per frequency (6-9 hours total)
- CartPole: 2-3 hours per frequency (6-9 hours total)
- Can parallelize across frequencies

---

### Phase 2: Model Implementation (Day 4)

**Goal**: Implement all dynamics model variants

**Tasks**:
1. Implement base `DynamicsModel` class
2. Implement MLP dynamics model (baseline)
3. Implement Δt-conditioned model
4. Implement multi-resolution training support
5. Implement data preprocessing (normalization)
6. Write unit tests for models
7. Implement training loop with validation

**Key Components**:

**Base Model Interface**:
```python
class DynamicsModel(nn.Module):
    def forward(self, state, action, delta_t=None):
        """Predict next state"""
        raise NotImplementedError
    
    def predict_sequence(self, state, actions, delta_t=None):
        """Multi-step rollout"""
        raise NotImplementedError
```

**Training Script Interface**:
```bash
python scripts/train_models.py \
    --env pendulum \
    --train_freq 20 \
    --model_type baseline \
    --config configs/training.yaml
```

**Deliverable**: Tested model implementations ready for training

---

### Phase 3: Model Training (Days 5-6)

**Goal**: Train all model variants on all datasets

**Training Matrix**:
For each environment (Pendulum, CartPole):

1. **Baseline models** (6 models):
   - Train on 10 Hz data → `baseline_10hz`
   - Train on 20 Hz data → `baseline_20hz`
   - Train on 50 Hz data → `baseline_50hz`
   - (Repeat for 3 model sizes = 9 models if testing size)

2. **Δt-conditioned models** (3 models):
   - Train on 20 Hz data with Δt input → `dt_conditioned_20hz`

3. **Multi-resolution models** (1 model):
   - Train on mixed data from all frequencies → `multi_resolution`

4. **Recurrent baseline** (1 model):
   - Use `baseline_50hz` with autoregressive rollouts

**Total models**: (6 + 3 + 1) × 2 environments = 20 models

**Tasks**:
1. Write training orchestration script
2. Train all models with early stopping
3. Monitor training curves
4. Save best models
5. Generate training summaries

**Commands**:
```bash
# Train all models for one environment
python scripts/train_models.py --env pendulum --all-variants

# Or train individually
python scripts/train_models.py --env pendulum --train_freq 20 --model baseline
python scripts/train_models.py --env pendulum --train_freq 20 --model dt_conditioned
python scripts/train_models.py --env pendulum --model multi_resolution
```

**Deliverable**: 
- 20 trained models
- Training curves and logs
- Model checkpoints

**Time estimate**: 
- ~1 hour per model
- ~20 hours total (can parallelize)

---

### Phase 4: Evaluation (Days 7-8)

**Goal**: Comprehensive evaluation of all models

**Evaluation Matrix**:
For each (environment, model, test_frequency) combination:
- 2 environments × 10 model variants × 3 test frequencies = 60 evaluations

**Tasks**:

1. **One-step prediction error**:
   - Load test dataset
   - Compute MSE per state dimension
   - Aggregate statistics

2. **Multi-step rollouts**:
   - 50 rollout episodes per configuration
   - Track error vs. horizon
   - Save trajectory visualizations

3. **Control performance**:
   - Implement simple MPC controller
   - 50 control episodes per configuration
   - Record returns and success rates

4. **Analysis**:
   - Generate error heatmaps (train_freq × test_freq)
   - Statistical significance tests
   - Identify failure modes

**Commands**:
```bash
# Evaluate all models
python scripts/evaluate_models.py \
    --env pendulum \
    --all-models \
    --all-test-freqs \
    --config configs/evaluation.yaml

# Generate summary report
python scripts/summarize_results.py \
    --results_dir results/ \
    --output results/summary_report.pdf
```

**Deliverable**:
- Comprehensive metrics CSV files
- Error heatmaps
- Trajectory rollout videos
- Statistical analysis report

**Time estimate**: 
- ~2 hours for all evaluations
- ~4 hours for analysis and visualization

---

### Phase 5: Analysis & Visualization (Days 9-10)

**Goal**: Create publication-ready figures and insights

**Tasks**:

1. **Main figures**:
   - Figure 1: Error heatmaps (train_freq × test_freq) for each method
   - Figure 2: Rollout error vs. horizon for key comparisons
   - Figure 3: Control performance comparison
   - Figure 4: State-space visualization with error coloring
   - Figure 5: Ablation studies (model size, training data amount)

2. **Analysis notebooks**:
   - Data exploration and statistics
   - Training dynamics analysis
   - Comprehensive results visualization
   - Failure mode analysis

3. **Documentation**:
   - Update README with key findings
   - Write results interpretation guide
   - Document insights and recommendations

**Commands**:
```bash
# Generate all paper figures
python scripts/generate_paper_figures.py \
    --results_dir results/ \
    --output results/figures/paper_figures/

# Launch analysis notebook
jupyter notebook notebooks/03_results_visualization.ipynb
```

**Deliverable**:
- Publication-ready figures
- Comprehensive analysis notebooks
- Written summary of findings
- Recommendations for practitioners

---

## Key Experimental Questions & Expected Outcomes

### Question 1: Do baseline models generalize across frequencies?

**Expected**: No. Baseline models will show significant performance degradation when test frequency differs from train frequency.

**Evidence to look for**:
- High MSE in off-diagonal cells of error heatmap
- Rapid error accumulation in rollouts at mismatched frequencies
- Poor control performance at non-training frequencies

### Question 2: Does Δt conditioning help?

**Expected**: Yes. Δt-conditioned models should show much better generalization.

**Evidence to look for**:
- More uniform error heatmap (lower off-diagonal errors)
- Stable rollouts at multiple frequencies
- Graceful interpolation between training frequencies

### Question 3: Is multi-resolution training better than Δt conditioning?

**Expected**: Slightly better, but with diminishing returns.

**Evidence to look for**:
- Small improvement over Δt conditioning
- Best performance at extreme frequencies (5 Hz, 100 Hz)
- More robust to distribution shift

### Question 4: Which environments are most sensitive?

**Expected**: CartPole more sensitive than Pendulum due to instability.

**Evidence to look for**:
- Larger performance gaps for CartPole
- CartPole baseline models completely fail at mismatched frequencies
- Pendulum shows graceful degradation

### Question 5: Does model size matter?

**Expected**: Larger models generalize better with Δt conditioning.

**Evidence to look for**:
- Baseline: size doesn't help much
- Δt-conditioned: larger models show better interpolation
- Multi-resolution: benefits most from capacity

---

## Success Criteria

**Minimum viable results**:
1. ✅ Baseline models show clear performance drop at non-training frequencies
2. ✅ Δt conditioning provides measurable improvement
3. ✅ Multi-resolution training matches or exceeds Δt conditioning
4. ✅ Clear documentation and reproducible code

**Strong results**:
5. ✅ Quantified relationship between frequency mismatch and error
6. ✅ Identified failure modes and when each approach works
7. ✅ Practical recommendations for real-world systems
8. ✅ Publication-quality figures and writing

**Stretch goals**:
9. ⭐ Neural ODE comparison
10. ⭐ Real robot experiments (if available)
11. ⭐ Theoretical analysis of learned velocity fields
12. ⭐ Extension to vision-based environments

---

## Timeline Summary

**Total: 10 days**

- Day 1: Setup and infrastructure
- Days 2-3: Data collection (can run overnight)
- Day 4: Model implementation
- Days 5-6: Model training (can parallelize)
- Days 7-8: Evaluation
- Days 9-10: Analysis and visualization

**Parallelization opportunities**:
- Data collection for different environments
- Model training (use multiple GPUs)
- Evaluation across models

**Checkpoints**:
- End of Day 3: All data collected and verified
- End of Day 6: All models trained
- End of Day 8: All evaluations complete
- End of Day 10: Paper-ready results

---

## Notes for Implementation

### Data Collection
- Use `tensorboard` to monitor PPO training
- Save checkpoints every 100 episodes for debugging
- Verify episode returns improve (sanity check)
- Check for NaNs in collected data

### Model Training
- Use early stopping (patience=10 epochs on validation loss)
- Save training curves for debugging
- Monitor gradient norms to detect instabilities
- Use learning rate warmup for stability

### Evaluation
- Set random seeds for reproducibility
- Run multiple evaluation seeds (50 episodes minimum)
- Save full trajectories for failure analysis
- Record videos of interesting cases

### Debugging Tips
- Start with Pendulum (simpler than CartPole)
- Use small dataset subset for quick iterations
- Visualize predictions vs. ground truth early
- Check normalization statistics are reasonable
- Verify Δt values are correctly passed to models

---

## Questions to Address During Experiment

1. What is the functional form of error vs. frequency mismatch?
2. Can we predict when temporal generalization will fail?
3. How much training data is needed for good generalization?
4. Does the optimal Δt conditioning strategy depend on the task?
5. Are there universal features of good temporal abstraction?

---

## Extensions for Future Work

1. **More environments**: MuJoCo humanoid, manipulation tasks
2. **Visual observations**: Atari, dm_control visual tasks
3. **Hierarchical models**: Different Δt for different state components
4. **Adaptive frequency**: Learn to choose optimal control frequency
5. **Continuous-time models**: Neural ODEs, latent ODEs
6. **Real-world validation**: Deploy on physical robot

---

## Contact & Contribution

For questions or contributions, please open an issue or pull request.

**Citation**: If you use this code, please cite:
```bibtex
@article{temporal-dynamics-2024,
  title={Temporal Resolution Generalization in Learned Dynamics Models},
  author={Your Name},
  year={2024}
}
```
