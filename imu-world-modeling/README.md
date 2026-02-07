# Lightweight World Modeling for IMU-Only Navigation

Lightweight deep learning-based world models for robust inertial navigation using only IMU data, optimized for deployment on resource-constrained mobile devices.

## Overview

This project uses the KAIST Urban Dataset's high-quality IMU stream to develop compact learned approaches that outperform traditional integration methods by learning motion priors, bias compensation, and leveraging temporal context.

**Target Application:** On-device pedestrian and vehicle navigation during GPS outages, optimized for real-time inference on Apple Neural Engine (ANE) or similar mobile accelerators.

## Motivation

Pure IMU integration suffers from rapid drift due to sensor bias/noise accumulation, double integration errors, and lack of external corrections. Lightweight world modeling addresses these by:

- Learning to compensate for IMU bias and noise patterns
- Incorporating motion priors (vehicles follow roads, pedestrians walk)
- Leveraging temporal patterns to predict and correct drift
- Being small enough for real-time mobile inference

## Project Structure

```
├── src/
│   ├── data/           # Data loading, preprocessing, augmentation
│   ├── models/         # Lightweight model architectures
│   ├── baselines/      # Traditional integration baselines
│   ├── training/       # Training loop, losses, metrics
│   ├── mobile/         # CoreML/TFLite export and benchmarking
│   └── utils/          # Geometry, IMU integration, config
├── experiments/        # Experiment scripts
├── configs/            # YAML configuration files
├── notebooks/          # Analysis notebooks
├── mobile_app/         # iOS/Android demo apps
└── results/            # Output figures, trajectories, metrics
```

## Installation

```bash
conda create -n imu-nav python=3.10
conda activate imu-nav
pip install -r requirements.txt
pip install -e .

# iOS deployment (macOS only)
pip install coremltools

# Android deployment
pip install tensorflow tf2onnx
```

## Dataset Setup

1. Download KAIST Urban Dataset IMU data and ground truth from
   https://sites.google.com/view/complex-urban-dataset

2. Extract and preprocess:
```bash
python scripts/extract_imu.py --input data/kaist/raw/ --output data/kaist/imu/
python scripts/preprocess_imu.py --input data/kaist/imu/ --output data/processed/ --window-size 5.0 --stride 0.5
python scripts/create_splits.py --config configs/dataset_config.yaml
```

## Architecture Variants

| Variant | Parameters | Size (FP32) | Size (INT8) | Target |
|---------|-----------|-------------|-------------|--------|
| Tiny TCN | ~50K | ~200 KB | ~50 KB | <1 MB |
| Small MobileNet-1D | ~200K | ~800 KB | ~200 KB | <5 MB |
| Compact Transformer | ~100K | ~400 KB | ~100 KB | <3 MB |

## Quick Start

```bash
# Train lightweight model
python experiments/train_lightweight.py --config configs/model_tiny.yaml

# Run baseline comparison
python experiments/baseline_integration.py --sequence urban01

# Export for iOS
python src/mobile/export_coreml.py --model checkpoints/best.ckpt --output mobile_app/ios/model.mlmodel
```

## Mobile Deployment Constraints

- **Model Size:** <5 MB (ideally <2 MB)
- **Inference Time:** <10 ms per prediction at 100 Hz
- **Memory:** <100 MB RAM usage
- **Precision:** INT8 quantization or FP16 mixed precision
- **Target:** Apple Neural Engine (A14+), iPhone 12+

## References

- IONet (Chen et al., 2018), RIDI (Yan et al., 2018), TLIO (Liu et al., 2020)
- MobileNets (Howard et al., 2017), EfficientNet (Tan & Le, 2019)
- Madgwick Filter, ZUPT, Preintegration (Forster et al., 2017)

## License

MIT
