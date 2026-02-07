from setuptools import setup, find_packages

setup(
    name="imu-world-modeling",
    version="0.1.0",
    description="Lightweight world models for IMU-only navigation",
    author="TERN.AI",
    python_requires=">=3.10",
    packages=find_packages(where="src", include=["*"]),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "h5py>=3.8.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "ahrs>=0.3.0",
    ],
    extras_require={
        "mobile": [
            "coremltools>=7.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "wandb>=0.15.0",
            "seaborn>=0.12.0",
        ],
    },
)
