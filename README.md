# NFR: Neural Feature-Guided Non-Rigid Shape Registration

Official implementation of **"NFR: Neural Feature-Guided Non-Rigid Shape Registration"**. [[arXiv]](https://arxiv.org/abs/2505.22445)

## Introduction
We propose a novel learning-based framework for 3D shape registration, which overcomes the challenges of significant non-rigid deformation and partiality undergoing among input shapes, and, remarkably, requires no correspondence annotation during training. Our key insight is to incorporate neural features learned by deep learning-based shape matching networks into an iterative, geometric shape registration pipeline. The advantage of our approach is two-fold -- On one hand, neural features provide more accurate and semantically meaningful correspondence estimation than spatial features (e.g., coordinates), which is critical in the presence of large non-rigid deformations; On the other hand, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching and partial shape matching across varying settings, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic deformations, in which case neither traditional registration methods nor intrinsic methods work.


<!-- NFR is a learning-based framework for 3D non-rigid shape registration that combines deep functional maps with iterative geometric optimization. It handles both **full-to-full** and **partial-to-full** shape matching **without correspondence supervision**. -->

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Registration](#registration)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Overview

Our pipeline consists of two main components:

1. **Feature Extractor Training** (`train/`): A DGCNN-based point feature extractor trained via a teacher-student paradigm with deep functional maps, supporting both full and partial shapes.
2. **Shape Registration** (`registration/`): An iterative geometric optimization that uses learned neural features to guide non-rigid deformation, with dynamic correspondence updating and bijectivity-based filtering.

### Key Features

- No correspondence annotation required during training
- Handles significant non-rigid deformations and heterogeneous shapes
- Supports both full-to-full (DFR) and partial-to-full (Partial-DFR) matching
- Automatic orientation alignment via a data-driven regressor
- Two-stage registration: feature-guided + coordinate-guided

## Repository Structure

```
NFR_release/
├── train/                          # Feature extractor training
│   ├── train_partial_dfr.py        # Train Partial-DFR feature extractor
│   ├── train_full_dfr.py           # Train full DFR feature extractor
│   ├── test_partial_dfr.py         # Evaluate partial feature extractor
│   ├── test_full_dfr.py            # Evaluate full feature extractor
│   ├── dataset_partial.py          # Dataset for partial training
│   ├── dataset_partial_fpcross.py  # Dataset for full training with cross features
│   ├── partial.py                  # Partial shape generation (ray-casting)
│   ├── utils.py                    # Loss functions and utilities
│   ├── model.py                    # Regularized FMNet
│   ├── lgattention.py              # LG attention module
│   ├── AverageMeter.py             # Training metric tracker
│   ├── cal_ico.py                  # Icosahedron viewpoint computation
│   ├── models/                     # Network architectures
│   │   ├── dgcnn_sample.py         # Modified DGCNN (main backbone)
│   │   ├── dgcnn.py                # Standard DGCNN
│   │   ├── attention_net.py        # Cross-attention refinement
│   │   ├── pointnet.py             # PointNet basis
│   │   └── DPFM.py                 # Deep partial functional maps
│   ├── diffusion_net/              # DiffusionNet (for spectral computation)
│   └── config/                     # Training configurations
│       ├── train_partial_sf.yaml   # Partial-DFR on S&F dataset (recommended)
│       └── train_full_sf.yaml      # Full DFR on S&F dataset
│
├── registration/                   # Shape registration (inference)
│   ├── test_full_register.py       # Full-to-full registration
│   ├── test_partial_register.py    # Partial-to-full registration
│   ├── test_direct_correspondence.py  # Direct correspondence evaluation
│   ├── test_realscan.py            # Real scan registration
│   ├── dataset.py                  # Test dataset loader
│   ├── dataset_scan_pcd_partial.py # Partial scan dataset
│   ├── dataset_preprocess.py       # Data preprocessing
│   ├── loss.py                     # Chamfer distance, ARAP loss
│   ├── tools.py                    # Rotation utilities
│   ├── utils.py                    # Geodesic error, FPS, etc.
│   ├── cal_ico.py                  # Icosahedron computation
│   ├── cal_geo.py                  # Geodesic distance computation
│   ├── geometry_util.py            # Geometry utilities
│   ├── models/                     # Same architectures as train/
│   ├── lib/                        # Registration core
│   │   ├── deformation_graph.py    # Deformation graph optimization
│   │   ├── mesh_sampling.py        # Mesh simplification
│   │   └── utils.py                # Mesh I/O utilities
│   ├── diffusion_net/              # DiffusionNet
│   └── config/                     # Registration configurations
│       ├── full_register.yaml      # Full-to-full config
│       └── partial_register.yaml   # Partial-to-full config
│
├── scripts/                        # Convenience shell scripts
│   ├── train_partial.sh            # Train Partial-DFR
│   ├── train_full.sh               # Train Full DFR
│   ├── register_full.sh            # Full-to-full registration
│   ├── register_partial.sh         # Partial-to-full registration
│   └── preprocess.sh               # Data preprocessing
│
├── pretrained/                     # Pretrained model weights
├── data/                           # Datasets (download separately)
├── results/                        # Output directory (auto-created)
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Python >= 3.8
- CUDA >= 11.3
- PyTorch >= 1.12.0

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/NFR.git
cd NFR

# Create conda environment
conda create -n nfr python=3.8 -y
conda activate nfr

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch=1.12.0 torchvision cudatoolkit=11.3 -c pytorch -y

# Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install pytorch3d -c pytorch3d -y

# Install MPI Mesh library (required for registration)
pip install git+https://github.com/MPI-IS/mesh.git

# Install other dependencies
pip install -r requirements.txt
```

> **Note**: If you have trouble installing `pytorch3d` via conda, refer to the [official installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
>
> **Note**: The `psbody-mesh` library requires Boost. On Ubuntu: `sudo apt-get install libboost-dev`. See the [MPI Mesh repository](https://github.com/MPI-IS/mesh) for details.

## Data Preparation

### Directory Layout

Each dataset should be organized under `data/` in the following structure:

```
data/
├── DATASET_NAME/
│   ├── shapes_train/          # Training meshes (.off format)
│   │   ├── shape_000.off
│   │   ├── shape_001.off
│   │   └── ...
│   ├── shapes_test/           # Test meshes (.off format)
│   │   ├── shape_080.off
│   │   └── ...
│   ├── corres/                # [Optional] Ground-truth vertex correspondences
│   │   ├── shape_000.vts
│   │   └── ...
│   ├── Phi/                   # Precomputed LBO eigenvectors (.mat files)
│   │   ├── shape_000.mat
│   │   └── ...
│   └── geodist/               # [Optional] Precomputed geodesic distance matrices
│       ├── shape_000.mat
│       └── ...
```

### Supported Datasets
You can download dataset refer to [this repo](https://github.com/dongliangcao/Self-Supervised-Multimodal-Shape-Matching) or from official repo. To get FAUST_r, SCAPE_r, SHREC19_r, SHREC07-H, DT4D-H, etc..

### Preprocessing

#### 1. Compute LBO eigenvectors

Use the provided preprocessing script:

```bash
cd registration
python dataset_preprocess.py
```

This computes the Laplace-Beltrami eigenvectors and eigenvalues and saves them as `.mat` files in the `Phi/` directory.

#### 2. Generate partial shapes (for Partial-DFR training)

Partial shapes are generated automatically during training via ray-casting from 12 icosahedron viewpoints. No separate preprocessing is needed.

#### 3. Train/Test Split

- **FAUST_r**: shapes 0-79 for training, 80-99 for testing
- **SCAPE_r**: shapes 0-50 for training, 51-70 for testing
- **S&F**: combined FAUST_r (80 train) + SCAPE_r (51 train) = 131 train, 40 test

Place training shapes in `shapes_train/` and test shapes in `shapes_test/`.

## Train Feature Extractor

> **Note**: All scripts assume they are run from their respective subdirectory (`train/` or `registration/`). Config paths (e.g., `../data/`, `../pretrained/`) are relative to the subdirectory. Alternatively, use the convenience scripts in `scripts/` which handle the directory change automatically.

The full-to-full code is based on our prior work [DFR](https://github.com/rqhuang88/DFR).

### [DFR] Train Full DFR Feature Extractor

```bash
cd train
python train_full_dfr.py --config train_full_sf
```

### [Partial-DFR] Train Partial-DFR Feature Extractor

```bash
# Option 1: run directly
cd train
python train_partial_dfr.py --config train_partial_sf

# Option 2: use convenience script (from project root)
bash scripts/train_partial.sh
```

### Evaluate Feature Extractor

```bash
cd train

# Evaluate full feature extractor
python test_full_dfr.py --config train_full_sf

# Evaluate partial feature extractor
python test_partial_dfr.py --config train_partial_sf
```

Checkpoints will be saved in `train/ckpt/{expname}/`.

## Registration

### Full-to-Full Registration

```bash
cd registration
python test_full_register.py --config full_register

# Or from project root:
bash scripts/register_full.sh
```

### Partial-to-Full Registration

```bash
cd registration
python test_partial_register.py --config partial_register

# Or from project root:
bash scripts/register_partial.sh
```

### Direct Correspondence Evaluation (without registration)

```bash
cd registration
python test_direct_correspondence.py --config full_register
```

### Registration Configuration

Key parameters in registration config:

| Parameter | Description |
|-----------|-------------|
| `source_temp_name` | Template mesh dataset name |
| `target_data_set_name` | Target point cloud dataset name |
| `model_path` | Path to trained feature extractor checkpoint |
| `stage1.rmse / cd / arap` | Stage-I weights (feature-guided) |
| `stage2.rmse / cd / arap` | Stage-II weights (coordinate-guided) |

#### Hyper-parameter Presets

| Setting | stage1 (rmse, cd, arap) | stage2 (rmse, cd, arap) |
|---------|------------------------|------------------------|
| Full registration | 1, 0.01, 20 | 0.01, 1, 1 |
| Partial registration | 0.01, 1, 20 | 0.01, 1, 1 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2025nfr,
  title={NFR: Neural Feature-Guided Non-Rigid Shape Registration},
  author={Jiang, Puhua and Zhang, Quan and Sun, Mingze and Huang, Ruqi},
  journal={arXiv preprint arXiv:2505.22445},
  year={2025}
}
```

## Acknowledgement

Parts of this codebase are built upon:
- [DFR](https://github.com/rqhuang88/DFR)
- [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [Deep Functional Maps](https://github.com/LIX-shape-analysis/GeomFmaps)

## License

This project is licensed under the MIT License.
