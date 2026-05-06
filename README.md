
# Reproducibility Guide

## 🛠️ Environment & Dependencies
The code has been verified and tested on the following setup:
- **GPU:** NVIDIA GeForce RTX 4090
- **Deep Learning Framework:** PyTorch 2.7.0

## 📂 Data Preparation
Our method relies on the **LSMI (Large Scale Multi-Illuminant)** dataset. To reproduce the training data, please refer to the official LSMI repository and follow **Steps 0 through 2** of their data generation pipeline.

**Required Files:**
Ensure the following files are generated:
- **Color-biased (Tinted) Images:** `.tiff` format
- **Ground Truth (GT) Images:** `.tiff` format
- **GT Illuminant Maps:** `.npy` format
- **Illuminant Mixture Weights:** `.npy` format

**PreTrained Model:**

The pre-trained models will be made publicly available upon paper acceptance.