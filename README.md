# UAV-Based Crack Detection and Quantification using U-Net

A deep learning-based framework for automated crack segmentation and quantification in UAV-acquired bridge deck images using a U-Net architecture and a standalone desktop application.

This repository contains:
- MATLAB implementation of the crack segmentation framework
- Pretrained U-Net model (`trained_unet.mat`)
- Standalone executable application (`.exe`)
- Crack quantification and visualization pipeline

---

# Overview

This repository presents a U-Net-based semantic segmentation framework for automated crack detection and quantification from UAV-acquired bridge deck images.

The framework performs:
- Pixel-wise crack segmentation
- Crack morphology analysis
- Crack severity quantification
- Automated visualization and reporting

A standalone desktop application was also developed to support practical implementation without requiring MATLAB programming experience.

---

# Features

## Crack Segmentation
- U-Net-based semantic segmentation
- Pixel-level crack localization
- Binary crack mask generation
- Post-processing refinement

## Crack Quantification
The framework computes:
- Crack density (%)
- Mean crack width
- Maximum crack width
- Total crack length
- Longest continuous crack
- Number of connected cracks
- Severity classification

## Standalone Application
Developed using MATLAB App Designer and MATLAB Compiler.

Capabilities include:
- Guided workflow
- Image upload
- AOI (Area of Interest) selection
- Crack overlay visualization
- Exportable results
- Fullscreen inspection mode

---

# Model Performance

| Metric | Value |
|---|---|
| IoU (Crack Class) | 74.15% |
| DSC/F1 Score | 85.17% |
| Pixel Accuracy | 95.04% |

## Class-wise Performance

| Class | IoU (%) | DSC/F1 (%) | Pixel Accuracy (%) |
|---|---|---|---|
| Background | 94.22 | 97.03 | 94.39 |
| Crack | 74.15 | 85.17 | 98.98 |

---

# Repository Structure

```text

├── code/
│   ├── main_model.m
│   ├── inference.m
│   └── predict_example.m
│
├── sample_images/
│   ├── sample1.jpg
│   ├── sample2.jpg
│   └── ...
│
├── sample_results/
│   ├── sample1_binary_mask.png
│   ├── sample1_overlay.png
│   ├── sample1_crack_summary.txt
│   └── ...
│
├── sample_app_results/
│   ├── app_overlay_example.png
│   ├── app_statistics_example.png
│   └── ...
│
├── LICENSE
└── README.md
```
## Download Pretrained Model and Standalone Application

Large files are hosted through GitHub Releases.

Download the following files from the latest release:

- `trained_unet.mat`

- `CrackSegmentationApp.exe`
---

# Reproducing Training from Source

## Requirements

### Software
- MATLAB R2025a  
  (earlier R202x releases with Deep Learning Toolbox and Computer Vision Toolbox should also work; tested on R2025a)

### Required MATLAB Toolboxes
- Deep Learning Toolbox
- Computer Vision Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox (recommended for GPU training)

### Hardware
A CUDA-capable NVIDIA GPU is strongly recommended.

The full core U-Net training on the complete 40,000-image dataset takes approximately:

- **12,656 minutes (~211 hours)**

depending on hardware configuration.

---

# Dataset

The training set combines:

## 1. UAV-acquired bridge deck imagery
Collected by the authors during bridge inspections.

> Note: These images are not redistributed in this repository.

## 2. Public dataset
Özgenel, Ç. F. (2019).  
*Concrete Crack Images for Classification.*  
Mendeley Data, V2.

DOI:  
https://doi.org/10.17632/5y9wdsg2zt.2

---

# Dataset Structure

To retrain the model using your own dataset, place:

## Input images
```text
images/
```

## Corresponding binary masks
```text
masks/
```

Requirements:
- Background pixels = 0
- Crack pixels = 1
- Filenames must match between images and masks

---

# Training Pipeline

The training script automatically performs:

- Conversion of masks to strict binary
- Image and mask resizing to 224 × 224
- Grayscale conversion
- Dataset splitting:
  - 88.89% training
  - 11.11% validation
- Data augmentation:
  - Rotation ±15°
  - Horizontal/vertical reflection
  - Scaling 0.9–1.1
  - Translation ±10 px

---

# Network Configuration

The implemented U-Net includes:

## Encoder Dropout Layers
- 0.2
- 0.3
- 0.3

## Class Weights
- Crack = 100
- Background = 1

## Training Settings
- Optimizer: Adam
- Epochs: 100
- Mini-batch size: 8
- L2 regularization: 1e-3
- Learning-rate schedule:
  - Piecewise
  - Drop factor = 0.8 every 50 epochs

The trained model is automatically saved as:

```text
trained_unet.mat
```

---

# Run Training

From the MATLAB Command Window:

```matlab
cd code
main_model
```

---

# Run Inference

Minimal inference example:

```matlab
load('model/trained_unet.mat', 'net');

img = imread('examples/input_sample.jpg');

if size(img, 3) == 3
    img = rgb2gray(img);
end

img = imresize(img, [224 224]);

predictedMask = semanticseg(img, net);

overlay = labeloverlay(img, predictedMask, ...
    'IncludedLabels', "Crack", ...
    'Colormap', 'autumn', ...
    'Transparency', 0.4);

figure;

subplot(1,2,1);
imshow(img);
title('Input');

subplot(1,2,2);
imshow(overlay);
title('Predicted crack overlay');
```

---

# Crack Analysis Methodology

After binary segmentation, the post-processing pipeline computes:

| Metric | Definition |
|---|---|
| Crack density | (crack pixels) / (total AOI pixels) × 100% |
| Mean / max width | Width estimated using skeletonization and Euclidean distance transform |
| Total length | Number of pixels in the morphological skeleton |
| Longest crack | Largest connected component in skeleton |
| Crack count | Number of connected crack components |
| Severity | <2% low, 2–5% moderate, ≥5% severe |

---

# Engineering Units

Reported widths and lengths are initially computed in pixels.

To convert into engineering units:

```text
metric_value × GSD (mm/pixel)
```

where:
- GSD = Ground Sampling Distance

---

# Important Notes

Single-image crack metrics depend on:
- Imaging distance
- Resolution
- Camera framing
- Lighting conditions

Therefore, the computed values should be interpreted as:
- Relative image-level indicators
- Screening/prioritization metrics

rather than absolute structural measurements.

For site-level assessments:
- Aggregate multiple images
- Normalize by deck sub-area
- Maintain consistent imaging conditions

---

# Standalone Application Workflow

1. Launch the application
2. Upload crack image
3. Select AOI (optional)
4. Run crack segmentation
5. Visualize crack masks and overlays
6. Export results and crack statistics

Generated outputs:
- Binary crack mask
- Overlay visualization
- Crack statistics report
- AOI image

---

# Example Results

Add screenshots here:
- Original image
- Binary crack mask
- Crack overlay
- GUI screenshots
- Quantification output

---

# Limitations

- Generalization beyond the represented training conditions has not been exhaustively benchmarked on third-party datasets.
- Quantitative metrics are sensitive to:
  - Camera distance
  - Focal length
  - Resolution
  - Lighting conditions
- Current standalone application release supports:
  - Windows only

Linux/macOS builds are not currently distributed but can be regenerated from the MATLAB App Designer source.

---

# Related Paper

**End-to-End UAV-Enabled Bridge Deck Inspection: From Localization to High-Precision Crack Detection and Quantification**

Accepted for publication in:

**ASCE Journal of Bridge Engineering**

> Note: This repository contains only the crack detection and quantification component of the proposed framework.

---

# Citation

```bibtex
@article{almasi2026crack,
  title={End-to-End UAV-Enabled Bridge Deck Inspection: From Localization to High-Precision Crack Detection and Quantification},
  author={Almasi, Pouya and Premadasa, Roshira and Jauregui, David and Zhang, Qianyun},
  journal={ASCE Journal of Bridge Engineering},
  note={Accepted for publication},
  year={2026}
}
```

---

# Author

Pouya Almasi  
Ph.D. Candidate in Civil Engineering  
New Mexico State University

Research Areas:
- Structural Health Monitoring (SHM)
- UAV-based Infrastructure Inspection
- Deep Learning
- Computer Vision
- Crack Detection and Quantification

---

# License

This project is licensed under the MIT License.

---

# Acknowledgment

The research reported in this work was conducted under a long-term project sponsored by the New Mexico Department of Transportation (NMDOT) Research Bureau. Q. Zhang acknowledges the startup fund from the College of Engineering at New Mexico State University.
