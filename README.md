# GI-Cancer-MSI-CCNet-MIL
This repository contains the code and resources for our paper "". The project focuses on enhancing microsatellite instability (MSI) prediction in Whole Slide Image (WSI) analysis of gastrointestinal cancers through a novel two-stage weakly supervised methodology.

## Overview
We propose a framework that integrates Multiple Instance Learning (MIL) with a unique Contrastive Clustering Network (CCNet) for feature extraction. The method leverages the synergy of these approaches to significantly improve MSI classification accuracy, surpassing existing methods in the field.

![Figure 1](https://github.com/napronald/GI-Cancer-MSI-CCNet-MIL/blob/main/Figures/Figure1.png)

## Key Contributions
- **Innovative Two-Stage Model:**: Our research introduces a new approach to analyzing and classifying histopathology images, particularly for microsatellite instability (MSI) in gastrointestinal cancers.
- **Contrastive Clustering-Based Feature Extraction**: We integrate a contrastive clustering-based feature extractor, leading to significant improvements in performance compared to previous network designs. 
- **Enhanced Prediction in MSI Classification**: Notable improvement in MSI classification, demonstrated through experiments using colorectal cancer and stomach adenocarcinoma datasets.

## Datasets
The model was trained and evaluated on the Colorectal Cancer (CRC) and Stomach Adenocarcinoma (STAD) datasets, with the performance assessed using AUROC and F1 Score metrics.
  
| Dataset | Folder | \# of WSIs (Train) | \# of WSIs (Test) | \# of Patches (Train) | \# of Patches (Test) |
|---------|--------|------------------|-----------------|--------------------|-------------------|
| CRC     | MSI    | 39               | 26              | 46,704             | 29,335             |
| CRC     | MSS    | 221              | 74              | 46,704             | 70,569             |
| STAD    | MSI    | 35               | 25              | 50,285             | 27,904             |
| STAD    | MSS    | 150              | 74              | 50,285             | 90,104             |

The datasets can be accessed [here](https://zenodo.org/records/2530835).

# Usage
The code is split into two folders: Section 1 focuses on Feature Extractor Training and Feature Extraction. Section 2 deals with Multiple Instance Learning Classifiers and constructing bags using the feature vectors.

## Requirements
- 
- 
- 
- 
- 

## Code Structure
- `data/`: Data loading and preprocessing scripts.
- `models/`: Implementation of CCNet and MIL classifiers.
- `training/`: Scripts for training and evaluating the model.
- `utils/`: Utility functions and helpers.

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{nap2024,
  title={},
  author={Nap, Ronald and Aburidi, Mohammed and Marcia, Roummel},
  journal={EMBC},
  year={2024}
}
```
