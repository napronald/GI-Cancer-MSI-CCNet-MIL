# GI-Cancer-MSI-CCNet-MIL
This repository contains the code and resources for our paper "". The project focuses on enhancing microsatellite instability (MSI) prediction in Whole Slide Image (WSI) analysis of gastrointestinal cancers through a novel two-stage weakly supervised methodology.

## Overview
We propose a framework that integrates Multiple Instance Learning (MIL) with a unique Contrastive Clustering Network (CCNet) for feature extraction. The method leverages the synergy of these approaches to significantly improve MSI classification accuracy, surpassing existing methods in the field.

![Figure 1](https://github.com/napronald/GI-Cancer-MSI-CCNet-MIL/blob/main/Figures/Figure1.png)

## Key Contributions
- **Innovative Two-Stage Model:**: Our research introduces a new approach to analyzing and classifying histopathology images, particularly for microsatellite instability (MSI) in gastrointestinal cancers.
- **Contrastive Clustering-Based Feature Extraction**: Efficient labeling through MIL, allowing for better handling of complex datasets like WSIs.
- **Enhanced Prediction in MSI Classification**: Notable improvement in MSI classification, demonstrated through experiments using colorectal cancer and stomach adenocarcinoma datasets.

## Datasets
The evaluation was conducted using two image datasets from the TCGA cohort:
- Colorectal Cancer (CRC)
- Stomach Adenocarcinoma (STAD)
  
| Dataset | Folder | \# of WSIs (Train) | \# of WSIs (Test) | \# of Patches (Train) | \# of Patches (Test) |
|---------|--------|------------------|-----------------|--------------------|-------------------|
| CRC     | MSI    | 39               | 26              | 46,704             | 29,335             |
| CRC     | MSS    | 221              | 74              | 46,704             | 70,569             |
| STAD    | MSI    | 35               | 25              | 50,285             | 27,904             |
| STAD    | MSS    | 150              | 74              | 50,285             | 90,104             |

The datasets can be accessed [here](https://zenodo.org/records/2530835).

# Usage

## Requirements
- 
- 
- 
- 
- 

## Code Structure
-
-
-
-

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{nap2024,
  title={},
  author={Nap, Ronald and Aburidi, Mohammed and Marcia, Roummel},
  journal={EMBC},
  year={2024}
}
