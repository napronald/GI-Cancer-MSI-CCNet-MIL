# GI-Cancer-MSI-CCNet-MIL
This repository contains the code for the paper "Contrastive Pre-Training and Multiple Instance Learning for Predicting Tumor Microsatellite Instability" (EMBC 2024). The project focuses on enhancing microsatellite instability (MSI) prediction in Whole Slide Image (WSI) analysis of gastrointestinal cancers through a two-stage weakly supervised methodology.

## Overview
We propose a framework that integrates Multiple Instance Learning (MIL) with a Contrastive Clustering Network (CCNet) for feature extraction. The method leverages the synergy of these approaches to significantly improve MSI classification accuracy, surpassing existing methods in the field.

![Figure 1](https://github.com/napronald/GI-Cancer-MSI-CCNet-MIL/blob/main/Figures/Figure1.png)


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

## Stage 1
You can start the training process by running:

```bash
python train.py
```
Once the training is completed, there will be a saved model in the "model_path" specified in arguments. To perform extraction with the trained model, run

```bash
python extractor.py
```

## Stage 2
You can start the 5-fold classification process by running:

```bash
python classifier.py
```

## Set Up
- Python 3.8+
- PyTorch 1.9.0
- torchvision 0.11.2
- scikit-learn 0.24.2
- pandas 1.1.5
- numpy 1.19.5
- scipy 1.5.4

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{nap2024,
  title={Contrastive Pre-Training and Multiple Instance Learning for Predicting Tumor Microsatellite Instability},
  author={Nap, Ronald and Aburidi, Mohammed and Marcia, Roummel},
  journal={EMBC},
  year={2024}
}
```
