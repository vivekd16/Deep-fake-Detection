# Deepfake Face Detection Project Details

## 1. Project Overview

This project detects whether a face image is **real** or **fake** using a deep learning image classifier built with **PyTorch** and deployed with **Gradio**.

The project includes:

- a notebook-based training pipeline
- a notebook for descriptive analytics and visualization
- a Gradio app for interactive prediction
- Grad-CAM visualization for interpretability
- saved training artifacts such as metrics, ROC data, and model checkpoint

The current implementation uses a **frozen ResNet18 backbone** with a custom classifier head for binary classification.

---

## 2. Problem Statement

Deepfakes are AI-generated or AI-manipulated media that can look highly realistic. In face-based deepfakes, synthetic images can be used for misinformation, impersonation, fraud, and other forms of misuse.

The aim of this project is to build a system that:

- takes a face image as input
- predicts whether it is **real** or **fake**
- explains the prediction visually using Grad-CAM
- presents performance analytics in a simple interface

---

## 3. Dataset Used

Dataset source:

- Kaggle: `140k Real and Fake Faces`
- URL: <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>

### 3.1 Dataset Summary

According to the Kaggle dataset card, the dataset contains:

- `70,000` real face images
- `70,000` fake face images
- total: `140,000` images

The dataset description states that:

- the **real** images come from the Flickr face dataset collected by NVIDIA
- the **fake** images are sampled from a larger set of StyleGAN-generated fake faces
- the images were resized to `256px`
- the dataset was already split into training, validation, and test sets
- CSV files were provided for convenience

### 3.2 Split Used in This Project

This project uses the provided CSV files and folder structure. The current split in the repository is:

| Split | Total | Real | Fake |
|---|---:|---:|---:|
| Train | 100,000 | 50,000 | 50,000 |
| Validation | 20,000 | 10,000 | 10,000 |
| Test | 20,000 | 10,000 | 10,000 |

Approximate percentages:

- Train: `71.43%`
- Validation: `14.29%`
- Test: `14.29%`

### 3.3 Dataset Layout in This Project

```text
deepfakedetection/
|-- train.csv
|-- valid.csv
|-- test.csv
`-- real_vs_fake/
    `-- real-vs-fake/
        |-- train/
        |   |-- fake/
        |   `-- real/
        |-- valid/
        |   |-- fake/
        |   `-- real/
        `-- test/
            |-- fake/
            `-- real/
```

### 3.4 Labels

This project uses:

- `fake -> 0`
- `real -> 1`

---

## 4. High-Level Project Architecture

The overall project flow is:

```text
Dataset
  -> CSV-driven sample loading
  -> Image preprocessing and augmentation
  -> ResNet18-based classifier
  -> Training and validation
  -> Best checkpoint + metrics saved in artifacts/
  -> Gradio app loads best model
  -> User uploads image
  -> Prediction + confidence + Grad-CAM
  -> Analytics charts shown in app
```

### 4.1 Main Components

1. **Dataset Loader**
   Reads image paths and labels from `train.csv`, `valid.csv`, and `test.csv`.

2. **Preprocessing and Augmentation**
   Applies resizing, random crop, horizontal flip, color jitter, normalization, and evaluation transforms.

3. **Model**
   Uses ResNet18 as the feature extractor and a custom fully connected head for binary classification.

4. **Training Pipeline**
   Handles dataloaders, loss, optimizer, scheduler, checkpointing, and early stopping.

5. **Evaluation**
   Computes loss, accuracy, F1-score, ROC-AUC, confusion matrix, and ROC curve points.

6. **Deployment**
   Loads the saved checkpoint in a Gradio app for inference.

7. **Interpretability**
   Uses Grad-CAM to highlight image regions that contributed most to the prediction.

8. **Analytics**
   Displays confusion matrix, ROC curve, and training curves in notebooks and the app.

---

## 5. ResNet Architecture Used

The current model is based on **ResNet18**.

### 5.1 Why ResNet

ResNet is a strong choice for this problem because:

- it is a proven architecture for image classification
- residual connections improve training stability
- transfer learning from ImageNet is straightforward
- it offers a good balance between performance and complexity
- it is easier to explain than heavier or more experimental architectures

### 5.2 ResNet18 Core Structure

ResNet18 mainly uses:

- `7 x 7` convolution in the first layer
- `3 x 3` convolutions in residual blocks
- `1 x 1` convolutions in shortcut/downsampling paths where needed

High-level structure:

1. Input image: `224 x 224 x 3`
2. Initial convolution: `7 x 7`, stride `2`
3. Max-pooling layer
4. Four residual stages with increasing channels:
   - `64`
   - `128`
   - `256`
   - `512`
5. Global average pooling
6. Fully connected classifier head

### 5.3 Classifier Head Used in This Project

The default ImageNet classifier is replaced with a custom head:

```text
in_features -> 256 -> 64 -> 2
```

With activation and regularization:

```text
Dropout
Linear(in_features, 256)
ReLU
Dropout
Linear(256, 64)
ReLU
Dropout
Linear(64, 2)
```

Output classes:

- `fake`
- `real`

### 5.4 Frozen Backbone Strategy

In the current setup:

- the pretrained ResNet18 backbone remains **frozen**
- only the classifier head is trained

This reduces training cost and makes the model more stable for the current project setup.

---

## 6. Training Pipeline

Training is implemented in:

- `deepfake_training.ipynb`

### 6.1 Preprocessing and Augmentation

Training transform:

- resize to slightly larger size
- random resized crop
- random horizontal flip
- light color jitter
- tensor conversion
- ImageNet normalization

Evaluation transform:

- resize to target input size
- tensor conversion
- ImageNet normalization

### 6.2 Current Training Configuration

Current configuration from `artifacts/config.json`:

| Parameter | Value |
|---|---|
| Model | `resnet18` |
| Image size | `224` |
| Batch size | `32` |
| Epochs | `10` |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Num workers | `0` |
| Patience | `3` |
| Dropout | `0.3` |
| Label smoothing | `0.05` |
| Seed | `42` |
| Freeze backbone epochs | `10` |

### 6.3 Loss Function

The model uses:

- `CrossEntropyLoss`
- with label smoothing

Why:

- the task is binary classification
- the model outputs `2` logits
- CrossEntropyLoss is the standard choice for this setup

### 6.4 Softmax Usage

Softmax is used during evaluation and inference to convert logits into probabilities.

Training does **not** manually apply softmax before loss because `CrossEntropyLoss` handles that internally.

### 6.5 Optimizer and Scheduler

Optimizer:

- `AdamW`

Scheduler:

- `ReduceLROnPlateau`

Purpose:

- `AdamW` helps with stable optimization
- `ReduceLROnPlateau` lowers the learning rate if validation performance stops improving

### 6.6 Early Stopping

The training pipeline uses patience-based early stopping logic.

Current patience:

- `3`

Meaning:

- if validation performance does not improve for `3` consecutive epochs, training can stop early

---

## 7. Evaluation Metrics

The project computes the following metrics:

- loss
- accuracy
- F1-score
- ROC-AUC
- confusion matrix
- classification report
- ROC curve points

### 7.1 Best Validation Metrics

From `artifacts/best_valid_metrics.json`:

- Loss: `0.4169`
- Accuracy: `0.83685`
- F1-score: `0.83678`
- ROC-AUC: `0.91698`

### 7.2 Final Test Metrics

From `artifacts/test_metrics.json`:

- Loss: `0.41783`
- Accuracy: `0.83755`
- F1-score: `0.83801`
- ROC-AUC: `0.91600`

### 7.3 Confusion Matrix

Current test confusion matrix:

```text
[[8347, 1653],
 [1596, 8404]]
```

Interpretation:

- correctly predicted fake: `8347`
- fake predicted as real: `1653`
- real predicted as fake: `1596`
- correctly predicted real: `8404`

These results are much more realistic than the earlier near-perfect runs and suggest better-balanced generalization within this dataset setup.

---

## 8. Artifacts Produced

The training pipeline saves outputs into `artifacts/`.

Current important files:

- `best_model.pt`
- `best_valid_metrics.json`
- `best_valid_roc_curve.json`
- `test_metrics.json`
- `best_model_roc_curve.json`
- `history.json`
- `config.json`

### 8.1 What Each Artifact Contains

- `best_model.pt`
  Best saved model checkpoint

- `best_valid_metrics.json`
  Metrics on the best validation epoch

- `best_valid_roc_curve.json`
  ROC curve points for the best validation checkpoint

- `test_metrics.json`
  Final evaluation metrics on the test set

- `best_model_roc_curve.json`
  ROC curve points for the best saved model on the test split

- `history.json`
  Epoch-wise training and validation history

- `config.json`
  Exact training configuration used for the saved model

---

## 9. Gradio Application

Deployment is implemented in:

- `app.py`

### 9.1 What the App Does

The app:

- loads the saved checkpoint from `artifacts/best_model.pt`
- preprocesses an uploaded image
- predicts whether the image is real or fake
- shows prediction confidence
- generates a Grad-CAM overlay
- renders descriptive analytics charts under the main interface

### 9.2 Current UI Outputs

Main inference section:

- face image input
- prediction summary
- Grad-CAM overlay

Analytics section:

- confusion matrix
- ROC curve
- training curves by epoch

### 9.3 Grad-CAM

Grad-CAM is used to improve interpretability.

Purpose:

- highlights image regions that influenced the model prediction
- helps verify whether the model focuses on face regions instead of irrelevant image areas

Target layer used:

- the last block of `layer4` in ResNet

---

## 10. Descriptive Analytics Notebook

Analytics are implemented in:

- `descriptive_analytics.ipynb`

Current notebook contents:

- dataset split and class balance summary
- best validation and final test metrics summary
- training curves
- confusion matrix
- per-class metrics chart
- ROC curve for the best model

This notebook is meant for:

- project presentation
- analysis of training behavior
- metric visualization
- report preparation

---

## 11. Project File Structure

Important project files:

```text
deepfakedetection/
|-- app.py
|-- deepfake_training.ipynb
|-- descriptive_analytics.ipynb
|-- requirements.txt
|-- README.md
|-- train.csv
|-- valid.csv
|-- test.csv
|-- artifacts/
|   |-- best_model.pt
|   |-- best_valid_metrics.json
|   |-- best_valid_roc_curve.json
|   |-- test_metrics.json
|   |-- best_model_roc_curve.json
|   |-- history.json
|   `-- config.json
|-- imgForDemo/
|   |-- fake/
|   `-- real/
`-- real_vs_fake/
    `-- real-vs-fake/
        |-- train/
        |-- valid/
        `-- test/
```

---

## 12. Requirements

From `requirements.txt`, the current Python dependencies are:

- `torch`
- `torchvision`
- `gradio`
- `huggingface_hub`
- `Pillow`
- `tqdm`
- `numpy`
- `matplotlib`
- `jupyter`

### 12.1 Software Requirements

- Python
- Jupyter Notebook
- PyTorch
- Torchvision
- Gradio
- NumPy
- Matplotlib
- PIL / Pillow

### 12.2 Hardware Requirements

Recommended:

- GPU-enabled system for training
- sufficient disk space for the dataset and artifacts

GPU used in this project:

- `NVIDIA GeForce RTX 4050 Laptop GPU`

---

## 13. How to Run the Project

### 13.1 Training

Open:

- `deepfake_training.ipynb`

Run the notebook cells from top to bottom.

### 13.2 Analytics

Open:

- `descriptive_analytics.ipynb`

Run the cells to view charts and summaries.

### 13.3 Gradio App

Run:

```powershell
python app.py --checkpoint artifacts/best_model.pt
```

Then open:

```text
http://127.0.0.1:7860
```

---

## 14. Current Strengths of the Project

- clean notebook-based training workflow
- balanced dataset split
- reproducible saved artifacts
- Grad-CAM-based interpretability
- browser-based deployment with Gradio
- integrated analytics for presentation and reporting
- realistic evaluation metrics after reducing over-optimistic earlier behavior

---

## 15. Current Limitations

- the current model uses only a frozen ResNet18 backbone
- generalization to completely different deepfake datasets is not yet tested
- the project focuses on image-level deepfake detection, not video-level detection
- no face-crop specific preprocessing pipeline is currently used
- the dataset may still contain dataset-specific shortcuts not fully removed

---

## 16. Possible Future Improvements

- partially unfreeze the backbone for fine-tuning
- compare ResNet18 with ResNet34 or ResNet50
- add face detection and train on cropped faces only
- evaluate on another external deepfake dataset
- add multiple checkpoint selection in the app
- extend from image-based detection to video deepfake detection
- add confidence calibration and deeper bias analysis

---

## 17. Source and Attribution

Dataset used:

- Kaggle dataset page: <https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces>

Dataset description used for this document was based on the Kaggle dataset card, which states that the dataset contains `70k` real faces and `70k` StyleGAN-generated fake faces, resized and split into train, validation, and test sets with CSV support.
