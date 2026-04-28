# 🏃‍♂️ Human Action Recognition with CTR-GCN

![Accuracy](https://img.shields.io/badge/Accuracy-95.76%25-brightgreen)
![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![Tracking](https://img.shields.io/badge/Tracking-MediaPipe-blue)

A high-performance skeleton-based Human Action Recognition system trained on the **NTU RGB+D** dataset. This project implements a **Channel-wise Topology Refinement Graph Convolutional Network (CTR-GCN)** to achieve state-of-the-art accuracy in predicting 60 different action classes.

It completely replaces previous generation LSTM-based approaches, boosting test accuracy from ~85% to **95.76%**, while enabling real-time, lightweight inference using OpenCV and MediaPipe.

---

## 🎯 Key Achievements

- **Architecture Upgrade:** Migrated from a Bidirectional LSTM baseline to a highly robust Graph Convolutional Network (CTR-GCN) ensemble, significantly improving spatial-temporal feature extraction.
- **Top-Tier Accuracy:** Achieved **95.76%** global accuracy across all 60 complex action classes of the NTU RGB+D benchmark.
- **Live Video Inference:** Engineered a seamless inference pipeline (`inference.py`) that uses MediaPipe to extract human pose landmarks from live video, normalizes them, maps them to the 17-joint COCO layout expected by the model, and classifies actions in real-time.

---

## 🏗️ Technical Architecture

The core model is an implementation of **CTR-GCN**, designed to dynamically learn the topology of the human skeleton across different channels rather than relying on a static, predefined physical graph.

### Pipeline Breakdown
1. **Adaptive Graph Convolution (AdaptiveGCNBlock):** Learns adaptive joint connections dynamically (`A_adaptive = A + PA`) using Einstein summation, enabling the model to capture non-physical interactions between joints (e.g., the relationship between a hand and a foot while jumping).
2. **Multi-Scale Temporal Convolution (MultiScaleTCN):** Captures temporal motion dynamics (like walking or waving) across multiple frames using multi-scale kernel convolutions and residual connections.
3. **Data Normalization:** Includes `BatchNorm1d` applied over the temporal and spatial dimensions to heavily normalize coordinate variances, preventing vanishing gradients on raw pixel input.
4. **Logit Fusion:** Capable of ensembling Joint and Bone streams for maximum prediction reliability.

---

## 📊 Results & Performance

The model was trained for 60 epochs using a cosine annealing learning rate scheduler with Nesterov momentum SGD.

| Metric | Score |
| ------ | ----- |
| **Global Accuracy** | 95.76% |
| **Macro Avg Precision** | 95.85% |
| **Weighted Avg F1-Score**| 95.77% |

### Training History & Confusion Matrix
*Visualizations of the model's convergence and class-wise predictions:*

![Confusion Matrix](outputs/ctr_gcn_best/confusion.png)

*(You can also view the full training curves at `outputs/ctr_gcn_best/training.png`)*

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/human-action-recognition.git
cd human-action-recognition
```

### 2. Download the Dataset
The model expects the HRNet skeletons dataset formatted for NTU RGB+D 60.
Download the dataset (`ntu60_hrnet.pkl`) from Kaggle and place it in the `models/` directory:
- [Skeleton Dataset NTU-RGB+D 60 (Kaggle)](https://www.kaggle.com/datasets/namanjain321/skeleton-dataset-ntu-rgbd60)

### 3. Install Dependencies
It is highly recommended to use a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Live Inference Quickstart

You can run the trained model directly on your webcam feed. The script dynamically tracks your skeleton and bridges the 33 landmarks detected by MediaPipe to the 17 COCO landmarks the CTR-GCN model was trained on. It automatically scales normalized coordinates back to absolute image pixels to match the model's trained Batch Normalization variance.

```bash
python inference.py
```
*Press `q` inside the video window to quit.*

---

## 📂 Project Structure

| File / Folder | Purpose |
|---------------|---------|
| `inference.py` | Real-time action recognition script using OpenCV, MediaPipe, and PyTorch. |
| `notebooks/building_model.ipynb` | Comprehensive training pipeline, dataset pre-processing, and model architecture definitions. |
| `models/` | Stores the downloaded skeletons and exported model weights (`best_ctr_gcn_joints.pth`). |
| `outputs/ctr_gcn_best/` | Contains the classification reports, loss/accuracy curves, and confusion matrices. |

---

## 🔮 Future Work
- **3-Stream Fusion:** Expanding the current 2-stream (Joint, Bone) architecture to incorporate Joint-Motion and Bone-Motion streams for enhanced velocity tracking.
- **Dynamic Channel Detection:** Refining the pipeline to dynamically swap between `in_channels=2` (X, Y) and `in_channels=3` (X, Y, Confidence) based on the loaded checkpoint configuration.

---
*Developed for robust, real-time spatial-temporal analysis.*
