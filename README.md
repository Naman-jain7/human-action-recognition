# NTU RGB+D Skeleton Action Recognition Classifier

Complete LSTM-based action recognition system for the NTU RGB+D skeleton dataset. Optimized for Google Colab with T4 GPU.

## 📋 Overview

This project implements a bidirectional LSTM with attention mechanism for skeleton-based action recognition. The model processes sequences of 3D joint coordinates (25 joints × 3 coordinates = 75 features) and predicts one of 60 action classes.

**Features:**
- ✅ Bidirectional LSTM + Multi-head Attention
- ✅ Data normalization and preprocessing
- ✅ Early stopping & learning rate scheduling
- ✅ Comprehensive evaluation metrics
- ✅ Google Colab optimized (T4 GPU)
- ✅ Production-ready inference pipeline

## 🚀 Quick Start (Google Colab)

### 1. **Mount Google Drive & Install Dependencies**

```python
# Cell 1: Setup
!pip install -q torch scikit-learn matplotlib seaborn tqdm

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
```

### 2. **Upload Code**

```python
# Cell 2: Download files
!cd /content && git clone https://github.com/your-repo/skeleton-action-recognition.git
# OR manually upload the 3 Python files:
# - action_recognition_classifier.py
# - inference.py
# - colab_quick_start.py
```

### 3. **Organize Your Data**

```
/content/gdrive/My Drive/
└── nturgbd_data/
    └── skeletons/
        ├── S001C001P001R001A001.skeleton
        ├── S001C001P001R002A001.skeleton
        ├── S001C001P002R001A001.skeleton
        └── ... (56,000 files)
```

### 4. **Train Model**

```python
# Cell 3: Training
import sys
sys.path.insert(0, '/content/skeleton-action-recognition')

from action_recognition_classifier import main
import torch

# Configuration
DATA_DIR = "/content/gdrive/My Drive/nturgbd_data/skeletons"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Train
print(f"GPU: {torch.cuda.get_device_name(0)}")
model, trainer, action_map = main(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE
)

# Save results
!cp final_model.pth /content/gdrive/My\ Drive/
!cp model_config.json /content/gdrive/My\ Drive/
!cp scaler.pkl /content/gdrive/My\ Drive/
```

## 📊 Data Format

**NTU RGB+D Skeleton File Structure:**

```
103                          # Number of frames
1                           # Number of bodies (usually 1)
72057594037931101 0 1 ...  # Body metadata
25                          # Number of joints (always 25)
x1 y1 z1 ... (12 values)   # Joint 1 data
x2 y2 z2 ... (12 values)   # Joint 2 data
... (25 joints total)
```

**Features extracted:** 25 joints × 3 coordinates (x, y, z) = 75 features

**25 Joints in NTU RGB+D:**
0. Base of Spine, 1. Middle Spine, 2. Neck, 3. Head
4. Left Shoulder, 5. Left Elbow, 6. Left Wrist, 7. Left Hand
8. Right Shoulder, 9. Right Elbow, 10. Right Wrist, 11. Right Hand
12. Left Hip, 13. Left Knee, 14. Left Ankle, 15. Left Foot
16. Right Hip, 17. Right Knee, 18. Right Ankle, 19. Right Foot
20. Spine (center), 21-24. Additional spine/head joints

## 🏗️ Model Architecture

```
Input: (batch_size, sequence_length, 75)
    ↓
Bidirectional LSTM (2 layers, hidden_size=256)
    ↓
Multi-head Attention (8 heads)
    ↓
Global Average Pooling + Last Frame Output
    ↓
Fully Connected Layers (256 → 128 → 60)
    ↓
Output: Class logits (60 action classes)
```

**Key Design Choices:**
- **Bidirectional LSTM:** Captures temporal patterns in both directions
- **Multi-head Attention:** Focuses on relevant frames
- **Dual Features:** Combines average pooling + final state (captures both global and local temporal patterns)
- **Dropout & L2 Regularization:** Prevents overfitting

## 📁 File Guide

| File | Purpose |
|------|---------|
| `action_recognition_classifier.py` | Main training pipeline, data loading, model definition |
| `inference.py` | Inference, evaluation, and reporting utilities |
| `colab_quick_start.py` | Colab setup guide, troubleshooting, advanced features |

## 🔧 Configuration

**Hyperparameters** (in `main()` function):

```python
batch_size = 32          # Batch size (reduce to 16 if OOM)
epochs = 50              # Maximum epochs
learning_rate = 0.001    # Initial learning rate
```

**Model Parameters** (in `SkeletonLSTM`):

```python
input_size = 75          # 25 joints × 3 coordinates
hidden_size = 256        # LSTM hidden dimension (↓ to 128 if OOM)
num_layers = 2           # Number of LSTM layers
num_classes = 60         # Action classes
dropout = 0.3            # Dropout rate
```

**Data Parameters** (in `SkeletonDataset`):

```python
max_sequence_length = 300  # Pad/truncate sequences
```

## 📈 Expected Performance

- **Train Time:** ~2-3 hours per 50 epochs on T4 GPU
- **Expected Accuracy:** 75-85% (depends on data quality & hyperparameters)
- **Memory Usage:** ~6-8 GB GPU (with batch_size=32)

| Dataset Split | Size | Purpose |
|---|---|---|
| Train | 44,800 | Model training |
| Val | 5,600 | Hyperparameter tuning & early stopping |
| Test | 5,600 | Final evaluation |

## 💾 Output Files

After training:

```
final_model.pth          # Trained model weights
model_config.json        # Model architecture & action mapping
scaler.pkl              # Feature normalization (StandardScaler)
training_history.png    # Loss and accuracy curves
```

## 🎯 Using Trained Model for Inference

```python
from inference import ActionRecognitionInference
import numpy as np

# Load model
inference = ActionRecognitionInference(
    model_path='final_model.pth',
    config_path='model_config.json',
    scaler_path='scaler.pkl',
    device='cuda'
)

# Predict single skeleton
skeleton = np.random.randn(150, 75)  # 150 frames, 75 features
action_id, confidence, top_3 = inference.predict_single(skeleton)

print(f"Predicted Action: {action_id}")
print(f"Confidence: {confidence:.4f}")
for action_name, prob in top_3:
    print(f"  {action_name}: {prob:.4f}")
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
```python
# Option 1: Reduce batch size
batch_size = 16  # instead of 32

# Option 2: Reduce model size
SkeletonLSTM(
    hidden_size=128,  # instead of 256
    num_layers=1,     # instead of 2
)

# Option 3: Reduce sequence length
max_sequence_length = 200  # instead of 300
```

### Model Not Converging

**Symptoms:** Accuracy stuck at low value

**Solutions:**
1. Check label mapping with `print(action_map)`
2. Visualize skeleton sequences to ensure valid data
3. Try different learning rates: `[0.00005, 0.0001, 0.0005]`
4. Increase training epochs

### Slow Data Loading

**Solutions:**
```python
# Preprocess once and save
train_dataset = SkeletonDataset(...)
torch.save({
    'sequences': train_dataset.sequences,
    'labels': train_dataset.sequence_labels
}, 'preprocessed_data.pt')

# Load preprocessed data
data = torch.load('preprocessed_data.pt')
```

### Colab Disconnection

**Solutions:**
1. Save checkpoints to Google Drive
2. Use `!cp` to copy important files to /gdrive
3. Implement checkpoint loading in training loop:

```python
def save_checkpoint(epoch, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, '/gdrive/checkpoint.pth')
```

## 📊 Evaluation & Visualization

```python
from inference import ActionRecognitionInference

inference = ActionRecognitionInference(...)

# Generate evaluation report
report = inference.generate_report(test_loader)

# Output:
# - confusion_matrix.png (shows misclassified actions)
# - per_class_accuracy.png (accuracy per action)
# - classification_report.txt (precision, recall, F1)
```

## 🚀 Advanced Techniques

### Data Augmentation

```python
from colab_quick_start import DataAugmentation

augmenter = DataAugmentation()

# Apply augmentations
skeleton = augmenter.temporal_jitter(skeleton, jitter_range=0.05)
skeleton = augmenter.spatial_noise(skeleton, noise_std=0.02)
```

### Ensemble Methods

Train multiple models with different random seeds:

```python
models = []
for seed in [42, 123, 456]:
    torch.manual_seed(seed)
    model, _, _ = main(DATA_DIR, epochs=50)
    models.append(model)

# Average predictions
predictions = torch.stack([m(x) for m in models]).mean(dim=0)
```

### Model Quantization (for deployment)

```python
import torch.quantization as q

model.eval()
model_quant = q.quantize_dynamic(
    model,
    {torch.nn.LSTM, torch.nn.Linear},
    dtype=torch.qint8
)
torch.save(model_quant.state_dict(), 'model_quantized.pth')
```

## 📚 References

- **NTU RGB+D Dataset:** [Official Paper](https://arxiv.org/abs/1604.02424)
- **LSTM Architecture:** [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Attention Mechanism:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 🤝 Contributing

Improvements welcome! Areas for enhancement:
- Graph Convolutional Networks (GCN) for skeleton-aware learning
- Temporal Segment Networks (TSN)
- Data augmentation strategies
- Multi-stream architectures

## 📝 License

MIT License - feel free to use for research and commercial projects

## ✉️ Support

For issues or questions:
1. Check troubleshooting section above
2. Review console output for error messages
3. Ensure data format matches expected structure
4. Verify GPU memory availability

---

**Happy training! 🎯**
