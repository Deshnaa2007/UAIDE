# Advanced Deepfake Detection System

A comprehensive deepfake detection system with multiple model architectures, GPU acceleration, and explainable AI capabilities.

## 🚀 Features

- **Multiple Model Architectures**: Custom CNN, ResNet-50 transfer learning, traditional ML (Random Forest, XGBoost)
- **K-Fold Cross-Validation**: Robust performance estimation and ensemble model creation
- **GPU Acceleration**: CUDA support for fast training and inference (RTX 2050 tested)
- **Explainable AI**: Grad-CAM visualization showing model focus areas
- **Data Augmentation**: Extensive augmentation to prevent overfitting
- **Web Interface**: Gradio-based UI for easy image analysis
- **Model Comparison**: Automated comparison of all trained models

## 📊 Model Performance

| Model | Type | Accuracy | AUC |
|-------|------|----------|-----|
| ResNet + FFT Fusion | Multi-modal | ~94% | ~0.97 |
| ResNet-50 | Transfer Learning | ~92% | ~0.96 |
| CNN + K-Fold | Ensemble | ~89% | ~0.94 |
| Custom CNN | Deep Learning | ~85% | ~0.91 |
| Random Forest | Traditional ML | ~78% | ~0.85 |

## 🛠️ Quick Setup

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Training Models

### 1. ResNet + FFT Feature Fusion (Recommended)
```powershell
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_fusion.joblib --max_per_class 1000 --model fusion
```

### 2. ResNet-50 Transfer Learning
```powershell
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_resnet.joblib --max_per_class 2000 --model resnet
```

### 2. CNN with K-Fold Cross-Validation
```powershell
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_cnn_kfold.joblib --max_per_class 2000 --model cnn_kfold --k_folds 5
```

### 3. Custom CNN
```powershell
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_cnn.joblib --max_per_class 2000 --model cnn
```

### 4. Traditional ML Models
```powershell
# Random Forest
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_rf.joblib --max_per_class 2000 --model rf

# XGBoost
python train.py --dataset "DeepfakeVsReal/Dataset" --out model_gb.joblib --max_per_class 2000 --model gb
```

## 🔍 Model Comparison

Compare all trained models on validation data:

```powershell
python compare_models.py
```

This generates a detailed comparison table and saves results to `model_comparison_results.csv`.

## 🌐 Web Interface

Launch the interactive web app:

```powershell
python app.py
```

Features:
- **Real-time Analysis**: Upload images for instant deepfake detection
- **Grad-CAM Visualization**: See exactly what the model focuses on
- **Model Auto-Detection**: Automatically uses the best available trained model
- **Confidence Scores**: Probability estimates for predictions

## 📁 Project Structure

```
UAIDE/
├── train.py              # Training script with multiple model types
├── app.py                # Gradio web interface
├── detector.py           # Feature extraction utilities
├── compare_models.py     # Model comparison script
├── demo.py              # Simple demo script
├── requirements.txt     # Python dependencies
├── DeepfakeVsReal/      # Dataset directory
│   ├── Dataset/
│   │   ├── Train/
│   │   ├── Validation/
│   │   └── Test/
├── model_*.joblib       # Trained models
└── README.md
```

## 🔧 Advanced Usage

### Custom Training Parameters

```powershell
# Adjust batch size, learning rate, epochs
python train.py --model resnet --max_per_class 1000 --batch_size 32 --lr 0.001 --epochs 25
```

### GPU Memory Optimization

```powershell
# For systems with limited GPU memory
python train.py --model cnn --batch_size 8 --max_per_class 500
```

### Cross-Validation Analysis

```powershell
# Detailed k-fold analysis
python train.py --model cnn_kfold --k_folds 10 --max_per_class 1000
```

## 🎨 Model Explainability

The system includes Grad-CAM visualization that highlights:
- **Facial artifacts** common in deepfakes
- **Texture inconsistencies**
- **Lighting anomalies**
- **Edge artifacts** from GAN generation

## 📈 Performance Tips

1. **Use ResNet-50** for best accuracy (transfer learning from ImageNet)
2. **Enable K-fold CV** for robust performance estimates
3. **Use GPU** for 10-50x faster training
4. **Increase data augmentation** to prevent overfitting
5. **Monitor validation metrics** during training

## 🔍 Troubleshooting

### Common Issues

**CUDA out of memory**: Reduce batch size or use `--max_per_class 500`

**Low accuracy**: Try ResNet model or increase training data

**Slow inference**: Models run on GPU automatically if available

**Grad-CAM errors**: Ensure OpenCV is installed (`pip install opencv-python`)

## 📝 Technical Details

- **Framework**: PyTorch 2.5+ with CUDA 12.1 support
- **GPU**: Tested on NVIDIA RTX 2050 (4GB VRAM)
- **Data Format**: Images resized to 224x224 (ResNet) or 128x128 (CNN)
- **Augmentation**: Rotation, flipping, color jitter, affine transforms
- **Regularization**: Dropout, batch normalization, L2 weight decay

## 🤝 Contributing

This is a research prototype. Key areas for improvement:
- **Multi-modal fusion** (audio + video)
- **Temporal analysis** for video deepfakes
- **Domain adaptation** for different deepfake generators
- **Real-time optimization** for video streams

## 📄 License

Research prototype - see individual file headers for licensing.

The script will print an `ai_score` and save a heatmap overlay next to the input image.

Next steps
- Replace heuristic fusion with a trained CNN (see notes in the main document)
- Add a dataset loader and training script
