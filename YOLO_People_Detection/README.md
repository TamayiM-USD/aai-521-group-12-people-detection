# YOLO People Detection - Training Results

This folder contains the complete training output from YOLOv8n model training for people detection.

## Training Configuration

- **Model**: YOLOv8-Nano (yolov8n.pt)
- **Dataset**: 15,210 images (People Detection Dataset from Roboflow)
- **Hardware**: NVIDIA L4 GPU (24GB VRAM)
- **Epochs**: 100 epochs with automatic checkpointing
- **Batch Size**: 48
- **Checkpoint Frequency**: Every 8 epochs (save_period=8)

## Folder Contents

### Model Files
- `best_model.pt` - Best performing checkpoint (top-level)
- `face_detection_l4/weights/best.pt` - Best model based on mAP
- `face_detection_l4/weights/last.pt` - Final epoch checkpoint
- `face_detection_l4/weights/epoch*.pt` - Training snapshots saved every 8 epochs (epoch0, epoch8, epoch16, ..., epoch96)

### Training Outputs
- `face_detection_l4/results.csv` - Training metrics per epoch
- `face_detection_l4/*.png` - Performance curves (F1, Precision, Recall, etc.)
- `face_detection_l4/train_batch*.jpg` - Training batch visualizations
- `face_detection_l4/val_batch*.jpg` - Validation predictions and labels
- `face_detection_l4/args.yaml` - Complete training configuration

### Model Exports
The best checkpoint was exported to multiple formats for deployment:
- `weights/best.onnx` - ONNX format (12MB)
- `weights/best.pb` - TensorFlow format (12MB)
- `weights/best.torchscript.zip` - TorchScript format (12MB)
- `weights/best_float32.tflite` - TensorFlow Lite FP32 (12MB)
- `weights/best_float16.tflite` - TensorFlow Lite FP16 (5.9MB)
- `weights/best_saved_model/` - TensorFlow SavedModel

## Training Snapshots

Regular checkpoints were automatically saved during training to enable:
- **Resume capability**: Training can resume from `last.pt` if interrupted
- **Model selection**: Compare performance across different epochs
- **Training analysis**: Study model evolution throughout training

The snapshots capture the model state at epochs: 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, and 96.

## Usage

Load the best model:
```python
from ultralytics import YOLO

model = YOLO('YOLO_People_Detection/best_model.pt')
results = model('image.jpg')
```

## Training Time
- **Duration**: ~2 hours on NVIDIA L4
- **Speed**: 60-80 images/sec
- **Optimizer**: AdamW with cosine learning rate schedule
