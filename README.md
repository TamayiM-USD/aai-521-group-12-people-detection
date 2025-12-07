# Situational Awareness via Computer Vision: People Counting and Facial Emotion Recognition

**AAI-521 Group-12**
Manoj Nair, Tamayi Mlanda
University of San Diego
December 2024

## Overview
This project implements a dual-pipeline computer vision system for automated situational awareness that simultaneously detects individuals and classifies their emotional states from static images. The system combines YOLOv8 for object detection with ResNet18 for facial emotion recognition, providing both quantitative (people counting) and qualitative (emotion analysis) insights from visual data.

**Key Features:**
- YOLOv8-based people detection and counting
- ResNet18 transfer learning for 7-class emotion recognition (anger, disgust, fear, happy, sad, surprise, neutral)
- Integrated inference pipeline using Haar Cascade for face localization
- Applications in surveillance, retail analytics, and crowd safety

## Repository Structure

```
.
├── Team13_FinalProject-Notebook.ipynb   # Complete integrated system with emotion recognition
├── YOLO_People_Detection/               # YOLO model for people detection
│   ├── best_model.pt                    # Best trained model (6MB)
│   └── face_detection_l4/               # Training results and weights
│       ├── args.yaml                    # Training configuration
│       ├── results.csv                  # Training metrics
│       └── weights/                     # Model checkpoints and exports
│           ├── best.pt                  # Best checkpoint
│           ├── last.pt                  # Latest checkpoint
│           ├── epoch*.pt                # Training epoch checkpoints
│           ├── best.onnx                # ONNX format export
│           ├── best.pb                  # TensorFlow format
│           ├── best.tflite              # TensorFlow Lite formats
│           └── best_saved_model/        # TensorFlow SavedModel
└── README.md                            # This file
```

## Git LFS (Large File Storage)

This repository uses Git LFS to manage large model files efficiently.

### Tracked File Types
The following file types in the YOLO_People_Detection folder are tracked with Git LFS:
- PyTorch models: `*.pt`, `*.pth`
- ONNX models: `*.onnx`
- TensorFlow models: `*.pb`, `*.tflite`
- Model archives: `*.torchscript.zip`

### Cloning with Git LFS

To clone this repository with all large files:

1. Install Git LFS:
   ```bash
   # Windows (with Git for Windows)
   git lfs install

   # macOS
   brew install git-lfs
   git lfs install

   # Linux (Ubuntu/Debian)
   sudo apt-get install git-lfs
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

   Git LFS will automatically download the large files.

### Working with LFS Files

- **View tracked files**: `git lfs ls-files`
- **Manually pull LFS files**: `git lfs pull`
- **Check LFS status**: `git lfs status`

## YOLO_People_Detection Folder

Contains trained YOLO (You Only Look Once) model for people/face detection:

- **Main model**: `best_model.pt` - The primary trained model file (6MB)
- **Training results**: Multiple epoch checkpoints and training visualizations
- **Model formats**: Exports in ONNX, TensorFlow, TensorFlow Lite, and TorchScript formats for deployment flexibility

### Model Files
- Total size: ~420MB (managed via Git LFS)
- 16 PyTorch checkpoint files
- 6 different export formats for various deployment targets

## Requirements

### Dependencies
- Python 3.x
- PyTorch
- Ultralytics YOLOv8
- OpenCV (cv2)
- TorchVision
- PIL (Pillow)
- NumPy
- Matplotlib

### Installation
```bash
pip install torch torchvision ultralytics opencv-python pillow numpy matplotlib
```

## Usage

### 1. People Detection with YOLOv8
The YOLOv8 model is trained to detect and count people in images:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('YOLO_People_Detection/best_model.pt')

# Run inference
results = model('path/to/image.jpg')
results.show()  # Display results with bounding boxes
```

### 2. Facial Emotion Recognition
The ResNet18 model classifies emotions into 7 categories:

```python
import torch
from torchvision import transforms
from PIL import Image

# Load the emotion recognition model
model = torch.load('path/to/emotion_model.pth')
model.eval()

# Preprocess image (48x48 grayscale to 224x224 RGB)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Classify emotion
image = Image.open('face.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
output = model(input_tensor)
emotion = output.argmax(1)
```

### 3. Integrated Pipeline
Run the complete system using the Jupyter notebook:

- **Team13_FinalProject-Notebook.ipynb**: Full integrated system combining YOLOv8 detection and emotion classification

The integrated system:
1. Detects people using YOLOv8
2. Localizes faces using Haar Cascade
3. Classifies emotions using ResNet18
4. Visualizes results with bounding boxes and emotion labels

### 4. Model Files
The YOLO_People_Detection folder contains:
- **best_model.pt**: Best performing model checkpoint (6MB)
- **Epoch checkpoints**: Training progression snapshots (epoch0.pt through epoch96.pt)
- **Export formats**: ONNX, TensorFlow (.pb), TensorFlow Lite (.tflite), and TorchScript for deployment flexibility

## Datasets

This project utilizes two datasets:

1. **People Detection Dataset** (17,000 images)
   - Source: Roboflow
   - Format: YOLO-format bounding box annotations
   - Environments: Indoor hallways and outdoor crowds
   - Used for: Training YOLOv8 object detection model

2. **FER2013 Dataset** (35,887 images)
   - Source: Goodfellow et al., 2013
   - Format: 48x48 grayscale facial images
   - Classes: 7 emotions (anger, disgust, fear, happy, sad, surprise, neutral)
   - Used for: Training ResNet18 emotion classifier with transfer learning
   - Note: Dataset exhibits class imbalance (more "happy" samples than "disgust")

## Model Performance

**YOLOv8 (People Detection):**
- Architecture: YOLOv8-Nano (v8n)
- Metric: Mean Average Precision (mAP) based on IoU
- Performance: High precision across varied lighting conditions
- Speed: Real-time inference capability

**ResNet18 (Emotion Recognition):**
- Architecture: ResNet18 with transfer learning from ImageNet
- Training: Frozen convolutional layers, custom FC layer for 7 classes
- Accuracy: Higher on distinct emotions (happy, surprise), lower on subtle expressions (sad/neutral confusion)
- Limitation: 48x48 upsampled images limit fine-grained feature extraction

## Technical Notes

**Integration Pipeline:**
The system bridges YOLO (body detection) and ResNet (face analysis) using:
1. YOLOv8 generates bounding boxes for all detected persons
2. Haar Cascade classifier localizes faces within the scene
3. Facial ROIs are cropped, transformed to tensors, and passed to ResNet18
4. Results are visualized with composite overlays showing presence and mood

**Implementation Challenge:**
Data type mismatch between OpenCV (NumPy arrays) and PyTorch transforms (PIL Images) was resolved through explicit type conversion before transformation pipeline.

## References

- Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. Neural Networks, 64, 59-63.
- He, K., et al. (2016). Deep residual learning for image recognition. CVPR.
- Jocher, G., et al. (2023). Ultralytics YOLO (Version 8.0.0). https://github.com/ultralytics/ultralytics

## License
See [LICENSE](LICENSE) file for details.
