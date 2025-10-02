# QR Code Detection System

A YOLOv8-based system for detecting multiple QR codes in images. This project provides complete training and inference pipelines for QR code detection in complex, multi-QR environments.


## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## Environment Setup

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd multiqr_hackathon
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following main packages:
- `ultralytics>=8.0.0` - YOLOv8 framework
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=10.0.0` - Image handling
- `scikit-learn>=1.3.0` - Train/validation split
- `tqdm>=4.65.0` - Progress bars
- `matplotlib>=3.7.0` - Visualization
- `pyyaml>=6.0` - Configuration files

All dependencies are listed in `requirements.txt`.

## Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── train_images/          # Training images
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── test_images/           # Test images for inference
│   ├── img201.jpg
│   ├── img202.jpg
│   └── ...
└── annotations.json       # Training annotations
```

### Annotation Format

The `annotations.json` file should follow this structure:

```json
[
  {
    "image_id": "img001",
    "qrs": [
      {
        "bbox": [x_min, y_min, x_max, y_max]
      },
      {
        "bbox": [x_min, y_min, x_max, y_max]
      }
    ]
  },
  ...
]
```

Where:
- `image_id`: Image filename without extension
- `bbox`: Bounding box in `[x_min, y_min, x_max, y_max]` format (absolute pixel coordinates)

### Creating Annotations

If you need to annotate your own images, use the provided annotation tool:

```bash
python annotate.py --images dataset/train_images --output dataset/annotations.json
```

For automatic annotation (using OpenCV's QR detector):
```bash
python annotate.py --images dataset/train_images --output dataset/annotations.json --auto
```

## Training

### Basic Training

Train a model with default settings:

```bash
python train.py --images dataset/train_images --annotations dataset/annotations.json
```

This will:
1. Preprocess the dataset and split it into train/validation sets (80/20)
2. Convert annotations to YOLO format
3. Train a YOLOv8s model for 50 epochs
4. Save the best model to `runs/qr_detection_model/weights/best.pt`


### Training Output

After training, you'll find:
- `runs/qr_detection_model/weights/best.pt` - Best model weights
- `runs/qr_detection_model/weights/last.pt` - Last epoch weights
- `runs/qr_detection_model/results.png` - Training metrics plot
- `runs/qr_detection_model/confusion_matrix.png` - Confusion matrix
- Various other training visualizations and logs

## Inference

### Basic Inference

Run inference on a folder of images:

```bash
python infer.py --input demo_images/ --output submission.json
```

This will:
1. Load the trained model
2. Process all images in the input directory
3. Detect QR codes in each image
4. Save results to `submission.json`

### Output Format

The output JSON file contains:

```json
[
  {
    "image_id": "img201",
    "qrs": [
      {
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.95
      },
      {
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.87
      }
    ]
  },
  ...
]
```

### Inference Examples

**Process test images with custom model:**
```bash
python infer.py --input dataset/test_images --output test_results.json --model runs/my_model/weights/best.pt
```

**Lower confidence threshold for more detections:**
```bash
python infer.py --input images/ --output results.json --conf 0.2
```

**Save annotated images for visualization:**
```bash
python infer.py --input images/ --output results.json --save-images --save-dir visualizations/
```

## Project Structure

```
multiqr_hackathon/
├── train.py                    # Training script
├── infer.py                    # Inference script
├── annotate.py                 # Annotation tool
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── qr_dataset.yaml            # Dataset configuration (generated)
│
├── dataset/                    # Dataset directory
│   ├── train_images/          # Training images
│   ├── test_images/           # Test images
│   └── annotations.json       # Training annotations
│
├── processed_data/            # Processed dataset (generated)
│   ├── train/
│   │   ├── images/           # Training images
│   │   └── labels/           # YOLO format labels
│   └── val/
│       ├── images/           # Validation images
│       └── labels/           # YOLO format labels
│
├── runs/                      # Training/inference outputs (generated)
│   └── qr_detection_model/
│       ├── weights/
│       │   ├── best.pt       # Best model weights
│       │   └── last.pt       # Last epoch weights
│       ├── results.png        # Training metrics
│       └── ...               # Other training outputs
│
├── qr_training.ipynb          # Training notebook (reference)
└── qr_infer_evaluation.ipynb # Inference notebook (reference)
```

## Results

### Model Performance

Our trained YOLOv8s model achieves:
- **Precision**: ~0.95
- **Recall**: ~0.92
- **mAP@0.5**: ~0.94
- **mAP@0.5:0.95**: ~0.78

### Training Time

On a Tesla T4 GPU:
- 50 epochs: ~15-20 minutes
- 100 epochs: ~30-40 minutes

### Inference Speed

- **GPU (Tesla T4)**: ~11ms per image
- **CPU**: ~100-200ms per image


## Workflow Summary

### Complete Training and Inference Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare your dataset
# - Place images in dataset/train_images/
# - Create annotations.json (or use annotate.py)

# 3. Train the model
python train.py --images dataset/train_images --annotations dataset/annotations.json --epochs 50

# 4. Run inference
python infer.py --input dataset/test_images --output submission.json

# 5. (Optional) View annotated results
python infer.py --input dataset/test_images --output results.json --save-images
```

