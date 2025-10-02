# Quick Start Guide

Get started with QR Code Detection in 5 minutes! 🚀

## 🏃 Quick Setup (3 steps)

### Step 1: Install Dependencies

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Or use the setup scripts:
- Windows: `setup.bat`
- Linux/Mac: `bash setup.sh`

### Step 2: Prepare Dataset

Create this structure:
```
dataset/
├── train_images/       # Put your training images here
│   ├── img001.jpg
│   └── ...
└── annotations.json    # Create annotations
```

Create `annotations.json`:
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [10, 20, 100, 110]}
    ]
  }
]
```

### Step 3: Train & Run

```bash
# Train model (15-20 min on GPU)
python train.py

# Run inference
python infer.py --input dataset/test_images --output submission.json
```

Done! 🎉

## 📊 Expected Results

After training, you should see:
- ✅ Model saved to: `runs/qr_detection_model/weights/best.pt`
- ✅ Training metrics: `runs/qr_detection_model/results.png`
- ✅ mAP@0.5: ~0.94

## 🎯 Common Commands

```bash
# Basic training
python train.py

# Training with custom epochs
python train.py --epochs 100

# Training with smaller batch (if GPU memory is limited)
python train.py --batch 8

# Basic inference
python infer.py --input test_images/ --output results.json

# Inference with visualization
python infer.py --input test_images/ --output results.json --save-images

# Inference with custom confidence threshold
python infer.py --input test_images/ --output results.json --conf 0.3
```

## 🔍 Verify Installation

Test if everything is installed correctly:

```bash
python -c "import ultralytics; print('Ultralytics version:', ultralytics.__version__)"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

Expected output:
```
Ultralytics version: 8.x.x
OpenCV version: 4.x.x
```

## 🐛 Quick Troubleshooting

**Problem: CUDA not available**
- Solution: Install CPU version or use smaller batch size
- Command: `python train.py --batch 4`

**Problem: Module not found**
- Solution: `pip install -r requirements.txt`

**Problem: Model not found during inference**
- Solution: Train first with `python train.py`

**Problem: No images found**
- Solution: Check image extensions (.jpg, .jpeg, .png)

## 📚 Full Documentation

See [README.md](README.md) for complete documentation.

## 🎓 Tutorial

### Complete Example

```bash
# 1. Setup (one time)
pip install -r requirements.txt

# 2. Organize your data
# dataset/train_images/ -> your training images
# dataset/annotations.json -> your annotations

# 3. Train
python train.py --images dataset/train_images --annotations dataset/annotations.json --epochs 50

# 4. Test
python infer.py --input dataset/test_images --output submission.json

# 5. View results
cat submission.json
```

## 💡 Pro Tips

1. **Start small**: Test with 10-20 images first
2. **Use GPU**: Training is 10x faster on GPU
3. **Visualize**: Use `--save-images` to see detection results
4. **Tune confidence**: Lower `--conf` if missing detections
5. **More epochs**: Train longer for better accuracy

## 🎉 You're Ready!

You now have everything you need to detect QR codes. Happy coding! 📱✨

---

Need help? Check the [README.md](README.md) or open an issue.

