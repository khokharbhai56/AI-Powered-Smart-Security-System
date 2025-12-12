# Complete Setup & Model Training Guide

## Quick Start - Enable All Models

### Step 1: Auto-Initialize (Automatic)
The dashboard now automatically initializes the pipeline on startup. All models will be loaded automatically:
- ✅ YOLO Model
- ✅ CNN Model  
- ✅ DeepSort Tracker (if available)
- ✅ Mask R-CNN (if detectron2 installed)

### Step 2: Train CNN on Kaggle Datasets

#### Option A: Download from Kaggle (Recommended)

1. **Setup Kaggle Credentials:**
   ```bash
   # Windows: Create %USERPROFILE%\.kaggle\kaggle.json
   # Get token from: https://www.kaggle.com/settings/account
   ```

2. **Download Datasets:**
   ```bash
   cd project
   python setup_kaggle_datasets.py
   ```

3. **Train CNN:**
   ```bash
   python cnn_classifier/train_cnn_simple.py \
     --data-dir data/action_dataset \
     --epochs 20 \
     --batch-size 16
   ```

#### Option B: Use Dummy Data (Fast Testing)

```bash
cd project
python cnn_classifier/train_cnn_simple.py \
  --data-dir data/action_dataset \
  --epochs 10 \
  --batch-size 8
```

This creates random dummy data and trains a model in ~5 minutes.

### Step 3: Train YOLO on Custom Dataset

```bash
cd project/yolo_training
python train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --epochs 100 \
  --batch-size 16
```

### Step 4: Verify Models Loaded

The dashboard will show all models as loaded:
```
Model Status Dashboard:
✅ YOLO Model: Loaded
✅ CNN Model: Loaded
✅ DeepSort Tracker: Loaded (or Disabled)
✅ Mask R-CNN: Loaded (or Not Installed)
```

## Complete Dataset Paths

```
project/
├── data/
│   ├── yolo_dataset/          # For YOLO training
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── data.yaml          # YOLO dataset config
│   │
│   ├── action_dataset/        # For CNN training
│   │   ├── train/
│   │   │   ├── fighting/
│   │   │   ├── falling/
│   │   │   ├── running/
│   │   │   ├── stealing/
│   │   │   └── normal_walking/
│   │   └── val/
│   │       ├── fighting/
│   │       ├── falling/
│   │       ├── running/
│   │       ├── stealing/
│   │       └── normal_walking/
│   │
│   └── videos/                # Test videos
│       └── security_footage.mp4
│
├── models/
│   ├── yolov12_security/
│   │   └── weights/
│   │       └── best.pt        # Your YOLO model
│   └── cnn_action_classifier.pth  # Your CNN model
│
└── logs/
    └── security_system.log
```

## Kaggle Datasets to Use

### 1. Security & Violence Detection
- `tawsifur/computer-vision-complete-data` - General CV dataset
- `mohamedmustafa/violence-detection-dataset` - Violence/fighting detection
- `harshvardhangada/violence-recognition` - Video-based violence
- `astraea/violence-videos` - Raw violence videos

### 2. Action Recognition
- `meetnagpal/human-action-recognition-dataset` - UCF101 style
- `mostafamohajeri/human-activity-recognition-dataset` - HAR dataset
- `kingabzpro/action-recognition-v1` - Multi-action dataset

### 3. Object Detection (For YOLO)
- `arjunprasadsarkhel/2m-objects-detection-dataset` - General objects
- `ultralytics/coco128` - COCO128 (minimal COCO subset)
- `snehitvaddi/weapons-detection-dataset` - Weapons detection

## Complete Training Commands

### Train Everything From Scratch

```bash
#!/bin/bash
cd project

# Step 1: Download datasets
echo "Downloading Kaggle datasets..."
python setup_kaggle_datasets.py

# Step 2: Prepare YOLO dataset
echo "Training YOLO model..."
python yolo_training/train_yolo.py \
  --data data/yolo_dataset/data.yaml \
  --epochs 100 \
  --batch-size 16 \
  --img 640

# Step 3: Train CNN classifier
echo "Training CNN model..."
python cnn_classifier/train_cnn_simple.py \
  --data-dir data/action_dataset \
  --epochs 50 \
  --batch-size 32

# Step 4: Start dashboard
echo "Starting dashboard..."
streamlit run dashboard/app.py
```

## Model Files Location

After training, models will be saved here:

```
models/
├── yolov12_security/
│   └── weights/
│       └── best.pt                 ← YOLO model (100+ MB)
│
└── cnn_action_classifier.pth       ← CNN model (50-100 MB)
```

These are automatically loaded by the system on startup.

## Training Status Monitoring

### CNN Training Output Example
```
Epoch 1: Train Loss=2.3421, Acc=22.33% | Val Loss=2.1234, Acc=28.45%
Epoch 2: Train Loss=1.9876, Acc=35.67% | Val Loss=1.8765, Acc=40.12%
✓ Saved model: models/cnn_action_classifier.pth (Acc: 40.12%)
```

### YOLO Training Output Example
```
     Epoch   gpu_mem       box       obj       cls     total      labels  img_size
      1/100      2.3G    0.8234    0.9123    0.5612    2.297        32        640
      2/100      2.3G    0.7234    0.8123    0.4612    2.097        32        640
✓ Model saved to: models/yolov12_security/weights/best.pt
```

## Troubleshooting

### Models Not Loading
```python
# Check which models loaded:
from real_time_system.detection_pipeline import DetectionPipeline
p = DetectionPipeline()

print(f"YOLO: {p.yolo_model is not None}")      # Should be True
print(f"CNN: {p.cnn_model is not None}")        # Should be True  
print(f"Tracker: {p.tracker is not None}")      # May be False
print(f"Mask R-CNN: {p.mask_rcnn_predictor is not None}")  # May be False
```

### Kaggle Download Issues
1. Verify credentials at `~/.kaggle/kaggle.json`
2. Check dataset exists: `kaggle datasets list | grep violence`
3. Manually download from kaggle.com and extract to `data/`

### Out of Memory
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--img 416` (for YOLO)
- Use smaller model for YOLO: `yolov8n.pt` instead of `yolov8x.pt`

## Next Steps

1. ✅ Dashboard shows all models loaded
2. ⏳ Download dataset using `setup_kaggle_datasets.py`
3. ⏳ Train CNN: `python cnn_classifier/train_cnn_simple.py`
4. ⏳ Train YOLO: `python yolo_training/train_yolo.py`
5. ⏳ Test with real videos in dashboard
6. ⏳ Deploy to production

## Support

For issues:
1. Check logs: `logs/security_system.log`
2. Verify dataset structure
3. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Update dependencies: `pip install -r requirements.txt`
