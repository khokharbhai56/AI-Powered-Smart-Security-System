# AI-Powered Smart Security System

An advanced AI-powered security system that combines multiple computer vision models for comprehensive surveillance and threat detection.

## Features

- **Multi-Model Detection Pipeline**: Combines YOLOv12, Mask R-CNN, and CNN for object detection, person segmentation, and action recognition
- **Real-time Processing**: Optimized for real-time video analysis with DeepSort tracking
- **Intelligent Alert System**: Multi-channel alerts (email, audio, popup) with configurable rules
- **Interactive Dashboard**: Streamlit-based monitoring dashboard with real-time metrics
- **MLOps Integration**: MLflow and Weights & Biases integration for experiment tracking
- **Comprehensive Logging**: Detailed logging and database storage of detections and alerts

## Architecture

```
├── YOLOv12          - Object detection (weapons, suspicious objects)
├── Mask R-CNN       - Person segmentation and pose analysis
├── CNN Classifier   - Action recognition (fighting, running, etc.)
├── DeepSort         - Multi-object tracking
├── Rule Engine      - Suspicious activity detection
└── Alert System     - Multi-channel notifications
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/ai-security-system.git
cd ai-security-system
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create necessary directories:
```bash
./create_dirs.bat  # On Windows
# or manually create directories
```

## Configuration

Edit `config.yaml` to configure:
- Dataset paths
- Model parameters
- Alert settings
- MLOps tracking URIs

## Usage

### Training Models

Train individual models:
```bash
# Train YOLO model
python run.py train --model yolo

# Train Mask R-CNN
python run.py train --model mask_rcnn

# Train CNN classifier
python run.py train --model cnn

# Train all models
python run.py train --model all
```

### Dataset Preparation

Extract frames from videos for training:
```bash
python run.py prepare_dataset --input_dir /path/to/videos --output_dir /path/to/frames --fps 1.0
```

### Real-time Detection

Run the detection pipeline:
```bash
# Webcam
python run.py infer --source 0

# Video file
python run.py infer --source /path/to/video.mp4

# RTSP stream
python run.py infer --source rtsp://your_stream_url
```

### Monitoring Dashboard

Launch the Streamlit dashboard:
```bash
python run.py dashboard
```

## Project Structure

```
project/
├── config.yaml                 # Configuration file
├── run.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   ├── config.py              # Configuration management
│   └── logger.py              # Logging utilities
├── datasets/
│   └── extract_frames.py      # Video frame extraction
├── yolo_training/
│   └── train_yolo.py          # YOLO training script
├── mask_rcnn_training/
│   └── train_mask_rcnn.py     # Mask R-CNN training script
├── cnn_classifier/
│   └── train_cnn.py           # CNN training script
├── real_time_system/
│   └── detection_pipeline.py  # Real-time detection pipeline
├── dashboard/
│   └── app.py                 # Streamlit monitoring dashboard
├── models/                    # Trained models
├── logs/                      # Log files
├── screenshots/               # Alert screenshots
└── data/                      # Datasets and databases
```

## Models and Datasets

### Required Models
- YOLOv12 (Ultralytics YOLO)
- Mask R-CNN (Detectron2)
- ResNet50 (PyTorch)

### Datasets
- COCO dataset for person detection/segmentation
- Custom action recognition dataset
- Security video datasets

## Alert System

The system supports multiple alert channels:

- **Email Alerts**: SMTP-based email notifications
- **Audio Alerts**: Sound notifications using pygame
- **Popup Alerts**: Desktop notifications
- **Database Logging**: SQLite-based event logging

## MLOps Integration

- **MLflow**: Experiment tracking and model registry
- **Weights & Biases**: Advanced experiment monitoring and visualization

## Performance

- Real-time processing at 30+ FPS
- Multi-object tracking with DeepSort
- Configurable detection confidence thresholds
- GPU acceleration support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```
@software{ai_security_system,
  title={AI-Powered Smart Security System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ai-security-system}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review the configuration examples

## Roadmap

- [ ] Add support for additional camera types
- [ ] Implement edge deployment
- [ ] Add facial recognition module
- [ ] Integrate with existing security systems
- [ ] Add mobile app for remote monitoring
