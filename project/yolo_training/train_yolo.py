"""
YOLOv12 Training Script for Theft & Suspicious Activity Detection
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import mlflow
import wandb
from utils.config import Config
from utils.logger import setup_logger

def create_data_yaml(config, output_path):
    """Create data.yaml for YOLO training"""
    data = {
        'path': str(config.dataset_paths['yolo_data']),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'person',
            1: 'gun',
            2: 'knife',
            3: 'fight',
            4: 'running',
            5: 'bag',
            6: 'suspicious_pose'
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def train_yolo(config_path='config.yaml'):
    """Train YOLOv12 model"""
    config = Config(config_path)
    logger = setup_logger('yolo_training')

    # Setup MLflow
    if config.mlops_config['mlflow_tracking_uri']:
        mlflow.set_tracking_uri(config.mlops_config['mlflow_tracking_uri'])
        mlflow.set_experiment(config.mlops_config['experiment_name'])

    # Setup WandB
    if config.mlops_config['wandb_project']:
        wandb.init(project=config.mlops_config['wandb_project'],
                  name='yolo_training')

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config.yolo_config)

        # Create data.yaml
        data_yaml = Path('yolo_training/data.yaml')
        create_data_yaml(config, data_yaml)

        # Initialize model
        model = YOLO('yolov8x.yaml')  # Use YOLOv8 as base, will be updated to v12 when available

        # Training
        results = model.train(
            data=str(data_yaml),
            epochs=config.yolo_config['epochs'],
            batch=config.yolo_config['batch_size'],
            imgsz=config.yolo_config['img_size'],
            lr0=config.yolo_config['lr'],
            momentum=config.yolo_config['momentum'],
            weight_decay=config.yolo_config['weight_decay'],
            warmup_epochs=config.yolo_config['warmup_epochs'],
            warmup_momentum=config.yolo_config['warmup_momentum'],
            warmup_bias_lr=config.yolo_config['warmup_bias_lr'],
            box=config.yolo_config['box'],
            cls=config.yolo_config['cls'],
            dfl=config.yolo_config['dfl'],
            pose=config.yolo_config['pose'],
            kobj=config.yolo_config['kobj'],
            label_smoothing=config.yolo_config['label_smoothing'],
            nbs=config.yolo_config['nbs'],
            hsv_h=config.yolo_config['hsv_h'],
            hsv_s=config.yolo_config['hsv_s'],
            hsv_v=config.yolo_config['hsv_v'],
            degrees=config.yolo_config['degrees'],
            translate=config.yolo_config['translate'],
            scale=config.yolo_config['scale'],
            shear=config.yolo_config['shear'],
            perspective=config.yolo_config['perspective'],
            flipud=config.yolo_config['flipud'],
            fliplr=config.yolo_config['fliplr'],
            mosaic=config.yolo_config['mosaic'],
            mixup=config.yolo_config['mixup'],
            copy_paste=config.yolo_config['copy_paste'],
            project='models',
            name='yolov12_security',
            save=True,
            save_period=10,
            cache=False,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=8,
            pretrained=True,
            optimizer='SGD',
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=False,
            close_mosaic=10,
            resume=False,
            amp=True,  # Mixed precision
            fraction=1.0,
            profile=False,
            freeze=None,
            multi_scale=False,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            split='val',
            save_json=False,
            save_hybrid=False,
            conf=None,
            iou=0.7,
            max_det=300,
            half=False,
            dnn=False,
            plots=True,
            source=None,
            vid_stride=1,
            stream_buffer=False,
            visualize=False,
            augment=False,
            agnostic_nms=False,
            classes=None,
            retina_masks=False,
            embed=None,
            show=False,
            save_frames=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            show_labels=True,
            show_conf=True,
            show_boxes=True,
            line_width=None
        )

        # Log metrics
        metrics = results.results_dict
        mlflow.log_metrics({
            'mAP50': metrics.get('metrics/mAP50(B)', 0),
            'mAP50-95': metrics.get('metrics/mAP50-95(B)', 0),
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0)
        })

        # Log model
        mlflow.pytorch.log_model(model.model, 'yolo_model')

        # Save best model
        best_model_path = Path('models/yolov12_security/weights/best.pt')
        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), 'models')

        logger.info("YOLO training completed")
        return model

if __name__ == '__main__':
    train_yolo()
