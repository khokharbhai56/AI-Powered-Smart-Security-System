"""
Main entry point for AI Security System
"""

import argparse
import sys
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from utils.config import Config
from utils.logger import setup_logger

def run_training(args):
    """Run training pipeline"""
    logger = setup_logger('training')

    if args.model == 'yolo':
        logger.info("Starting YOLO training...")
        from yolo_training.train_yolo import train_yolo
        train_yolo(args.config)

    elif args.model == 'mask_rcnn':
        logger.info("Starting Mask R-CNN training...")
        from mask_rcnn_training.train_mask_rcnn import train_mask_rcnn
        train_mask_rcnn(args.config)

    elif args.model == 'cnn':
        logger.info("Starting CNN training...")
        from cnn_classifier.train_cnn import train_cnn
        train_cnn(args.config)

    elif args.model == 'all':
        logger.info("Starting full training pipeline...")

        # Train YOLO
        logger.info("Training YOLO...")
        from yolo_training.train_yolo import train_yolo
        yolo_model = train_yolo(args.config)

        # Train Mask R-CNN
        logger.info("Training Mask R-CNN...")
        from mask_rcnn_training.train_mask_rcnn import train_mask_rcnn
        mask_rcnn_model = train_mask_rcnn(args.config)

        # Train CNN
        logger.info("Training CNN...")
        from cnn_classifier.train_cnn import train_cnn
        cnn_model = train_cnn(args.config)

        logger.info("Full training pipeline completed")

    else:
        logger.error(f"Unknown model: {args.model}")
        sys.exit(1)

def run_inference(args):
    """Run inference pipeline"""
    logger = setup_logger('inference')

    logger.info("Starting detection pipeline...")
    from real_time_system.detection_pipeline import DetectionPipeline

    pipeline = DetectionPipeline(args.config)
    pipeline.run(args.source)

def run_dashboard(args):
    """Run dashboard"""
    logger = setup_logger('dashboard')

    logger.info("Starting dashboard...")
    import streamlit as st
    from dashboard.app import main

    # Run streamlit app
    import subprocess
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'dashboard/app.py'])

def run_dataset_preparation(args):
    """Run dataset preparation"""
    logger = setup_logger('dataset_prep')

    logger.info("Starting dataset preparation...")
    from datasets.extract_frames import extract_frames_recursive

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    extract_frames_recursive(
        input_dir=input_dir,
        output_dir=output_dir,
        fps=args.fps,
        max_workers=args.max_workers
    )

    logger.info("Dataset preparation completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Security System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', type=str, required=True,
                             choices=['yolo', 'mask_rcnn', 'cnn', 'all'],
                             help='Model to train')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--source', type=str, default=0,
                             help='Video source (0 for webcam, path for video file, or RTSP URL)')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run dashboard')

    # Dataset preparation command
    dataset_parser = subparsers.add_parser('prepare_dataset', help='Prepare dataset')
    dataset_parser.add_argument('--input_dir', type=str, required=True,
                               help='Input directory containing videos')
    dataset_parser.add_argument('--output_dir', type=str, required=True,
                               help='Output directory for extracted frames')
    dataset_parser.add_argument('--fps', type=float, default=1.0,
                               help='Frames per second to extract')
    dataset_parser.add_argument('--max_workers', type=int, default=4,
                               help='Maximum number of worker threads')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    # Run command
    if args.command == 'train':
        run_training(args)
    elif args.command == 'infer':
        run_inference(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    elif args.command == 'prepare_dataset':
        run_dataset_preparation(args)

if __name__ == '__main__':
    main()
