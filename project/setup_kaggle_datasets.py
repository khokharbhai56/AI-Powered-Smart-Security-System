#!/usr/bin/env python3
"""
Download and setup datasets from Kaggle for training
"""

import os
import sys
from pathlib import Path
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def install_kaggle():
    """Install kaggle CLI if not already installed"""
    try:
        import kaggle
        print("✓ Kaggle CLI already installed")
    except ImportError:
        print("Installing Kaggle CLI...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✓ Kaggle CLI installed")

def setup_kaggle_credentials():
    """Setup Kaggle credentials"""
    credentials_path = Path.home() / ".kaggle" / "kaggle.json"
    
    if credentials_path.exists():
        print("✓ Kaggle credentials found")
        return True
    
    print("\n⚠ Kaggle credentials not found!")
    print("Follow these steps:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Click 'Create New Token' to download kaggle.json")
    print(f"3. Place it at: {credentials_path}")
    print("4. Run this script again")
    
    return False

def download_dataset(dataset_name, output_path):
    """Download dataset from Kaggle"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading {dataset_name}...")
    try:
        subprocess.check_call([
            "kaggle", "datasets", "download", "-d", dataset_name, "-p", str(output_path)
        ])
        print(f"✓ Downloaded {dataset_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")
        return False

def main():
    print("=" * 60)
    print("AI Security System - Kaggle Dataset Setup")
    print("=" * 60)
    
    # Install kaggle CLI
    install_kaggle()
    
    # Setup credentials
    if not setup_kaggle_credentials():
        print("\nPlease setup Kaggle credentials and run again.")
        return
    
    project_path = Path(__file__).parent
    
    # Dataset configurations
    datasets = {
        'YOLO Security': {
            'kaggle': 'tawsifur/computer-vision-complete-data',
            'path': project_path / 'data' / 'yolo_dataset',
            'description': 'COCO-style dataset for object detection'
        },
        'Action Recognition': {
            'kaggle': 'meetnagpal/human-action-recognition-dataset',
            'path': project_path / 'data' / 'action_dataset',
            'description': 'Dataset for action/activity classification'
        },
        'Violence Detection': {
            'kaggle': 'mohamedmustafa/violence-detection-dataset',
            'path': project_path / 'data' / 'violence_dataset',
            'description': 'Videos and frames for violence detection'
        }
    }
    
    print("\nAvailable Datasets:")
    print("-" * 60)
    for i, (name, info) in enumerate(datasets.items(), 1):
        print(f"{i}. {name}")
        print(f"   Description: {info['description']}")
        print(f"   Kaggle ID: {info['kaggle']}")
        print()
    
    print("Download Options:")
    print("1. Download all datasets")
    print("2. Download specific dataset (select number)")
    print("3. Skip download")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        for name, info in datasets.items():
            download_dataset(info['kaggle'], info['path'])
    elif choice == '2':
        dataset_choice = input("Enter dataset number (1-3): ").strip()
        dataset_list = list(datasets.items())
        if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(dataset_list):
            name, info = dataset_list[int(dataset_choice) - 1]
            download_dataset(info['kaggle'], info['path'])
    elif choice == '3':
        print("Skipped dataset download")
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Extract downloaded datasets if needed")
    print("2. Run training scripts:")
    print("   - python yolo_training/train_yolo.py")
    print("   - python cnn_classifier/train_cnn.py")

if __name__ == "__main__":
    main()
