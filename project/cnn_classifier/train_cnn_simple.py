#!/usr/bin/env python3
"""
Complete CNN Training Script with Kaggle Dataset Support
Trains CNN for action classification with automatic model saving
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger('cnn_training')


class SimpleActionDataset(Dataset):
    """Simple dataset for action images"""
    
    CLASS_NAMES = ['fighting', 'falling', 'running', 'stealing', 'normal_walking']
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        
        self._load_samples()
    
    def _load_samples(self):
        """Load images from class subdirectories"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = split_dir / class_name
            
            if class_dir.exists():
                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), class_idx))
            else:
                logger.debug(f"Class directory not found: {class_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), 0


class SimpleCNN(nn.Module):
    """Efficient CNN for action classification"""
    
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_dummy_data(data_dir, split='train', num_per_class=50):
    """Create dummy dataset for testing"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    split_dir = data_dir / split
    split_dir.mkdir(exist_ok=True)
    
    class_names = ['fighting', 'falling', 'running', 'stealing', 'normal_walking']
    
    logger.info(f"Creating {split} dummy dataset...")
    for class_name in class_names:
        class_dir = split_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(num_per_class):
            # Create random image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f'{class_name}_{i:04d}.jpg')


def train_cnn(args):
    """Train CNN model"""
    
    logger.info("=" * 70)
    logger.info("CNN Action Classification Training")
    logger.info("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data if needed
    if not (data_dir / 'train').exists():
        logger.info("Creating dummy training dataset...")
        create_dummy_data(data_dir, 'train', num_per_class=50)
        create_dummy_data(data_dir, 'val', num_per_class=10)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = SimpleActionDataset(data_dir, 'train', train_transform)
    val_dataset = SimpleActionDataset(data_dir, 'val', val_transform)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        logger.error("No training data!")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    logger.info("Creating model...")
    model = SimpleCNN(num_classes=5).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0
    patience = 0
    
    logger.info(f"Training for {args.epochs} epochs...")
    logger.info("-" * 70)
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_correct += pred.eq(labels).sum().item()
            train_total += labels.size(0)
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                   f"Acc={train_acc:.2f}% | Val Loss={val_loss/len(val_loader):.4f}, "
                   f"Acc={val_acc:.2f}%")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / 'cnn_action_classifier.pth'
            torch.save(model, str(model_path))
            logger.info(f"✓ Saved model: {model_path} (Acc: {val_acc:.2f}%)")
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info("-" * 70)
    logger.info(f"✓ Training complete! Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/action_dataset')
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    
    args = parser.parse_args()
    train_cnn(args)
