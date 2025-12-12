"""
CNN Training Script for Action Recognition
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import wandb
from tqdm import tqdm
import yaml
from utils.config import Config
from utils.logger import setup_logger

class ActionDataset(Dataset):
    """Dataset for action classification"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = plt.imread(image_path)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label

class ActionClassifier(nn.Module):
    """CNN model for action classification"""

    def __init__(self, num_classes=5):
        super(ActionClassifier, self).__init__()

        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)

        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def create_data_splits(data_dir, test_size=0.2, val_size=0.2, random_state=42):
    """Create train/val/test splits"""

    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    class_names = []

    # Get all class directories
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            class_names.append(class_name)
            class_idx = len(class_names) - 1

            # Get all images in class directory
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_size/(1-test_size),
        stratify=train_labels, random_state=random_state
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names

def train_cnn(config_path='config.yaml'):
    """Train CNN model"""
    config = Config(config_path)
    logger = setup_logger('cnn_training')

    # Setup MLflow
    if config.mlops_config['mlflow_tracking_uri']:
        mlflow.set_tracking_uri(config.mlops_config['mlflow_tracking_uri'])
        mlflow.set_experiment(config.mlops_config['experiment_name'] + '_cnn')

    # Setup WandB
    if config.mlops_config['wandb_project']:
        wandb.init(project=config.mlops_config['wandb_project'],
                  name='cnn_training')

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config.cnn_config)

        # Data transforms
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create data splits
        logger.info("Creating data splits...")
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names = create_data_splits(
            config.dataset_paths['action_data']
        )

        logger.info(f"Found {len(class_names)} classes: {class_names}")
        logger.info(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

        # Create datasets
        train_dataset = ActionDataset(train_paths, train_labels, train_transform)
        val_dataset = ActionDataset(val_paths, val_labels, val_transform)
        test_dataset = ActionDataset(test_paths, test_labels, val_transform)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.cnn_config['batch_size'],
            shuffle=True,
            num_workers=config.cnn_config['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.cnn_config['batch_size'],
            shuffle=False,
            num_workers=config.cnn_config['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.cnn_config['batch_size'],
            shuffle=False,
            num_workers=config.cnn_config['num_workers'],
            pin_memory=True
        )

        # Initialize model
        model = ActionClassifier(num_classes=len(class_names))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.cnn_config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Training loop
        best_val_acc = 0.0
        patience = config.cnn_config['patience']
        patience_counter = 0

        for epoch in range(config.cnn_config['epochs']):
            logger.info(f"Epoch {epoch+1}/{config.cnn_config['epochs']}")

            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in tqdm(train_loader, desc="Training"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validating"):
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch
            }, step=epoch)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save model
                model_path = Path('models/cnn_action_classifier.pth')
                model_path.parent.mkdir(exist_ok=True)
                torch.save(model.state_dict(), model_path)

                mlflow.log_artifact(str(model_path), 'models')
                logger.info("Best model saved")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping")
                    break

            scheduler.step()

        # Test best model
        logger.info("Testing best model...")
        model.load_state_dict(torch.load('models/cnn_action_classifier.pth'))
        model.eval()

        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = outputs.max(1)

                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = 100. * test_correct / test_total

        # Classification report
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        mlflow.log_metrics({
            'test_accuracy': test_acc,
            'test_precision': report['weighted avg']['precision'],
            'test_recall': report['weighted avg']['recall'],
            'test_f1': report['weighted avg']['f1-score']
        })

        logger.info(f"Test Accuracy: {test_acc:.2f}%")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        cm_path = Path('models/confusion_matrix.png')
        plt.savefig(cm_path)
        mlflow.log_artifact(str(cm_path), 'plots')

        plt.close()

        logger.info("CNN training completed")
        return model

if __name__ == '__main__':
    train_cnn()
