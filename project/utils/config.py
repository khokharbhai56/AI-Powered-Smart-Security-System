"""
Configuration management for AI Security System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager"""

    def __init__(self, config_path: str = 'config.yaml'):
        # Convert to Path object
        config_path_obj = Path(config_path)
        
        # If relative path, check multiple locations
        if not config_path_obj.is_absolute():
            # List of places to search for config.yaml
            search_paths = [
                Path(config_path),  # Current directory
                Path(__file__).parent.parent / config_path,  # Project root (utils/../config.yaml)
                Path.cwd() / config_path,  # Working directory
            ]
            
            # Find the first existing config file
            for path in search_paths:
                if path.exists():
                    config_path_obj = path
                    break
        
        self.config_path = config_path_obj
        self._config = {}

        # Load environment variables
        load_dotenv()

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # Override with environment variables
        self._override_from_env()

    def _override_from_env(self):
        """Override configuration with environment variables"""
        # Dataset paths
        if os.getenv('DATASET_YOLO_PATH'):
            self._config['dataset_paths']['yolo_data'] = os.getenv('DATASET_YOLO_PATH')
        if os.getenv('DATASET_COCO_PATH'):
            self._config['dataset_paths']['coco'] = os.getenv('DATASET_COCO_PATH')
        if os.getenv('DATASET_ACTION_PATH'):
            self._config['dataset_paths']['action_data'] = os.getenv('DATASET_ACTION_PATH')

        # Model paths
        if os.getenv('MODEL_YOLO_PATH'):
            self._config['model_paths']['yolo_model_path'] = os.getenv('MODEL_YOLO_PATH')
        if os.getenv('MODEL_MASK_RCNN_PATH'):
            self._config['model_paths']['mask_rcnn_model_path'] = os.getenv('MODEL_MASK_RCNN_PATH')
        if os.getenv('MODEL_CNN_PATH'):
            self._config['model_paths']['cnn_model_path'] = os.getenv('MODEL_CNN_PATH')

        # MLOps
        if os.getenv('MLFLOW_TRACKING_URI'):
            self._config['mlops_config']['mlflow_tracking_uri'] = os.getenv('MLFLOW_TRACKING_URI')
        if os.getenv('WANDB_PROJECT'):
            self._config['mlops_config']['wandb_project'] = os.getenv('WANDB_PROJECT')

        # Email settings
        if os.getenv('SMTP_SERVER'):
            self._config['alert_config']['smtp_server'] = os.getenv('SMTP_SERVER')
        if os.getenv('SMTP_PORT'):
            self._config['alert_config']['smtp_port'] = int(os.getenv('SMTP_PORT'))
        if os.getenv('EMAIL_USER'):
            self._config['alert_config']['email_user'] = os.getenv('EMAIL_USER')
        if os.getenv('EMAIL_PASSWORD'):
            self._config['alert_config']['email_password'] = os.getenv('EMAIL_PASSWORD')

    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path

        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    # Property accessors for common configurations
    @property
    def dataset_paths(self) -> Dict[str, str]:
        return self._config.get('dataset_paths', {})

    @property
    def model_paths(self) -> Dict[str, str]:
        return self._config.get('model_paths', {})

    @property
    def yolo_config(self) -> Dict[str, Any]:
        return self._config.get('yolo_config', {})

    @property
    def mask_rcnn_config(self) -> Dict[str, Any]:
        return self._config.get('mask_rcnn_config', {})

    @property
    def cnn_config(self) -> Dict[str, Any]:
        return self._config.get('cnn_config', {})

    @property
    def detection_config(self) -> Dict[str, Any]:
        return self._config.get('detection_config', {})

    @property
    def alert_config(self) -> Dict[str, Any]:
        return self._config.get('alert_config', {})

    @property
    def mlops_config(self) -> Dict[str, Any]:
        return self._config.get('mlops_config', {})

    @property
    def logging_config(self) -> Dict[str, Any]:
        return self._config.get('logging_config', {})

    @property
    def system_config(self) -> Dict[str, Any]:
        return self._config.get('system_config', {})

    def validate_config(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'dataset_paths',
            'model_paths',
            'yolo_config',
            'mask_rcnn_config',
            'cnn_config',
            'detection_config',
            'alert_config'
        ]

        for key in required_keys:
            if key not in self._config:
                print(f"Missing required configuration key: {key}")
                return False

        # Validate dataset paths exist if specified
        for path_key, path_value in self.dataset_paths.items():
            if path_value and not Path(path_value).exists():
                print(f"Warning: Dataset path does not exist: {path_value}")

        return True

    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print("=" * 50)
        for section, values in self._config.items():
            print(f"\n{section.upper()}:")
            if isinstance(values, dict):
                for key, value in values.items():
                    if 'password' in key.lower():
                        print(f"  {key}: ***")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {values}")
        print("=" * 50)
