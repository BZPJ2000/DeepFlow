"""
Configuration Manager Module

This module provides configuration management for experiments,
including saving, loading, and validating configurations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings


class ConfigManager:
    """
    Configuration manager for experiment settings.
    
    This class provides methods to:
    - Save experiment configurations
    - Load saved configurations
    - Validate configuration parameters
    - Export configurations to various formats
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_config = {}
    
    def save_config(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        Save a configuration to a JSON file.
        
        Args:
            config: Configuration dictionary to save
            name: Optional name for the configuration file
            
        Returns:
            Path to the saved configuration file
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"config_{timestamp}"
        
        filename = f"{name}.json"
        filepath = self.config_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            self.current_config = config
            return str(filepath)
            
        except Exception as e:
            warnings.warn(f"Error saving configuration: {e}")
            return ""
    
    def load_config(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Configuration dictionary or None if loading fails
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.current_config = config
            return config
            
        except FileNotFoundError:
            warnings.warn(f"Configuration file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            warnings.warn(f"Invalid JSON in configuration file: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Error loading configuration: {e}")
            return None
    
    def list_configs(self) -> List[Dict[str, Any]]:
        """
        List all available configurations.
        
        Returns:
            List of configuration metadata
        """
        configs = []
        
        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                configs.append({
                    'filename': config_file.name,
                    'filepath': str(config_file),
                    'name': config.get('name', 'Unnamed'),
                    'created': datetime.fromtimestamp(config_file.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    'model': config.get('model', {}).get('name', 'N/A'),
                    'dataset': config.get('dataset', {}).get('name', 'N/A')
                })
            except Exception as e:
                warnings.warn(f"Error reading config file {config_file}: {e}")
        
        return sorted(configs, key=lambda x: x['created'], reverse=True)
    
    def delete_config(self, filepath: str) -> bool:
        """
        Delete a configuration file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            Path(filepath).unlink()
            return True
        except Exception as e:
            warnings.warn(f"Error deleting configuration: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        errors = []
        warnings_list = []
        
        # Validate model configuration
        if 'model' in config:
            model_config = config['model']
            if 'name' not in model_config:
                errors.append("Model configuration missing 'name' field")
            if 'category' not in model_config:
                warnings_list.append("Model configuration missing 'category' field")
        
        # Validate dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            if 'name' not in dataset_config:
                errors.append("Dataset configuration missing 'name' field")
            if 'path' not in dataset_config:
                warnings_list.append("Dataset configuration missing 'path' field")
        
        # Validate training configuration
        if 'training' in config:
            training_config = config['training']
            if 'epochs' in training_config and training_config['epochs'] <= 0:
                errors.append("Number of epochs must be positive")
            if 'batch_size' in training_config and training_config['batch_size'] <= 0:
                errors.append("Batch size must be positive")
            if 'learning_rate' in training_config and training_config['learning_rate'] <= 0:
                errors.append("Learning rate must be positive")
        
        return {
            'errors': errors,
            'warnings': warnings_list
        }
    
    def create_experiment_config(
        self,
        model_name: str,
        model_category: str,
        model_params: Dict[str, Any],
        dataset_name: str,
        dataset_category: str,
        dataset_params: Dict[str, Any],
        training_params: Dict[str, Any],
        evaluation_metrics: List[str],
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a complete experiment configuration.
        
        Args:
            model_name: Name of the model
            model_category: Category of the model
            model_params: Model parameters
            dataset_name: Name of the dataset
            dataset_category: Category of the dataset
            dataset_params: Dataset parameters
            training_params: Training parameters
            evaluation_metrics: List of evaluation metrics
            name: Optional name for the experiment
            
        Returns:
            Complete configuration dictionary
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"experiment_{timestamp}"
        
        config = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'model': {
                'name': model_name,
                'category': model_category,
                'parameters': model_params
            },
            'dataset': {
                'name': dataset_name,
                'category': dataset_category,
                'parameters': dataset_params
            },
            'training': training_params,
            'evaluation': {
                'metrics': evaluation_metrics
            }
        }
        
        return config
    
    def export_to_python(self, config: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Export configuration to a Python script.
        
        Args:
            config: Configuration dictionary
            filepath: Optional path for the output file
            
        Returns:
            Path to the exported file
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.config_dir / f"config_{timestamp}.py"
        
        try:
            python_code = self._config_to_python(config)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(python_code)
            
            return str(filepath)
            
        except Exception as e:
            warnings.warn(f"Error exporting configuration to Python: {e}")
            return ""
    
    def _config_to_python(self, config: Dict[str, Any]) -> str:
        """
        Convert configuration dictionary to Python code.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Python code string
        """
        code = f'''"""
Experiment Configuration
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        code += f"""
# Model Configuration
model_name = "{config.get('model', {}).get('name', 'N/A')}"
model_category = "{config.get('model', {}).get('category', 'N/A')}"
model_params = {json.dumps(config.get('model', {}).get('parameters', {}), indent=4)}

# Dataset Configuration
dataset_name = "{config.get('dataset', {}).get('name', 'N/A')}"
dataset_category = "{config.get('dataset', {}).get('category', 'N/A')}"
dataset_params = {json.dumps(config.get('dataset', {}).get('parameters', {}), indent=4)}

# Training Configuration
training_params = {json.dumps(config.get('training', {}), indent=4)}

# Evaluation Configuration
evaluation_metrics = {json.dumps(config.get('evaluation', {}).get('metrics', []), indent=4)}
"""
        return code
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.current_config.copy()
    
    def set_current_config(self, config: Dict[str, Any]):
        """
        Set the current configuration.
        
        Args:
            config: Configuration dictionary to set as current
        """
        self.current_config = config
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_config_template(self, experiment_type: str = "classification") -> Dict[str, Any]:
        """
        Get a template configuration for a given experiment type.
        
        Args:
            experiment_type: Type of experiment (classification, detection, segmentation, etc.)
            
        Returns:
            Template configuration dictionary
        """
        templates = {
            "classification": {
                "model": {
                    "name": "ResNet",
                    "category": "ClassicNetwork",
                    "parameters": {
                        "depth": 18,
                        "num_classes": 10,
                        "pretrained": True
                    }
                },
                "dataset": {
                    "name": "CIFAR10",
                    "category": "ImageClassification",
                    "parameters": {
                        "batch_size": 32,
                        "train_split": 0.8,
                        "val_split": 0.1,
                        "test_split": 0.1
                    }
                },
                "training": {
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "optimizer": "Adam",
                    "loss_function": "CrossEntropyLoss"
                },
                "evaluation": {
                    "metrics": ["accuracy", "precision", "recall", "f1_score"]
                }
            },
            "segmentation": {
                "model": {
                    "name": "UNet",
                    "category": "SemanticSegmentation",
                    "parameters": {
                        "encoder_depth": 5,
                        "decoder_channels": [64, 128, 256, 512],
                        "num_classes": 21
                    }
                },
                "dataset": {
                    "name": "PascalVOC",
                    "category": "SemanticSegmentation",
                    "parameters": {
                        "batch_size": 16,
                        "image_size": (256, 256),
                        "augmentation": True
                    }
                },
                "training": {
                    "epochs": 200,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.00001,
                    "optimizer": "Adam",
                    "loss_function": "CrossEntropyLoss"
                },
                "evaluation": {
                    "metrics": ["iou", "pixel_accuracy", "dice_coefficient"]
                }
            }
        }
        
        return templates.get(experiment_type, templates["classification"])