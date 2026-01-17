"""
Dynamic Loader Module

This module provides dynamic loading capabilities for models, datasets, and functions
from the integrated deep learning framework without code repetition.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable
import warnings


class DynamicLoader:
    """
    Dynamic loader for discovering and loading classes and functions from the framework.
    
    This class provides methods to:
    - Scan directories for Python modules
    - Discover classes and functions in modules
    - Dynamically import and instantiate classes
    - Load functions with parameters
    """
    
    def __init__(self, framework_path: Optional[str] = None):
        """
        Initialize the dynamic loader.
        
        Args:
            framework_path: Path to the deep learning framework
        """
        self.framework_path = framework_path or r"E:\Projects\Learning_space\2025_learn\torch-template-for-deep-learning-main"
        self._cache = {}
        self._module_cache = {}
        
        # Add framework path to Python path if not already present
        if self.framework_path not in sys.path:
            sys.path.insert(0, self.framework_path)
    
    def scan_directory(self, directory: str, pattern: str = "*.py") -> List[str]:
        """
        Scan a directory for Python files.
        
        Args:
            directory: Directory path to scan
            pattern: File pattern to match (default: "*.py")
            
        Returns:
            List of file paths matching the pattern
        """
        try:
            dir_path = Path(self.framework_path) / directory
            if not dir_path.exists():
                warnings.warn(f"Directory not found: {dir_path}")
                return []
            
            files = list(dir_path.glob(pattern))
            return [str(f) for f in files if f.is_file() and not f.name.startswith('__')]
        except Exception as e:
            warnings.warn(f"Error scanning directory {directory}: {e}")
            return []
    
    def discover_models(self, model_dir: str = "models") -> Dict[str, List[str]]:
        """
        Discover available models in the framework.
        
        Args:
            model_dir: Directory containing models (default: "models")
            
        Returns:
            Dictionary mapping model categories to list of model names
        """
        cache_key = f"models_{model_dir}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        models = {}
        model_path = Path(self.framework_path) / model_dir
        
        if not model_path.exists():
            warnings.warn(f"Model directory not found: {model_path}")
            return models
        
        # Scan subdirectories (categories)
        for category_dir in model_path.iterdir():
            if category_dir.is_dir() and not category_dir.name.startswith('__'):
                category_name = category_dir.name
                model_files = []
                
                # Scan for Python files in the category
                for py_file in category_dir.glob("*.py"):
                    if py_file.is_file() and not py_file.name.startswith('__'):
                        model_name = py_file.stem
                        model_files.append(model_name)
                
                if model_files:
                    models[category_name] = sorted(model_files)
        
        self._cache[cache_key] = models
        return models
    
    def load_model_class(self, model_path: str, class_name: Optional[str] = None) -> Optional[Type]:
        """
        Dynamically load a model class from the framework.
        
        Args:
            model_path: Path to the model file (e.g., "models/ClassicNetwork/ResNet.py")
            class_name: Name of the class to load (if None, auto-detect)
            
        Returns:
            The loaded class or None if loading fails
        """
        try:
            # Convert file path to module path
            module_path = model_path.replace('.py', '').replace('/', '.').replace('\\', '.')
            
            # Remove framework path prefix if present
            if module_path.startswith(self.framework_path):
                module_path = module_path[len(self.framework_path):].lstrip('/\\')
            
            # Import the module
            module = self._import_module(module_path)
            
            if module is None:
                return None
            
            # Find the class
            if class_name:
                cls = getattr(module, class_name, None)
                if cls is None:
                    warnings.warn(f"Class {class_name} not found in module {module_path}")
                return cls
            else:
                # Auto-detect the first nn.Module class
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and hasattr(obj, '__module__'):
                        # Check if it's a PyTorch module
                        if obj.__module__ == module.__name__:
                            return obj
                
                warnings.warn(f"No suitable class found in module {module_path}")
                return None
                
        except Exception as e:
            warnings.warn(f"Error loading model class from {model_path}: {e}")
            return None
    
    def load_dataset_class(self, dataset_path: str, class_name: Optional[str] = None) -> Optional[Type]:
        """
        Dynamically load a dataset class from the framework.
        
        Args:
            dataset_path: Path to the dataset file
            class_name: Name of the class to load (if None, auto-detect)
            
        Returns:
            The loaded class or None if loading fails
        """
        return self.load_model_class(dataset_path, class_name)
    
    def load_function(self, module_path: str, function_name: str) -> Optional[Callable]:
        """
        Dynamically load a function from a module.
        
        Args:
            module_path: Path to the module
            function_name: Name of the function to load
            
        Returns:
            The loaded function or None if loading fails
        """
        try:
            module = self._import_module(module_path)
            
            if module is None:
                return None
            
            func = getattr(module, function_name, None)
            if func is None:
                warnings.warn(f"Function {function_name} not found in module {module_path}")
            elif not inspect.isfunction(func):
                warnings.warn(f"{function_name} is not a function in module {module_path}")
                return None
            
            return func
            
        except Exception as e:
            warnings.warn(f"Error loading function {function_name} from {module_path}: {e}")
            return None
    
    def discover_functions(self, module_path: str) -> List[str]:
        """
        Discover available functions in a module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            List of function names
        """
        try:
            module = self._import_module(module_path)
            
            if module is None:
                return []
            
            functions = []
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    functions.append(name)
            
            return functions
            
        except Exception as e:
            warnings.warn(f"Error discovering functions in {module_path}: {e}")
            return []
    
    def instantiate_class(self, cls: Type, **kwargs) -> Optional[Any]:
        """
        Instantiate a class with given parameters.
        
        Args:
            cls: The class to instantiate
            **kwargs: Parameters to pass to the class constructor
            
        Returns:
            Instance of the class or None if instantiation fails
        """
        try:
            # Get class signature
            sig = inspect.signature(cls.__init__)
            params = sig.parameters
            
            # Filter kwargs to match class parameters
            filtered_kwargs = {}
            for key, value in kwargs.items():
                if key in params:
                    filtered_kwargs[key] = value
                else:
                    warnings.warn(f"Parameter {key} not found in class {cls.__name__}")
            
            # Create instance
            return cls(**filtered_kwargs)
            
        except Exception as e:
            warnings.warn(f"Error instantiating class {cls.__name__}: {e}")
            return None
    
    def _import_module(self, module_path: str) -> Optional[Any]:
        """
        Import a module by path.
        
        Args:
            module_path: Path to the module (e.g., "models.ClassicNetwork.ResNet")
            
        Returns:
            The imported module or None if import fails
        """
        if module_path in self._module_cache:
            return self._module_cache[module_path]
        
        try:
            module = importlib.import_module(module_path)
            self._module_cache[module_path] = module
            return module
        except ImportError as e:
            warnings.warn(f"Cannot import module {module_path}: {e}")
            return None
        except Exception as e:
            warnings.warn(f"Error importing module {module_path}: {e}")
            return None
    
    def get_class_signature(self, cls: Type) -> Dict[str, Any]:
        """
        Get the signature of a class constructor.
        
        Args:
            cls: The class to inspect
            
        Returns:
            Dictionary mapping parameter names to their default values
        """
        try:
            sig = inspect.signature(cls.__init__)
            params = {}
            
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                
                if param.default != inspect.Parameter.empty:
                    params[name] = param.default
                else:
                    params[name] = None
            
            return params
            
        except Exception as e:
            warnings.warn(f"Error getting signature for class {cls.__name__}: {e}")
            return {}
    
    def get_function_signature(self, func: Callable) -> Dict[str, Any]:
        """
        Get the signature of a function.
        
        Args:
            func: The function to inspect
            
        Returns:
            Dictionary mapping parameter names to their default values
        """
        try:
            sig = inspect.signature(func)
            params = {}
            
            for name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    params[name] = param.default
                else:
                    params[name] = None
            
            return params
            
        except Exception as e:
            warnings.warn(f"Error getting signature for function {func.__name__}: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self._module_cache.clear()


class ModelLoader(DynamicLoader):
    """
    Specialized loader for model classes.
    """
    
    def __init__(self, framework_path: Optional[str] = None):
        super().__init__(framework_path)
        self._model_cache = {}
    
    def load_model(self, model_name: str, category: str, **kwargs) -> Optional[Any]:
        """
        Load and instantiate a model.
        
        Args:
            model_name: Name of the model file (e.g., "ResNet")
            category: Model category (e.g., "ClassicNetwork")
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            Model instance or None if loading fails
        """
        cache_key = f"{category}_{model_name}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        model_path = f"models.{category}.{model_name}"
        model_class = self.load_model_class(model_path)
        
        if model_class is None:
            return None
        
        model_instance = self.instantiate_class(model_class, **kwargs)
        
        if model_instance is not None:
            self._model_cache[cache_key] = model_instance
        
        return model_instance


class DatasetLoader(DynamicLoader):
    """
    Specialized loader for dataset classes.
    """
    
    def __init__(self, framework_path: Optional[str] = None):
        super().__init__(framework_path)
    
    def load_dataset(self, dataset_name: str, category: str, **kwargs) -> Optional[Any]:
        """
        Load and instantiate a dataset.
        
        Args:
            dataset_name: Name of the dataset file
            category: Dataset category
            **kwargs: Parameters to pass to the dataset constructor
            
        Returns:
            Dataset instance or None if loading fails
        """
        dataset_path = f"dataset.{category}.{dataset_name}"
        dataset_class = self.load_dataset_class(dataset_path)
        
        if dataset_class is None:
            return None
        
        return self.instantiate_class(dataset_class, **kwargs)


class FunctionLoader(DynamicLoader):
    """
    Specialized loader for functions (losses, metrics, etc.).
    """
    
    def __init__(self, framework_path: Optional[str] = None):
        super().__init__(framework_path)
    
    def load_loss_function(self, loss_name: str, category: str) -> Optional[Callable]:
        """
        Load a loss function.
        
        Args:
            loss_name: Name of the loss function
            category: Loss category
            
        Returns:
            Loss function or None if loading fails
        """
        loss_path = f"losses.{category}.{loss_name}"
        return self.load_function(loss_path, loss_name)
    
    def load_metric_function(self, metric_name: str, category: str) -> Optional[Callable]:
        """
        Load a metric function.
        
        Args:
            metric_name: Name of the metric function
            category: Metric category
            
        Returns:
            Metric function or None if loading fails
        """
        metric_path = f"metrics.{category}.{metric_name}"
        return self.load_function(metric_path, metric_name)