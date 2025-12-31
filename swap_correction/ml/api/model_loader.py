"""
Model loader for swap detection models.

Provides unified interface to load models, preprocessors, and metadata.
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# Model directory mappings
MODEL_DIRS = {
    'level1': 'ml_models',
    'raw': 'ml_models_raw',
    'raw_data': 'ml_models_raw',  # Alias
}

# Model file names
MODEL_FILES = {
    'model': 'swap_detector_xgb.pkl',
    'scaler': 'feature_scaler.pkl',
    'imputer': 'feature_imputer.pkl',
    'feature_names': 'feature_names.pkl',
    'results': 'training_results.json',
    'importance': 'feature_importance.csv',
}


def get_model_dir(model_type: str) -> Path:
    """
    Get the directory path for a model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'level1' or 'raw'/'raw_data'
        
    Returns:
    --------
    Path
        Path to model directory
        
    Raises:
    -------
    ValueError
        If model_type is not recognized
    """
    if model_type not in MODEL_DIRS:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(MODEL_DIRS.keys())}")
    
    model_dir = Path(MODEL_DIRS[model_type])
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Make sure the model has been trained. See docs/ML_TRAINING_GUIDE.md"
        )
    
    return model_dir


def load_model(model_type: Literal['level1', 'raw', 'raw_data'] = 'level1',
               base_dir: Optional[str] = None) -> Tuple[xgb.XGBClassifier, StandardScaler, SimpleImputer, list]:
    """
    Load a trained swap detection model and its preprocessors.
    
    Parameters:
    -----------
    model_type : str
        Type of model to load: 'level1' or 'raw'/'raw_data'
        - 'level1': Model trained on level1.csv vs level2.csv
        - 'raw'/'raw_data': Model trained on raw _data.csv vs level2.csv
    base_dir : str, optional
        Base directory for model files (default: current working directory)
        
    Returns:
    --------
    tuple
        (model, scaler, imputer, feature_names)
        - model: Trained XGBoost classifier
        - scaler: StandardScaler for feature normalization
        - imputer: SimpleImputer for handling missing values
        - feature_names: List of feature names in order
        
    Raises:
    -------
    FileNotFoundError
        If model files are not found
    ValueError
        If model_type is invalid
        
    Example:
    --------
    >>> model, scaler, imputer, feature_names = load_model('level1')
    >>> print(f"Loaded model with {len(feature_names)} features")
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    model_dir = Path(base_dir) / get_model_dir(model_type)
    
    # Load model
    model_path = model_dir / MODEL_FILES['model']
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    scaler_path = model_dir / MODEL_FILES['scaler']
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load imputer
    imputer_path = model_dir / MODEL_FILES['imputer']
    if not imputer_path.exists():
        raise FileNotFoundError(f"Imputer file not found: {imputer_path}")
    
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    
    # Load feature names
    feature_names_path = model_dir / MODEL_FILES['feature_names']
    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return model, scaler, imputer, feature_names


def get_model_info(model_type: Literal['level1', 'raw', 'raw_data'] = 'level1',
                  base_dir: Optional[str] = None) -> Dict:
    """
    Get metadata and performance information for a model.
    
    Parameters:
    -----------
    model_type : str
        Type of model: 'level1' or 'raw'/'raw_data'
    base_dir : str, optional
        Base directory for model files (default: current working directory)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'model_type': Model type
        - 'model_dir': Path to model directory
        - 'performance': Test set performance metrics
        - 'training_data': Training data characteristics
        - 'configuration': Model configuration
        - 'feature_count': Number of features
        
    Example:
    --------
    >>> info = get_model_info('level1')
    >>> print(f"Test F1-Score: {info['performance']['test']['f1']:.4f}")
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    model_dir = Path(base_dir) / get_model_dir(model_type)
    
    # Load training results
    results_path = model_dir / MODEL_FILES['results']
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load feature names to get count
    feature_names_path = model_dir / MODEL_FILES['feature_names']
    if feature_names_path.exists():
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        feature_count = len(feature_names)
    else:
        feature_count = None
    
    info = {
        'model_type': model_type,
        'model_dir': str(model_dir),
        'performance': {
            'train': results.get('train', {}),
            'val': results.get('val', {}),
            'test': results.get('test', {}),
        },
        'training_data': {
            'train_samples': results.get('train', {}).get('n_samples'),
            'train_positive': results.get('train', {}).get('n_positive'),
            'test_samples': results.get('test', {}).get('n_samples'),
            'test_positive': results.get('test', {}).get('n_positive'),
        },
        'feature_count': feature_count,
        'configuration': {
            'algorithm': 'XGBoost',
            'gaussian_filter_sigma': 4.6,  # From training
        }
    }
    
    return info


def list_available_models(base_dir: Optional[str] = None) -> Dict[str, bool]:
    """
    List all available models and their availability status.
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory for model files (default: current working directory)
        
    Returns:
    --------
    dict
        Dictionary mapping model types to availability (True/False)
        
    Example:
    --------
    >>> available = list_available_models()
    >>> for model_type, is_available in available.items():
    ...     print(f"{model_type}: {'Available' if is_available else 'Not found'}")
    """
    if base_dir is None:
        base_dir = os.getcwd()
    
    available = {}
    
    for model_type, model_dir_name in MODEL_DIRS.items():
        if model_type == 'raw_data':  # Skip alias
            continue
        
        model_dir = Path(base_dir) / model_dir_name
        model_path = model_dir / MODEL_FILES['model']
        available[model_type] = model_path.exists()
    
    return available

