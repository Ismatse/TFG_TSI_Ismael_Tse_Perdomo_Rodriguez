import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

def create_dirs(directories):
    """
    Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def plot_history(train_losses, train_accuracies, val_losses, val_accuracies, log_dir, model_name):
    """
    Plot training and validation history.
    
    Args:
        train_losses: List of training losses
        train_accuracies: List of training accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        log_dir: Directory to save the plots
        model_name: Name of the model for plot titles and filenames
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{model_name}_training_history.png'))
    print(f"Training history plot saved to {os.path.join(log_dir, f'{model_name}_training_history.png')}")
    

def plot_confusion_matrix(y_true, y_pred, class_names, log_dir, model_name, phase='test'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        log_dir: Directory to save the plot
        model_name: Name of the model for plot title and filename
        phase: 'train', 'val', or 'test'
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix ({phase.capitalize()})')
    plt.tight_layout()
    
    plt.savefig(os.path.join(log_dir, f'{model_name}_{phase}_confusion_matrix.png'))
    print(f"Confusion matrix plot saved to {os.path.join(log_dir, f'{model_name}_{phase}_confusion_matrix.png')}")


def save_metrics(metrics, log_dir, model_name):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        log_dir: Directory to save the metrics
        model_name: Name of the model for filename
    """
    # Convert numpy arrays to lists for JSON serialization
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    
    with open(os.path.join(log_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(log_dir, f'{model_name}_metrics.json')}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def print_model_summary(model, input_size=None):
    """
    Print a summary of the model similar to Keras model.summary().
    
    Args:
        model: PyTorch model
        input_size: Input size as tuple (batch_size, seq_length, input_dim)
    """
    print(f"Model Summary:")
    print("=" * 80)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("-" * 80)
    print(model)
    print("=" * 80)
    
    if input_size is not None:
        # Calculate model's memory usage
        batch_size, seq_length, input_dim = input_size
        # Estimate memory usage (very rough approximation)
        param_size = sum([p.nelement() * p.element_size() for p in model.parameters()])
        buffer_size = sum([b.nelement() * b.element_size() for b in model.buffers()])
        
        # Assuming 4 bytes per parameter for gradients and optimizer state
        estimated_total = param_size + buffer_size + (param_size * 2)  # parameters + buffers + gradients + optimizer
        
        print(f"Estimated model size: {param_size / 1024 / 1024:.2f} MB")
        print(f"Estimated total memory usage: {estimated_total / 1024 / 1024:.2f} MB")
        print("=" * 80)


def get_available_memory():
    """
    Get available GPU memory if CUDA is available.
    
    Returns:
        available_memory: Available memory in MB
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = (total_memory - reserved_memory - allocated_memory) / 1024 / 1024
        
        return {
            'device': torch.cuda.get_device_name(device),
            'total_memory_mb': total_memory / 1024 / 1024,
            'reserved_memory_mb': reserved_memory / 1024 / 1024,
            'allocated_memory_mb': allocated_memory / 1024 / 1024,
            'available_memory_mb': available_memory
        }
    else:
        return {'device': 'CPU', 'available_memory_mb': 0}