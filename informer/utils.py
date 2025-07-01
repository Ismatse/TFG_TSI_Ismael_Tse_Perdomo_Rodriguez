import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime

def create_run_directory(base_checkpoint_dir, base_log_dir, run_name=None):
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    log_dir = os.path.join(base_log_dir, run_name)
    
    return checkpoint_dir, log_dir, run_name


def list_available_runs(base_checkpoint_dir):
    if not os.path.exists(base_checkpoint_dir):
        return []
    
    runs = []
    for item in os.listdir(base_checkpoint_dir):
        item_path = os.path.join(base_checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("run_"):
            runs.append(item)
    
    return sorted(runs)


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') and file.startswith('model_epoch_'):
            epoch_num = int(file.split('_')[-1].split('.')[0])
            checkpoints.append((epoch_num, os.path.join(checkpoint_dir, file)))
    
    if checkpoints:
        return sorted(checkpoints, key=lambda x: x[0])[-1][1]
    
    return None


def save_run_info(log_dir, opt, start_time=None):
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'start_time': start_time.isoformat() if start_time else None,
        'model': opt.model,
        'epochs': opt.epochs,
        'batch_size': opt.batch_size,
        'learning_rate': opt.lr,
        'weight_decay': opt.weight_decay,
        'early_stopping_patience': opt.early_stopping_patience,
        'device': opt.device,
        'resume_from': opt.resume_from if hasattr(opt, 'resume_from') else None
    }
    
    info_path = os.path.join(log_dir, 'run_info.json')
    with open(info_path, 'w') as f:
        json.dump(run_info, f, indent=4)
    
    print(f"Run info saved to {info_path}")


def create_dirs(directories):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created/verified: {directory}")


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def plot_history(train_losses, train_accuracies, val_losses, val_accuracies, log_dir, model_name):
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
    plot_path = os.path.join(log_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")
    plt.close()
    

def plot_confusion_matrix(y_true, y_pred, class_names, log_dir, model_name, phase='test'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix ({phase.capitalize()})')
    plt.tight_layout()
    
    plot_path = os.path.join(log_dir, f'{model_name}_{phase}_confusion_matrix.png')
    plt.savefig(plot_path)
    print(f"Confusion matrix plot saved to {plot_path}")
    plt.close()


def save_metrics(metrics, log_dir, model_name):
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics[key] = value.tolist()
    
    metrics_path = os.path.join(log_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def print_model_summary(model, input_size=None):
    print(f"Model Summary:")
    print("=" * 80)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("-" * 80)
    print(model)
    print("=" * 80)
    
    if input_size is not None:
        batch_size, seq_length, input_dim = input_size
        param_size = sum([p.nelement() * p.element_size() for p in model.parameters()])
        buffer_size = sum([b.nelement() * b.element_size() for b in model.buffers()])
        
        estimated_total = param_size + buffer_size + (param_size * 2)
        
        print(f"Estimated model size: {param_size / 1024 / 1024:.2f} MB")
        print(f"Estimated total memory usage: {estimated_total / 1024 / 1024:.2f} MB")
        print("=" * 80)


def get_available_memory():
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