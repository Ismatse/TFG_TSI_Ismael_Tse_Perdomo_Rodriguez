#!/usr/bin/env python3
"""
Run management utility for the transformer time series project.
Helps users list, inspect, and resume from previous runs.
"""

import argparse
import os
import json
from datetime import datetime
from plain_transformer.utils import list_available_runs, find_latest_checkpoint

def list_runs(base_checkpoint_dir="./checkpoints"):
    """List all available runs."""
    runs = list_available_runs(base_checkpoint_dir)
    
    if not runs:
        print("No runs found.")
        return
    
    print("\nAvailable runs:")
    print("-" * 80)
    print(f"{'Run Name':<25} {'Date':<12} {'Time':<8} {'Latest Checkpoint':<20}")
    print("-" * 80)
    
    for run in runs:
        # Extract timestamp from run name
        if run.startswith("run_"):
            timestamp_str = run[4:]  # Remove "run_" prefix
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                date_str = timestamp.strftime("%Y-%m-%d")
                time_str = timestamp.strftime("%H:%M:%S")
            except ValueError:
                date_str = "Unknown"
                time_str = "Unknown"
        else:
            date_str = "Unknown"
            time_str = "Unknown"
        
        # Find latest checkpoint
        run_checkpoint_dir = os.path.join(base_checkpoint_dir, run)
        latest_checkpoint = find_latest_checkpoint(run_checkpoint_dir)
        
        if latest_checkpoint:
            checkpoint_name = os.path.basename(latest_checkpoint)
        else:
            checkpoint_name = "No checkpoints"
        
        print(f"{run:<25} {date_str:<12} {time_str:<8} {checkpoint_name:<20}")

def inspect_run(run_name, base_checkpoint_dir="./checkpoints", base_log_dir="./logs"):
    """Inspect details of a specific run."""
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    log_dir = os.path.join(base_log_dir, run_name)
    
    if not os.path.exists(checkpoint_dir):
        print(f"Run '{run_name}' not found.")
        return
    
    print(f"\nInspecting run: {run_name}")
    print("=" * 60)
    
    # Read run info if available
    run_info_path = os.path.join(log_dir, "run_info.json")
    if os.path.exists(run_info_path):
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)
        
        print("Run Configuration:")
        print("-" * 20)
        print(f"Model: {run_info.get('model', 'Unknown')}")
        print(f"Epochs: {run_info.get('epochs', 'Unknown')}")
        print(f"Batch Size: {run_info.get('batch_size', 'Unknown')}")
        print(f"Learning Rate: {run_info.get('learning_rate', 'Unknown')}")
        print(f"Weight Decay: {run_info.get('weight_decay', 'Unknown')}")
        print(f"Early Stopping Patience: {run_info.get('early_stopping_patience', 'Unknown')}")
        print(f"Device: {run_info.get('device', 'Unknown')}")
        if run_info.get('start_time'):
            start_time = datetime.fromisoformat(run_info['start_time'])
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List checkpoints
    print(f"\nCheckpoints in {checkpoint_dir}:")
    print("-" * 30)
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoints:
            for checkpoint in sorted(checkpoints):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
                print(f"  {checkpoint} ({size:.1f} MB)")
        else:
            print("  No checkpoints found")
    
    # Check for metrics
    metrics_path = os.path.join(log_dir, "plain_transformer_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print(f"\nFinal Results:")
        print("-" * 15)
        print(f"Test Accuracy: {metrics.get('test_accuracy', 'Unknown'):.4f}")
        print(f"Test Loss: {metrics.get('test_loss', 'Unknown'):.4f}")
        print(f"Best Epoch: {metrics.get('best_epoch', 'Unknown')}")
        print(f"Best Val Loss: {metrics.get('best_val_loss', 'Unknown'):.4f}")
        if metrics.get('training_time'):
            hours = int(metrics['training_time'] // 3600)
            minutes = int((metrics['training_time'] % 3600) // 60)
            print(f"Training Time: {hours}h {minutes}m")

def get_resume_command(run_name, base_checkpoint_dir="./checkpoints"):
    """Get the command to resume a specific run."""
    checkpoint_dir = os.path.join(base_checkpoint_dir, run_name)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint:
        print(f"\nTo resume run '{run_name}', use:")
        print(f"python main.py --resume_from {latest_checkpoint}")
    else:
        print(f"No checkpoints found for run '{run_name}'")

def main():
    parser = argparse.ArgumentParser(description="Manage transformer training runs")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available runs')
    list_parser.add_argument('--checkpoint_dir', default='./checkpoints', 
                           help='Base checkpoint directory')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a specific run')
    inspect_parser.add_argument('run_name', help='Name of the run to inspect')
    inspect_parser.add_argument('--checkpoint_dir', default='./checkpoints', 
                               help='Base checkpoint directory')
    inspect_parser.add_argument('--log_dir', default='./logs', 
                               help='Base log directory')
    
    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Get resume command for a run')
    resume_parser.add_argument('run_name', help='Name of the run to resume')
    resume_parser.add_argument('--checkpoint_dir', default='./checkpoints', 
                              help='Base checkpoint directory')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_runs(args.checkpoint_dir)
    elif args.command == 'inspect':
        inspect_run(args.run_name, args.checkpoint_dir, args.log_dir)
    elif args.command == 'resume':
        get_resume_command(args.run_name, args.checkpoint_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()