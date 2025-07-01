import os
import time

import numpy as np
import torch
import numpy as np
import os
import time
from datetime import datetime
from options import Options
from plain_transformer import (
    PlainTransformer, create_data_loaders as plain_create_data_loaders, 
    Trainer as PlainTrainer,
    create_dirs, set_seeds, plot_history, plot_confusion_matrix,
    save_metrics, print_model_summary, get_available_memory, save_run_info
)
from informer import (
    Informer, create_data_loaders as informer_create_data_loaders,
    Trainer as InformerTrainer
)


def main():
    # Parse arguments
    opt_parser = Options()
    opt = opt_parser.parse()

    # Set random seeds for reproducibility
    set_seeds(42)

    # Create necessary directories
    create_dirs([opt.checkpoint_dir, opt.log_dir])

    # Check available GPU memory
    memory_info = get_available_memory()
    print(f"\nDevice: {memory_info['device']}")
    if "cuda" in opt.device:
        print(f"Available GPU memory: {memory_info['available_memory_mb']:.2f} MB")

    # Create data loaders based on model type
    print("\nCreating data loaders...")
    if opt.model == "plain_transformer":
        train_loader, val_loader, test_loader, train_dataset = plain_create_data_loaders(opt)
    elif opt.model == "informer":
        train_loader, val_loader, test_loader, train_dataset = informer_create_data_loaders(opt)
    else:
        raise ValueError(f"Unknown model type: {opt.model}")

    # Calculate input dimension based on dataset
    input_dim = train_dataset.sequences.shape[2]
    print(f"Input dimension: {input_dim}")

    # Create model
    print(f"\nCreating {opt.model} model...")
    if opt.model == "plain_transformer":
        model = PlainTransformer(opt, input_dim)
    elif opt.model == "informer":
        model = Informer(opt, input_dim)
    else:
        raise ValueError(f"Unknown model type: {opt.model}")

    # Print model summary
    print_model_summary(model, input_size=(opt.batch_size, opt.seq_length, input_dim))

    # Create trainer based on model type
    if opt.model == "plain_transformer":
        trainer = PlainTrainer(model, opt, train_loader, val_loader, test_loader)
    elif opt.model == "informer":
        trainer = InformerTrainer(model, opt, train_loader, val_loader, test_loader)
    else:
        raise ValueError(f"Unknown model type: {opt.model}")
    
    # Resume from checkpoint if specified
    if opt.resume_from:
        if os.path.exists(opt.resume_from):
            print(f"Resuming training from checkpoint: {opt.resume_from}")
            trainer.load_checkpoint(opt.resume_from, resume_training=True)
        else:
            print(f"Warning: Checkpoint file {opt.resume_from} not found. Starting from scratch.")

    # Save run information
    run_start_time = datetime.now()
    save_run_info(opt.log_dir, opt, start_time=run_start_time)

    # Start training
    start_time = time.time()
    training_history = trainer.train()
    training_time = time.time() - start_time

    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")

    # Plot training history
    plot_history(
        training_history["train_losses"],
        training_history["train_accuracies"],
        training_history["val_losses"],
        training_history["val_accuracies"],
        opt.log_dir,
        opt.model,
    )

    # Load best model and evaluate on test set
    best_model_path = os.path.join(opt.checkpoint_dir, "model_best.pt")
    test_loss, test_accuracy, test_report, test_conf_matrix = trainer.test(
        checkpoint_path=best_model_path
    )

    # Save test metrics
    test_metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_report": test_report,
        "test_confusion_matrix": test_conf_matrix.tolist()
        if test_conf_matrix is not None
        else None,
        "training_time": training_time,
        "best_epoch": training_history["best_epoch"],
        "best_val_loss": training_history["best_val_loss"],
    }
    save_metrics(test_metrics, opt.log_dir, opt.model)

    # Compute and save all true and predicted labels
    if test_loader:
        print("\nGathering predictions on test set...")
        trainer.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(opt.device)
                output = trainer.model(data)
                _, predicted = torch.max(output.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.numpy())

        # Plot confusion matrix
        class_names = ["Benign_traffic", "Benign_HH", "Malign_HH"]
        plot_confusion_matrix(
            np.array(all_targets),
            np.array(all_preds),
            class_names,
            opt.log_dir,
            opt.model,
            phase="test",
        )

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    main()
