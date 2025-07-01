import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .poly_lr import PolyLRScheduler


class InformerTrainer:
    def __init__(self, model, opt, train_loader, val_loader, test_loader=None):
        self.model = model
        self.opt = opt
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.opt.lr, weight_decay=self.opt.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = PolyLRScheduler(self.optimizer, self.opt.lr, self.opt.epochs)

        # Move model to device
        self.model = self.model.to(self.opt.device)

        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_epoch = 0

        # Early stopping
        self.early_stopping_patience = opt.early_stopping_patience
        self.early_stopping_counter = 0

        # Current epoch (for resuming)
        self.current_epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(
            tqdm(self.train_loader, desc="Training")
        ):
            data, target = data.to(self.opt.device), target.to(self.opt.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self, data_loader, phase="val"):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(data_loader, desc=f"Evaluating ({phase})"):
                data, target = data.to(self.opt.device), target.to(self.opt.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Calculate metrics
        accuracy = correct / total
        report = classification_report(
            all_targets,
            all_predictions,
            target_names=["Benign_traffic", "Benign_HH", "Malign_HH"],
            output_dict=True,
            zero_division=0
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)

        return total_loss / len(data_loader), accuracy, report, conf_matrix

    def train(self):
        start_epoch = self.current_epoch
        print(
            f"\nStarting training from epoch {start_epoch + 1} to {self.opt.epochs}..."
        )

        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}")

        for epoch in range(start_epoch, self.opt.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss, train_accuracy = self.train_epoch()

            # Evaluate on validation set
            val_loss, val_accuracy, val_report, val_conf_matrix = self.evaluate(
                self.val_loader, phase="val"
            )

            # Record history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                improved = True
                self.save_checkpoint("model_best.pt")
                print(f"New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                self.early_stopping_counter += 1

            # Save checkpoint at regular intervals
            if (
                epoch + 1
            ) % self.opt.checkpoint_interval == 0 or epoch == self.opt.epochs - 1:
                self.save_checkpoint(f"model_epoch_{epoch + 1}.pt")

            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(
                f"Epoch {epoch + 1}/{self.opt.epochs} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
            )

            # Print classification report
            print("\nClassification Report (Validation):")
            for cls in val_report:
                if cls not in ["accuracy", "macro avg", "weighted avg"]:
                    print(
                        f"{cls}: Precision = {val_report[cls]['precision']:.4f}, "
                        f"Recall = {val_report[cls]['recall']:.4f}, "
                        f"F1 = {val_report[cls]['f1-score']:.4f}"
                    )

            # Early stopping check
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(
                    f"\nEarly stopping triggered after {self.early_stopping_patience} epochs without improvement."
                )
                print(
                    f"Best model was at epoch {self.best_epoch + 1} with validation loss {self.best_val_loss:.4f}"
                )
                break

            if not improved:
                print(
                    f"No improvement for {self.early_stopping_counter}/{self.early_stopping_patience} epochs"
                )

        if self.current_epoch == self.opt.epochs - 1:
            print(
                f"\nTraining completed. Best model from epoch {self.best_epoch + 1} "
                f"with validation loss {self.best_val_loss:.4f}."
            )

        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "early_stopped": self.early_stopping_counter
            >= self.early_stopping_patience,
        }

    def test(self, checkpoint_path=None):
        if self.test_loader is None:
            print("No test data loader provided.")
            return None, None, None, None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, resume_training=False)
            print(f"Loaded model from {checkpoint_path}")

        print("\nEvaluating on test set...")
        test_loss, test_accuracy, test_report, test_conf_matrix = self.evaluate(
            self.test_loader, phase="test"
        )

        # Print test results
        print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report (Test):")
        for cls in test_report:
            if cls not in ["accuracy", "macro avg", "weighted avg"]:
                print(
                    f"{cls}: Precision = {test_report[cls]['precision']:.4f}, "
                    f"Recall = {test_report[cls]['recall']:.4f}, "
                    f"F1 = {test_report[cls]['f1-score']:.4f}"
                )

        return test_loss, test_accuracy, test_report, test_conf_matrix

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(self.opt.checkpoint_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_losses": self.train_losses,
                "train_accuracies": self.train_accuracies,
                "val_losses": self.val_losses,
                "val_accuracies": self.val_accuracies,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
                "current_epoch": self.current_epoch,
                "early_stopping_counter": self.early_stopping_counter,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path, resume_training=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.opt.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if resume_training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "train_losses" in checkpoint:
                self.train_losses = checkpoint["train_losses"]
                self.train_accuracies = checkpoint["train_accuracies"]
                self.val_losses = checkpoint["val_losses"]
                self.val_accuracies = checkpoint["val_accuracies"]
                self.best_val_loss = checkpoint["best_val_loss"]
                self.best_epoch = checkpoint["best_epoch"]

            if "current_epoch" in checkpoint:
                self.current_epoch = (
                    checkpoint["current_epoch"] + 1
                )

            if "early_stopping_counter" in checkpoint:
                self.early_stopping_counter = checkpoint["early_stopping_counter"]

            print(f"Resuming training from epoch {self.current_epoch + 1}")
            print(
                f"Best validation loss so far: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}"
            )
        else:
            print(f"Loaded model weights from {checkpoint_path}")