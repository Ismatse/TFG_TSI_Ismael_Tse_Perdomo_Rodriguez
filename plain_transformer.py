import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Configuration
class Config:
    # Data parameters
    train_file = "datasets/train.csv"
    test_file = "datasets/test.csv"
    batch_size = 64
    seq_length = 50  # Number of time steps to consider as a sequence
    stride = 10  # Stride for sliding window

    # Model parameters
    embedding_dim = 128
    num_heads = 8
    num_encoder_layers = 6
    num_classes = 4  # 'Benign_traffic', 'Benign_HH', 'Malign_HH', 'Unknown'
    dropout = 0.1

    # Training parameters
    lr = 0.0001
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"

    # Features
    numerical_features = [
        "udps.src2dst_last",
        "udps.dst2src_last",
        "udps.src2dst_pkts_data",
        "udps.dst2src_pkts_data",
        "src2dst_bytes",
        "dst2src_bytes",
        "udps.diff_dst_src_first",
    ]
    categorical_features = ["udps.protocol"]

    # Preprocessing
    time_feature_engineering = True  # Extract time features from timestamp
    categorical_encoding = "one-hot"  # 'one-hot' or 'embedding'

    def __init__(self):
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Dataset class for network traffic data
class NetworkTrafficDataset(Dataset):
    def __init__(self, config, file_path, scaler=None, is_train=True):
        self.config = config
        self.is_train = is_train

        # Read data
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path, sep="*")

        # Preprocess data
        self._preprocess_data()

        # Scale numerical features
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[config.numerical_features] = self.scaler.fit_transform(
                self.df[config.numerical_features]
            )
        else:
            self.scaler = scaler
            self.df[config.numerical_features] = self.scaler.transform(
                self.df[config.numerical_features]
            )

        # Encode categorical features
        self._encode_categorical_features()

        # Engineer time features
        if config.time_feature_engineering:
            self._engineer_time_features()

        # Encode target variable
        self._encode_target()

        # Create sequences
        self._create_sequences()

    def _preprocess_data(self):
        # Convert timestamp to datetime
        self.df["timestamp"] = pd.to_datetime(self.df["udps.timestamp"])

        # Sort by timestamp
        self.df = self.df.sort_values("timestamp")

        # Handle missing values
        self.df = self.df.fillna(0)

    def _encode_categorical_features(self):
        # Encode protocol
        if self.config.categorical_encoding == "one-hot":
            # One-hot encode protocol
            protocol_dummies = pd.get_dummies(
                self.df["udps.protocol"], prefix="protocol"
            )
            self.df = pd.concat([self.df, protocol_dummies], axis=1)

            # Update categorical features list
            self.categorical_columns = list(protocol_dummies.columns)
        else:
            # We'll handle embedding in the model
            self.df["protocol_encoded"] = (
                self.df["udps.protocol"].astype("category").cat.codes
            )
            self.categorical_columns = ["protocol_encoded"]

    def _engineer_time_features(self):
        # Extract time features
        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["minute"] = self.df["timestamp"].dt.minute
        self.df["second"] = self.df["timestamp"].dt.second
        self.df["day_of_week"] = self.df["timestamp"].dt.dayofweek

        # Normalize time features
        time_features = ["hour", "minute", "second", "day_of_week"]
        self.df["hour"] = self.df["hour"] / 23.0
        self.df["minute"] = self.df["minute"] / 59.0
        self.df["second"] = self.df["second"] / 59.0
        self.df["day_of_week"] = self.df["day_of_week"] / 6.0

        # Add to numerical features
        self.config.numerical_features.extend(time_features)

    def _encode_target(self):
        # Map categories to numerical values
        category_mapping = {
            "Benign_traffic": 0,
            "Benign_HH": 1,
            "Malign_HH": 2,
            "Unknown": 3,
        }

        self.df["target"] = self.df["category"].map(category_mapping)

    def _create_sequences(self):
        # Prepare feature columns
        feature_cols = self.config.numerical_features + self.categorical_columns

        # Create sequences
        self.sequences = []
        self.targets = []

        # Use sliding window to create sequences
        for i in tqdm(
            range(0, len(self.df) - self.config.seq_length, self.config.stride)
        ):
            seq = self.df.iloc[i : i + self.config.seq_length][feature_cols].values
            target = self.df.iloc[i + self.config.seq_length - 1]["target"]

            self.sequences.append(seq)
            self.targets.append(target)

        # Convert to numpy arrays
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

        print(
            f"Created {len(self.sequences)} sequences of length {self.config.seq_length}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(
            self.targets[idx], dtype=torch.long
        )


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[: x.size(0), :]
        return x


# TimeSeriesTransformer model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, config, input_dim):
        super(TimeSeriesTransformer, self).__init__()

        self.config = config

        # Feature embedding
        self.embedding = nn.Linear(input_dim, config.embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.embedding_dim)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            batch_first=False,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, num_layers=config.num_encoder_layers
        )

        # Output layer
        self.output_layer = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]

        # Transpose for transformer input: [seq_len, batch_size, input_dim]
        src = src.permute(1, 0, 2)

        # Embedding
        src = self.embedding(src)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Transformer encoder
        output = self.transformer_encoder(src)

        # Take the output of the last token for classification
        output = output[-1]

        # Output layer
        output = self.output_layer(output)

        return output


# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return total_loss / len(train_loader), correct / total


# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

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
        target_names=["Benign_traffic", "Benign_HH", "Malign_HH", "Unknown"],
    )
    conf_matrix = confusion_matrix(all_targets, all_predictions)

    return total_loss / len(test_loader), accuracy, report, conf_matrix


# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, accuracy, config):
    checkpoint_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


# Plot training history
def plot_history(train_losses, train_accuracies, val_losses, val_accuracies, config):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, "training_history.png"))
    plt.show()


# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, config):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign_traffic", "Benign_HH", "Malign_HH", "Unknown"],
        yticklabels=["Benign_traffic", "Benign_HH", "Malign_HH", "Unknown"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.log_dir, "confusion_matrix.png"))
    plt.show()


# Main function
def main():
    # Initialize config
    config = Config()
    print(f"Using device: {config.device}")

    # Create datasets
    train_dataset = NetworkTrafficDataset(config, config.train_file, is_train=True)
    test_dataset = NetworkTrafficDataset(
        config, config.test_file, scaler=train_dataset.scaler, is_train=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate input dimension based on features
    input_dim = train_dataset.sequences.shape[2]

    # Initialize model
    model = TimeSeriesTransformer(config, input_dim).to(config.device)

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # Training history
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, config.device
        )

        # Evaluate
        val_loss, val_accuracy, report, conf_matrix = evaluate(
            model, test_loader, criterion, config.device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Record history
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, config)

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{config.epochs} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )
        print(report)

    # Plot training history
    plot_history(train_losses, train_accuracies, val_losses, val_accuracies, config)

    # Plot confusion matrix for the final model
    plot_confusion_matrix(conf_matrix, config)


if __name__ == "__main__":
    main()
