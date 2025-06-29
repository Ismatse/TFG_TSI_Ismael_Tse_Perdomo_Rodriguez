import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NetworkTrafficDataset(Dataset):
    """
    Dataset class for network traffic data.
    Handles preprocessing, feature engineering, and sequence creation.
    """

    def __init__(
        self,
        opt,
        file_path,
        scaler=None,
        is_train=True,
        split_type=None,
        train_data=None,
    ):
        """
        Initialize the dataset.

        Args:
            opt: Options object with configuration parameters
            file_path: Path to the CSV data file
            scaler: Optional pre-fitted scaler for numerical features
            is_train: Whether this is training data
            split_type: 'train', 'val', or None
            train_data: Preprocessed training dataframe for validation split
        """
        self.opt = opt
        self.is_train = is_train

        # Read data
        if train_data is not None:
            # Use provided training data for validation split
            self.df = train_data.copy()
        else:
            print(f"Loading data from {file_path}...")
            self.df = pd.read_csv(file_path, sep="*")
            # Preprocess data
            self._preprocess_data()

        # Handle train/validation split if needed
        if split_type is not None and train_data is None:
            # Split the data
            split_idx = int(len(self.df) * (1 - opt.val_split))
            if split_type == "train":
                self.df = self.df.iloc[:split_idx].reset_index(drop=True)
            elif split_type == "val":
                self.df = self.df.iloc[split_idx:].reset_index(drop=True)

        # Define feature columns
        self.numerical_features = [
            "udps.src2dst_last",
            "udps.dst2src_last",
            "udps.src2dst_pkts_data",
            "udps.dst2src_pkts_data",
            "src2dst_bytes",
            "dst2src_bytes",
            "udps.diff_dst_src_first",
        ]

        self.categorical_features = ["udps.protocol"]

        # Scale numerical features
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[self.numerical_features] = self.scaler.fit_transform(
                self.df[self.numerical_features]
            )
        else:
            self.scaler = scaler
            self.df[self.numerical_features] = self.scaler.transform(
                self.df[self.numerical_features]
            )

        # Encode categorical features
        self._encode_categorical_features()

        # Engineer time features if requested
        if opt.time_feature_engineering:
            self._engineer_time_features()

        # Encode target variable
        self._encode_target()

        # Create sequences
        self._create_sequences()

    def _preprocess_data(self):
        """Preprocess the raw data."""
        # Extract timestamp from the first column (udps.timestamp*udps.protocol)
        self.df["timestamp"] = pd.to_datetime(self.df["udps.timestamp"], unit="ms")

        # Sort by timestamp
        self.df = self.df.sort_values("timestamp")

        # Handle missing values
        self.df = self.df.fillna(0)

    def _encode_categorical_features(self):
        """Encode categorical features."""
        # Encode protocol
        if self.opt.categorical_encoding == "one-hot":
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
        """Extract features from timestamp."""
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
        self.numerical_features.extend(time_features)

    def _encode_target(self):
        """Encode target variable."""
        # Map categories to numerical values
        category_mapping = {"Benign_traffic": 0, "Benign_HH": 1, "Malign_HH": 2}

        self.df["target"] = self.df["category"].map(category_mapping)

    def _create_sequences(self):
        """Create sequences for training."""
        # Prepare feature columns
        feature_cols = self.numerical_features + self.categorical_columns

        # Create sequences
        self.sequences = []
        self.targets = []

        # Use sliding window to create sequences
        for i in tqdm(range(0, len(self.df) - self.opt.seq_length, self.opt.stride)):
            seq = self.df.iloc[i : i + self.opt.seq_length][feature_cols].values
            target = self.df.iloc[i + self.opt.seq_length - 1]["target"]

            self.sequences.append(seq)
            self.targets.append(target)

        # Convert to numpy arrays with explicit dtype
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int64)

        print(
            f"Created {len(self.sequences)} sequences of length {self.opt.seq_length}"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(
            self.targets[idx], dtype=torch.long
        )


def create_data_loaders(opt):
    """
    Create train, validation, and test data loaders.

    Args:
        opt: Options object with configuration parameters

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        train_dataset: Training dataset (for accessing the scaler)
    """
    # Create full training dataset first
    full_train_dataset = NetworkTrafficDataset(
        opt, opt.train_file, is_train=True, split_type="train"
    )

    # Create validation dataset using the same preprocessed data
    val_dataset = NetworkTrafficDataset(
        opt,
        opt.train_file,
        scaler=full_train_dataset.scaler,
        is_train=False,
        split_type="val",
    )

    # Create test dataset
    test_dataset = NetworkTrafficDataset(
        opt, opt.test_file, scaler=full_train_dataset.scaler, is_train=False
    )

    # Create data loaders
    train_loader = DataLoader(
        full_train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train_dataset
