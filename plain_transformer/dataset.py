import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class NetworkTrafficDataset(Dataset):
    """
    Dataset class for network traffic data.
    Handles preprocessing, feature engineering, and sequence creation.
    """
    def __init__(self, opt, file_path, scaler=None, is_train=True, val_split=None):
        """
        Initialize the dataset.
        
        Args:
            opt: Options object with configuration parameters
            file_path: Path to the CSV data file
            scaler: Optional pre-fitted scaler for numerical features
            is_train: Whether this is training data
            val_split: Optional validation split ratio (0.0-1.0)
        """
        self.opt = opt
        self.is_train = is_train
        
        # Read data
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path, sep='*')
        
        # Preprocess data
        self._preprocess_data()
        
        # Create train/validation split if needed
        if is_train and val_split:
            train_idx, val_idx = train_test_split(
                np.arange(len(self.df)), 
                test_size=val_split,
                shuffle=False,  # No shuffle for time series
                random_state=42
            )
            if val_split == 'train':
                self.df = self.df.iloc[train_idx].reset_index(drop=True)
            else:
                self.df = self.df.iloc[val_idx].reset_index(drop=True)
        
        # Define feature columns
        self.numerical_features = [
            'udps.src2dst_last', 'udps.dst2src_last', 'udps.src2dst_pkts_data',
            'udps.dst2src_pkts_data', 'src2dst_bytes', 'dst2src_bytes', 'udps.diff_dst_src_first'
        ]
        
        self.categorical_features = ['udps.protocol']
        
        # Scale numerical features
        if scaler is None:
            self.scaler = StandardScaler()
            self.df[self.numerical_features] = self.scaler.fit_transform(self.df[self.numerical_features])
        else:
            self.scaler = scaler
            self.df[self.numerical_features] = self.scaler.transform(self.df[self.numerical_features])
        
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
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['udps.timestamp*udps.protocol'].str.split('*').str[0], unit='ms')
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
    def _encode_categorical_features(self):
        """Encode categorical features."""
        # Encode protocol
        if self.opt.categorical_encoding == 'one-hot':
            # One-hot encode protocol
            protocol_dummies = pd.get_dummies(self.df['udps.protocol'], prefix='protocol')
            self.df = pd.concat([self.df, protocol_dummies], axis=1)
            
            # Update categorical features list
            self.categorical_columns = list(protocol_dummies.columns)
        else:
            # We'll handle embedding in the model
            self.df['protocol_encoded'] = self.df['udps.protocol'].astype('category').cat.codes
            self.categorical_columns = ['protocol_encoded']
    
    def _engineer_time_features(self):
        """Extract features from timestamp."""
        # Extract time features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['minute'] = self.df['timestamp'].dt.minute
        self.df['second'] = self.df['timestamp'].dt.second
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        
        # Normalize time features
        time_features = ['hour', 'minute', 'second', 'day_of_week']
        self.df['hour'] = self.df['hour'] / 23.0
        self.df['minute'] = self.df['minute'] / 59.0
        self.df['second'] = self.df['second'] / 59.0
        self.df['day_of_week'] = self.df['day_of_week'] / 6.0
        
        # Add to numerical features
        self.numerical_features.extend(time_features)
    
    def _encode_target(self):
        """Encode target variable."""
        # Map categories to numerical values
        category_mapping = {
            'Benign_traffic': 0,
            'Benign_HH': 1,
            'Malign_HH': 2,
            'Unknown': 3
        }
        
        self.df['target'] = self.df['category'].map(category_mapping)
        
    def _create_sequences(self):
        """Create sequences for training."""
        # Prepare feature columns
        feature_cols = self.numerical_features + self.categorical_columns
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        # Use sliding window to create sequences
        for i in tqdm(range(0, len(self.df) - self.opt.seq_length, self.opt.stride)):
            seq = self.df.iloc[i:i+self.opt.seq_length][feature_cols].values
            target = self.df.iloc[i+self.opt.seq_length-1]['target']
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        # Convert to numpy arrays
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        
        print(f"Created {len(self.sequences)} sequences of length {self.opt.seq_length}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(self.targets[idx], dtype=torch.long)

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
    # Create training dataset
    train_dataset = NetworkTrafficDataset(opt, opt.train_file, is_train=True, val_split=None)
    
    # Create validation dataset using training data and split
    val_dataset = NetworkTrafficDataset(
        opt, opt.train_file, scaler=train_dataset.scaler, is_train=False, val_split=opt.val_split
    )
    
    # Create test dataset
    test_dataset = NetworkTrafficDataset(
        opt, opt.test_file, scaler=train_dataset.scaler, is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset