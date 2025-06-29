# Time Series Forecasting with Transformers

This project implements a transformer architecture for time series forecasting on network traffic data. The goal is to forecast traffic categories based on network metrics.

## Problem Statement

The project focuses on forecasting the 'category' feature in network traffic datasets. These categories include:
- 'Benign_traffic' (labeled as 0)
- 'Benign_HH' (labeled as 1)
- 'Malign_HH' (labeled as 2)
- 'Unknown' (labeled as 3)

The datasets contain network traffic information including packets, bytes, and other metrics from a network line.

## Project Structure

The project is organized with a modular design to support multiple transformer architectures:

```
├── datasets/
│   ├── train.csv
│   └── test.csv
├── plain_transformer/
│   ├── __init__.py
│   ├── dataset.py     # Dataset handling and preprocessing
│   ├── model.py       # Plain transformer architecture
│   ├── trainer.py     # Training and evaluation
│   └── utils.py       # Helper functions and visualization
├── main.py            # Main script to run training and evaluation
├── options.py         # Command-line argument parser
└── requirements.txt   # Package dependencies
```

## Data Description

The datasets are located in the `datasets/` directory and include:
- `train.csv`: Training dataset (~2M rows)
- `test.csv`: Testing dataset (~2M rows)

The key features include:
- `udps.timestamp`: Unix timestamp (used as time dimension)
- `udps.protocol`: Network protocol
- Various network metrics: bytes, packets, time differentials

## Implementation Details

### Architecture

The project implements a Transformer architecture with:
- Sequence-based time series processing
- Positional encoding for temporal information
- Multi-head self-attention mechanism
- Classification head for category prediction

### Key Components

1. **Data Handling**:
   - Timestamp handling and feature engineering
   - Standardization of numerical features
   - Categorical feature encoding
   - Sequence creation with sliding window approach

2. **Model Architecture**:
   - Feature embedding layer
   - Positional encoding
   - Transformer encoder layers (custom or PyTorch's built-in)
   - Output classification layer

3. **Training Pipeline**:
   - Data loading with train/validation splitting
   - GPU acceleration
   - Learning rate scheduling
   - Gradient clipping
   - Model checkpointing

4. **Evaluation**:
   - Classification metrics (accuracy, precision, recall, F1)
   - Confusion matrix visualization
   - Training history plots

## Hardware Requirements

The model is designed to run on:
- NVIDIA RTX 4080 SUPER GPU (16GB VRAM)
- Batch processing to manage the large dataset size

## Usage

To train the model with default parameters:
```
python main.py
```

### Command-line Arguments

Key parameters can be adjusted through command-line arguments:

```
python main.py --epochs 30 --batch_size 128 --lr 0.0001
```

Important arguments:
- `--model`: Model architecture (currently only 'plain_transformer')
- `--train_file`: Path to training dataset
- `--test_file`: Path to test dataset
- `--val_split`: Validation split ratio (0.0-1.0)
- `--batch_size`: Batch size for training
- `--seq_length`: Sequence length for time series
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--checkpoint_interval`: Save checkpoint every N epochs
- `--use_torch_transformer`: Use PyTorch's built-in transformer implementation

For a complete list of arguments:
```
python main.py --help
```

## Output

The model produces:
- Trained model checkpoints in `./checkpoints/`
- Training visualization plots in `./logs/`
- Performance metrics including accuracy and classification report