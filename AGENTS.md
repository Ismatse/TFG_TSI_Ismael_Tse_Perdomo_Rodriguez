# AGENTS.md - Development Guidelines

## Build/Test Commands
- **Training**: `python main.py` or `bash run_train.sh`
- **Plain Transformer**: `python main.py --model plain_transformer`
- **Informer**: `python main.py --model informer`
- **Single test run**: `python main.py --epochs 1 --batch_size 32 --model [plain_transformer|informer]`
- **Install dependencies**: `pip install -r requirements.txt`

## Project Structure
This is a PyTorch-based transformer project for time series forecasting on network traffic data.
Supports two architectures: Plain Transformer and Informer.

Logs and checkpoints are organized by architecture:
- `logs/[architecture]/run_timestamp/`
- `checkpoints/[architecture]/run_timestamp/`

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party (torch, numpy, pandas), then local imports
- Use `import torch.nn as nn` convention
- Group imports with blank lines between categories

### Naming Conventions
- Snake_case for variables, functions, files: `train_loader`, `create_data_loaders()`
- PascalCase for classes: `PlainTransformer`, `Informer`, `PositionalEncoding`
- ALL_CAPS for constants
- Descriptive names: `embedding_dim` not `dim`

### Code Organization
- Use docstrings for classes and complex functions
- Keep functions focused and modular
- Organize related functionality in modules (e.g., `plain_transformer/`, `informer/`)
- Use `opt` object for configuration parameters

### Error Handling
- Use `raise ValueError()` for invalid inputs
- Check file existence with `os.path.exists()` before loading
- Print warnings for non-critical issues

No existing Cursor/Copilot rules found in this repository.