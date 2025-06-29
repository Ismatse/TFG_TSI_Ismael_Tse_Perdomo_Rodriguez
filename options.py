import argparse
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Time Series Forecasting with Transformer Models")
        self.initialized = False
        
    def initialize(self):
        # Basic parameters
        self.parser.add_argument('--model', type=str, default='plain_transformer', 
                                 choices=['plain_transformer'], 
                                 help='Model architecture to use')
        
        # Data parameters
        self.parser.add_argument('--train_file', type=str, default='datasets/train.csv',
                                 help='Path to the training dataset')
        self.parser.add_argument('--test_file', type=str, default='datasets/test.csv',
                                 help='Path to the testing dataset')
        self.parser.add_argument('--val_split', type=float, default=0.2,
                                 help='Validation split from the training data (0.0-1.0)')
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Batch size for training and validation')
        self.parser.add_argument('--seq_length', type=int, default=50,
                                 help='Number of time steps to consider as a sequence')
        self.parser.add_argument('--stride', type=int, default=10,
                                 help='Stride for sliding window in sequence creation')

        # Model parameters
        self.parser.add_argument('--embedding_dim', type=int, default=128,
                                 help='Embedding dimension for transformer model')
        self.parser.add_argument('--num_heads', type=int, default=8,
                                 help='Number of attention heads in transformer')
        self.parser.add_argument('--num_encoder_layers', type=int, default=6,
                                 help='Number of encoder layers in transformer')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout rate')
        self.parser.add_argument('--use_torch_transformer', action='store_true',
                                 help='Use PyTorch built-in transformer implementation')
        
        # Training parameters
        self.parser.add_argument('--epochs', type=int, default=20,
                                 help='Number of training epochs')
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                 help='Learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                                 help='Weight decay for optimizer')
        self.parser.add_argument('--checkpoint_interval', type=int, default=5,
                                 help='Save checkpoint every N epochs')
        
        # Paths
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                                 help='Directory to save checkpoints')
        self.parser.add_argument('--log_dir', type=str, default='./logs',
                                 help='Directory to save logs')
        
        # Feature engineering options
        self.parser.add_argument('--time_feature_engineering', action='store_true', default=True,
                                 help='Extract time features from timestamp')
        self.parser.add_argument('--categorical_encoding', type=str, default='one-hot',
                                 choices=['one-hot', 'embedding'],
                                 help='Method to encode categorical features')
        
        # Device
        self.parser.add_argument('--device', type=str, default='',
                                 help='Device to use (leave empty for auto-detection)')
        
        self.initialized = True
        return self.parser
        
    def parse(self):
        if not self.initialized:
            self.initialize()
        
        opt = self.parser.parse_args()
        
        # Auto-detect device if not specified
        if not opt.device:
            import torch
            opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        os.makedirs(opt.checkpoint_dir, exist_ok=True)
        os.makedirs(opt.log_dir, exist_ok=True)
        
        # Print options
        self.print_options(opt)
        
        return opt
    
    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {default}]'
            message += f'{str(k):>25}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)
        
        # Save to file
        file_dir = os.path.join(opt.log_dir, f'{opt.model}_opts.txt')
        with open(file_dir, 'w') as opt_file:
            opt_file.write(message)