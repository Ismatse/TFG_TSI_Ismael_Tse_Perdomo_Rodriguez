#!/bin/bash

echo "Testing Plain Transformer architecture with minimal settings..."

python3 main.py \
    --model plain_transformer \
    --train_file datasets/train.csv \
    --test_file datasets/test.csv \
    --val_split 0.2 \
    --batch_size 32 \
    --seq_length 16 \
    --stride 4 \
    --num_workers 4 \
    --embedding_dim 64 \
    --num_heads 4 \
    --num_encoder_layers 2 \
    --dropout 0.1 \
    --use_torch_transformer \
    --num_classes 3 \
    --epochs 1 \
    --lr 0.001 \
    --weight_decay 1e-5 \
    --checkpoint_interval 1 \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --time_feature_engineering \
    --categorical_encoding one-hot \
    --early_stopping_patience 5