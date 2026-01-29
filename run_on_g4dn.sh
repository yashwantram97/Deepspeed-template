#!/bin/bash
# Quick start script for g4dn.12xlarge (4x T4 GPUs)
# Usage: ./run_on_g4dn.sh [config_number]

set -e

echo "=================================="
echo "DeepSpeed Training on g4dn.12xlarge"
echo "4x NVIDIA T4 GPUs (16GB each)"
echo "=================================="
echo ""

# Check if running on correct instance
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    echo "Detected: $GPU_COUNT x $GPU_NAME"
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. Are you on a GPU instance?"
    echo ""
fi

# Configuration selection
CONFIG=${1:-1}

case $CONFIG in
    1)
        echo "Configuration 1: Quick Test (DistilGPT2)"
        echo "  Model: distilgpt2 (82M params)"
        echo "  Batch: 32, Seq Length: 128"
        echo "  Expected Memory: ~1GB/GPU (6% usage)"
        echo "  Expected Time: ~2 min/epoch"
        echo ""
        
        deepspeed --num_gpus=4 main.py \
            --deepspeed_config config/deepspeed/zero-2.json \
            --model_name distilgpt2 \
            --batch_size 32 \
            --max_length 128 \
            --num_epochs 1 \
            --max_train_steps 100 \
            --max_eval_steps 20
        ;;
        
    2)
        echo "Configuration 2: Medium Model (GPT2-Medium)"
        echo "  Model: gpt2-medium (355M params)"
        echo "  Batch: 64, Seq Length: 256"
        echo "  Expected Memory: ~4GB/GPU (25% usage)"
        echo "  Expected Time: ~8 min/epoch"
        echo ""
        
        deepspeed --num_gpus=4 main.py \
            --deepspeed_config config/deepspeed/zero-2.json \
            --model_name gpt2-medium \
            --batch_size 64 \
            --max_length 256 \
            --num_epochs 3 \
            --save_checkpoint \
            --output_dir ./checkpoints/gpt2-medium
        ;;
        
    3)
        echo "Configuration 3: Large Model (GPT2-Large)"
        echo "  Model: gpt2-large (774M params)"
        echo "  Batch: 32, Seq Length: 512"
        echo "  Expected Memory: ~8GB/GPU (50% usage)"
        echo "  Expected Time: ~15 min/epoch"
        echo ""
        
        deepspeed --num_gpus=4 main.py \
            --deepspeed_config config/deepspeed/zero-3.json \
            --model_name gpt2-large \
            --batch_size 32 \
            --max_length 512 \
            --num_epochs 3 \
            --save_checkpoint \
            --output_dir ./checkpoints/gpt2-large
        ;;
        
    4)
        echo "Configuration 4: XL Model - Push the Limits (GPT2-XL)"
        echo "  Model: gpt2-xl (1.5B params)"
        echo "  Batch: 16, Seq Length: 512"
        echo "  Expected Memory: ~12GB/GPU (75% usage)"
        echo "  Expected Time: ~40 min/epoch"
        echo ""
        
        deepspeed --num_gpus=4 main.py \
            --deepspeed_config config/deepspeed/zero-3.json \
            --model_name gpt2-xl \
            --batch_size 16 \
            --max_length 512 \
            --num_epochs 3 \
            --save_checkpoint \
            --output_dir ./checkpoints/gpt2-xl
        ;;
        
    *)
        echo "Invalid configuration number: $CONFIG"
        echo ""
        echo "Available configurations:"
        echo "  1 - Quick Test (DistilGPT2, ~2min)"
        echo "  2 - Medium Model (GPT2-Medium, ~8min/epoch)"
        echo "  3 - Large Model (GPT2-Large, ~15min/epoch)"
        echo "  4 - XL Model (GPT2-XL, ~40min/epoch)"
        echo ""
        echo "Usage: ./run_on_g4dn.sh [1|2|3|4]"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
