# AWS g4dn.12xlarge Setup Guide

Complete guide for setting up and running DeepSpeed training on AWS g4dn.12xlarge (4x NVIDIA T4 GPUs).

## 🖥️ Instance Specifications

```
Instance: g4dn.12xlarge
GPUs: 4x NVIDIA T4 (16GB each) = 64GB total GPU memory
vCPUs: 48
RAM: 192 GB
Storage: NVMe SSD (900 GB)
Network: Up to 50 Gbps
Cost: ~$3.91/hr on-demand, ~$1.17/hr spot (70% savings!)
```

## 📋 Step 1: Launch Instance

### Option A: AWS Console (Easier)

1. **Go to EC2 Dashboard** → Launch Instance

2. **Configure Instance**:
   ```
   Name: deepspeed-training
   AMI: Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04)
        AMI ID: ami-0c7217cdde317cfec (us-east-1)
   Instance Type: g4dn.12xlarge
   Key Pair: Create new or select existing
   ```

3. **Storage**:
   ```
   Root Volume: 150 GB gp3 (or more if you have large datasets)
   ```

4. **Network Settings**:
   ```
   Auto-assign public IP: Enable
   Security Group: Create new
     - SSH (22): Your IP only
     - Custom TCP (8888): Your IP (for Jupyter, optional)
   ```

5. **Advanced Details** (IMPORTANT for Spot):
   ```
   Request Spot Instances: YES
   Maximum price: $3.91 (on-demand price)
   Interruption behavior: Stop (not terminate!)
   ```

6. **Launch Instance**

### Option B: AWS CLI (Faster)

```bash
# Create spot instance request
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g4dn.12xlarge \
    --key-name YOUR_KEY_NAME \
    --security-group-ids YOUR_SG_ID \
    --subnet-id YOUR_SUBNET_ID \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"persistent","InstanceInterruptionBehavior":"stop","MaxPrice":"3.91"}}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=deepspeed-training}]'
```

## 🔌 Step 2: Connect to Instance

```bash
# Get instance public IP from AWS console
export INSTANCE_IP=<your-instance-public-ip>

# SSH into instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Or with port forwarding for Jupyter
ssh -i your-key.pem -L 8888:localhost:8888 ubuntu@$INSTANCE_IP
```

## ⚙️ Step 3: Verify GPU Setup

```bash
# Check GPUs are detected
nvidia-smi

# Should show 4x Tesla T4 GPUs
# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# |   0  Tesla T4            Off  | 00000000:00:1B.0 Off |                    0 |
# |   1  Tesla T4            Off  | 00000000:00:1C.0 Off |                    0 |
# |   2  Tesla T4            Off  | 00000000:00:1D.0 Off |                    0 |
# |   3  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# +-----------------------------------------------------------------------------+

# Check CUDA is working
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

## 📦 Step 4: Install Dependencies

```bash
# Update system
sudo apt-get update

# Install system dependencies
sudo apt-get install -y git tmux htop nvtop

# Clone your repository
git clone <your-repo-url>
cd Template

# Install Python packages
pip install --upgrade pip

# Install DeepSpeed and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed transformers datasets tqdm accelerate

# Or use requirements.txt
pip install -r requirements.txt

# Verify DeepSpeed installation
ds_report
```

## 🚀 Step 5: Run Training

### Quick Test Run

```bash
# Test with 2 GPUs first
deepspeed --num_gpus=2 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --num_epochs 1 \
    --max_train_steps 50

# If that works, use all 4 GPUs
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --num_epochs 1
```

### Recommended Configurations for g4dn.12xlarge

#### Configuration 1: Default (Testing)
```bash
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --model_name distilgpt2 \
    --batch_size 32 \
    --max_length 128 \
    --num_epochs 3
```
**Memory per GPU**: ~1 GB / 16 GB (6% usage)

#### Configuration 2: Better Utilization
```bash
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --model_name gpt2-medium \
    --batch_size 64 \
    --max_length 256 \
    --num_epochs 3
```
**Memory per GPU**: ~4 GB / 16 GB (25% usage)

#### Configuration 3: Push the Hardware
```bash
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-3.json \
    --model_name gpt2-large \
    --batch_size 32 \
    --max_length 512 \
    --num_epochs 3 \
    --save_checkpoint \
    --output_dir ./checkpoints/gpt2-large
```
**Memory per GPU**: ~8 GB / 16 GB (50% usage)

#### Configuration 4: Maximum Capacity
```bash
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-3.json \
    --model_name gpt2-xl \
    --batch_size 16 \
    --max_length 512 \
    --num_epochs 3 \
    --save_checkpoint
```
**Memory per GPU**: ~12 GB / 16 GB (75% usage)

## 📊 Step 6: Monitor Performance

### Terminal 1: GPU Monitoring
```bash
# Use the monitoring script
./monitor_gpu.sh

# Or use nvidia-smi directly
watch -n 1 nvidia-smi
```

### Terminal 2: System Monitoring
```bash
# Install and run htop
htop

# Or monitor with nvtop (better for GPUs)
nvtop
```

### Terminal 3: Training
```bash
# Run your training job here
deepspeed --num_gpus=4 main.py ...
```

## 🎯 Optimization Tips for T4 GPUs

### 1. Batch Size Tuning
```bash
# T4 has 16GB - you can go bigger!
# Start conservative, increase until you hit OOM

# For DistilGPT2: Can handle batch_size=128
# For GPT2: Can handle batch_size=64
# For GPT2-Medium: Can handle batch_size=32
# For GPT2-Large: Can handle batch_size=16
```

### 2. Mixed Precision
```bash
# T4 has Tensor Cores - FP16 is MUCH faster
# Your configs already have this enabled ✓
# "fp16": {"enabled": true}
```

### 3. Gradient Accumulation
```bash
# If you want effective larger batches without OOM
# Edit config: "gradient_accumulation_steps": 4
# Effective batch = batch_size × num_gpus × grad_accum_steps
# Example: 8 × 4 × 4 = 128 effective batch size
```

### 4. CPU Offloading
```bash
# g4dn.12xlarge has 192GB RAM - plenty for offloading!
# Your Stage 2/3 configs already use this ✓
# "offload_optimizer": {"device": "cpu"}
```

## 💾 Step 7: Save and Download Results

### Save Checkpoints
```bash
# Training with checkpoint saving
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --save_checkpoint \
    --output_dir ./checkpoints/run1
```

### Download to Local Machine
```bash
# From your local machine
scp -i your-key.pem -r ubuntu@$INSTANCE_IP:~/Template/checkpoints ./local_checkpoints

# Or use rsync for resume capability
rsync -avz -e "ssh -i your-key.pem" ubuntu@$INSTANCE_IP:~/Template/checkpoints ./local_checkpoints
```

## 🛑 Step 8: Stop Instance (Save Money!)

### Stop Instance (keeps data)
```bash
# From AWS Console: Select instance → Stop
# Or from CLI:
aws ec2 stop-instances --instance-ids i-xxxxx

# Cost while stopped: Only storage (~$15/month for 150GB)
```

### Terminate Instance (deletes everything)
```bash
# Only do this if you're done and have backed up everything!
aws ec2 terminate-instances --instance-ids i-xxxxx
```

## 🔄 Step 9: Resume Training (After Stop/Start)

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@$INSTANCE_IP

# Navigate to project
cd Template

# Continue training (DeepSpeed will resume from checkpoint if available)
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --output_dir ./checkpoints/run1
```

## 🐛 Troubleshooting

### GPU Not Found
```bash
# Check driver
nvidia-smi

# Reinstall if needed
sudo apt-get install -y nvidia-driver-525
sudo reboot
```

### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
--batch_size 4

# Solution 2: Use Stage 3 instead of Stage 2
--deepspeed_config config/deepspeed/zero-3.json

# Solution 3: Reduce sequence length
--max_length 128

# Solution 4: Enable gradient checkpointing (add to model)
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# If GPU util < 80%, bottleneck might be:
# - Data loading: increase num_workers in dataloader
# - CPU: batch size too small
# - Disk I/O: move data to instance storage
```

### Connection Lost
```bash
# Use tmux to keep training running
tmux new -s training

# Run training inside tmux
deepspeed --num_gpus=4 main.py ...

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## 📈 Expected Performance

### Training Speed Estimates (g4dn.12xlarge)

| Model | Batch/GPU | Seq Len | Tokens/sec | Time/Epoch* |
|-------|-----------|---------|------------|-------------|
| DistilGPT2 | 8 | 128 | ~8,000 | 2 min |
| GPT2 | 8 | 128 | ~6,000 | 3 min |
| GPT2-Medium | 4 | 256 | ~3,000 | 8 min |
| GPT2-Large | 4 | 256 | ~1,500 | 15 min |
| GPT2-XL | 2 | 512 | ~600 | 40 min |

*Approximate, on WikiText-2 dataset

## 💰 Cost Estimates

### Spot Instance Pricing (~$1.17/hr)

| Duration | Cost | What You Can Do |
|----------|------|-----------------|
| 1 hour | $1.17 | Several test runs |
| 4 hours | $4.68 | Full experimentation session |
| 8 hours | $9.36 | Complete project iteration |
| 40 hours | $46.80 | Week of development |

### Cost Optimization
```bash
# 1. Use Spot instances (70% savings) ✓
# 2. Stop instance when not using
# 3. Use tmux for long jobs (survive disconnects)
# 4. Download checkpoints and terminate when done
# 5. Consider Reserved Instances if using long-term (40-60% savings)
```

## 🎓 Next Steps

1. **Start Small**: Test with distilgpt2 first
2. **Scale Up**: Try gpt2-medium or gpt2-large
3. **Optimize**: Tune batch size and sequence length
4. **Monitor**: Watch GPU utilization, aim for >80%
5. **Save**: Always use `--save_checkpoint` for long runs
6. **Stop**: Remember to stop instance when done!

---

**Happy Training on AWS! 🚀**

For issues or questions, check:
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [AWS EC2 G4 Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html)
