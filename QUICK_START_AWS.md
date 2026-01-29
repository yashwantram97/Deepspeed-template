# 🚀 Quick Start Guide - g4dn.12xlarge

**Instance**: g4dn.12xlarge | **GPUs**: 4x NVIDIA T4 (16GB) | **Cost**: ~$1.17/hr (spot)

---

## ⚡ Super Quick Start (3 Steps)

### 1️⃣ Launch Instance
- **AWS Console** → EC2 → Launch Instance
- **AMI**: Deep Learning AMI GPU PyTorch 2.1 (Ubuntu 20.04)
- **Type**: `g4dn.12xlarge`
- **Spot**: ✅ YES (save 70%!)
- **Storage**: 150 GB
- **Security**: SSH from your IP

### 2️⃣ Setup (5 minutes)
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Run setup script
bash setup_aws_instance.sh

# Clone your repo
git clone <your-repo-url>
cd Template
```

### 3️⃣ Train! 🎉
```bash
# Quick test (2 minutes)
./run_on_g4dn.sh 1

# Full training (GPT2-Medium, ~8 min/epoch)
./run_on_g4dn.sh 2
```

---

## 📋 Available Training Configurations

Run with: `./run_on_g4dn.sh [1|2|3|4]`

| Config | Model | Params | Batch | Memory/GPU | Time/Epoch |
|--------|-------|--------|-------|------------|------------|
| **1** | distilgpt2 | 82M | 32 | 1 GB (6%) | 2 min |
| **2** | gpt2-medium | 355M | 64 | 4 GB (25%) | 8 min |
| **3** | gpt2-large | 774M | 32 | 8 GB (50%) | 15 min |
| **4** | gpt2-xl | 1.5B | 16 | 12 GB (75%) | 40 min |

---

## 🎯 Recommended Workflow

### First Time Setup
```bash
# 1. Launch spot instance (save 70%)
# 2. SSH in
ssh -i key.pem ubuntu@<ip>

# 3. Run setup
bash setup_aws_instance.sh

# 4. Clone repo
git clone <repo-url>
cd Template
```

### Every Session
```bash
# 1. Connect
ssh -i key.pem ubuntu@<ip>

# 2. Use tmux (survives disconnects!)
tmux new -s training

# 3. Monitor GPUs (in separate pane)
./monitor_gpu.sh

# 4. Run training (Ctrl+B, C for new pane)
./run_on_g4dn.sh 2

# 5. Detach safely (Ctrl+B, D)
# Training keeps running!

# 6. Reattach anytime
tmux attach -t training
```

### When Done
```bash
# 1. Download checkpoints
exit  # from tmux
scp -r ubuntu@<ip>:~/Template/checkpoints ./

# 2. STOP instance (AWS Console)
# Don't terminate - you can resume later!
```

---

## 💾 Cost Breakdown

### Spot Instance (~$1.17/hr)
```
Quick test (1 hr):     $1.17
Half day (4 hrs):      $4.68
Full day (8 hrs):      $9.36
Week (40 hrs):        $46.80
```

### While Stopped
```
Storage only: ~$15/month (150GB)
```

**Pro Tip**: Stop instance when not training! Only pay for storage.

---

## 🔥 Pro Tips

### 1. Always Use Tmux
```bash
tmux new -s training     # Create session
# Run your training here
Ctrl+B, then D           # Detach (safe!)
tmux attach -t training  # Reattach later
```

### 2. Monitor Everything
```bash
# Terminal 1: GPU monitoring
./monitor_gpu.sh

# Terminal 2: Training
./run_on_g4dn.sh 2
```

### 3. Optimize Batch Size
```bash
# Start here
./run_on_g4dn.sh 2

# If GPU usage < 50%, increase batch:
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --batch_size 128  # ← bigger!

# If OOM, decrease or use Stage 3:
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-3.json \
    --batch_size 32   # ← smaller
```

### 4. Save Checkpoints
```bash
# Always save for long runs!
deepspeed --num_gpus=4 main.py \
    --deepspeed_config config/deepspeed/zero-2.json \
    --save_checkpoint \
    --output_dir ./checkpoints/my_run
```

---

## 🐛 Common Issues

### "CUDA out of memory"
```bash
# Solution 1: Reduce batch size
--batch_size 8

# Solution 2: Use ZeRO Stage 3
--deepspeed_config config/deepspeed/zero-3.json

# Solution 3: Reduce sequence length
--max_length 128
```

### "nvidia-smi not found"
```bash
# Check drivers
sudo apt-get install nvidia-driver-525
sudo reboot
```

### Connection Lost
```bash
# Use tmux! Training keeps running
tmux attach -t training
```

### Slow Download
```bash
# Datasets auto-download to ~/.cache/huggingface
# First run is slower, then cached
```

---

## 📊 Monitor Commands

```bash
# Quick checks
nvidia-smi              # GPU status
htop                    # CPU/RAM
df -h                   # Disk space

# Continuous monitoring
./monitor_gpu.sh        # Custom GPU monitor
watch nvidia-smi        # Watch GPUs
nvtop                   # Interactive GPU monitor

# Aliases (after setup)
gpu                     # = watch nvidia-smi
gpustat                 # Detailed stats
pycheck                 # Check Python/CUDA
```

---

## 🎓 Learning Path

### Day 1: Get Comfortable
```bash
./run_on_g4dn.sh 1      # Quick test (2 min)
./run_on_g4dn.sh 2      # Medium model
```

### Day 2: Experiment
```bash
# Try different batch sizes
deepspeed --num_gpus=4 main.py \
    --batch_size 64 \
    --max_length 256

# Try ZeRO Stage 3
--deepspeed_config config/deepspeed/zero-3.json
```

### Day 3: Scale Up
```bash
./run_on_g4dn.sh 3      # Large model
./run_on_g4dn.sh 4      # XL model
```

---

## 📞 Support Resources

- **AWS Console**: Check instance status, stop/start
- **DeepSpeed Docs**: https://www.deepspeed.ai/
- **AWS G4 Guide**: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html

---

## ⚠️ Important Reminders

1. **ALWAYS use Spot instances** (70% cheaper!)
2. **ALWAYS use tmux** for long jobs
3. **ALWAYS save checkpoints** before stopping
4. **STOP instance when done** (not terminate!)
5. **Download results** before terminating

---

## 🎯 Your First Run (Copy-Paste Ready)

```bash
# === ON YOUR LOCAL MACHINE ===
# Launch instance, get IP, then:
export IP=<your-instance-ip>
scp -i key.pem setup_aws_instance.sh ubuntu@$IP:~/
ssh -i key.pem ubuntu@$IP

# === ON AWS INSTANCE ===
bash setup_aws_instance.sh
git clone <your-repo>
cd Template

# Start tmux session
tmux new -s training

# Split window (Ctrl+B, then ")
# Top pane: monitor
./monitor_gpu.sh

# Bottom pane (Ctrl+B, arrow down, then):
./run_on_g4dn.sh 2

# Detach safely (Ctrl+B, D)
# Training continues!

# Check progress anytime:
tmux attach -t training

# When done, download results:
exit
scp -r ubuntu@$IP:~/Template/checkpoints ./

# STOP INSTANCE in AWS Console!
```

---

**Happy Training! Don't forget to stop your instance! 🚀💰**
