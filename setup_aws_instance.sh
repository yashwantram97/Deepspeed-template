#!/bin/bash
# AWS Instance Setup Script for g4dn.12xlarge
# Run this after SSHing into your fresh AWS instance
# Usage: bash setup_aws_instance.sh

set -e

echo "=================================="
echo "AWS g4dn.12xlarge Setup Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        echo -e "${YELLOW}Warning: This script is designed for Ubuntu. Your OS: $ID${NC}"
    fi
fi

# Step 1: Update system
echo -e "${GREEN}[1/6] Updating system packages...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install essential tools
echo -e "${GREEN}[2/6] Installing essential tools...${NC}"
sudo apt-get install -y \
    git \
    tmux \
    htop \
    nvtop \
    wget \
    curl \
    vim \
    build-essential

# Step 3: Verify GPU drivers
echo -e "${GREEN}[3/6] Verifying GPU drivers...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"
    nvidia-smi
else
    echo -e "${RED}✗ NVIDIA drivers not found. Installing...${NC}"
    sudo apt-get install -y nvidia-driver-525
    echo -e "${YELLOW}Please reboot the instance and run this script again.${NC}"
    exit 1
fi

# Step 4: Setup Python environment
echo -e "${GREEN}[4/6] Setting up Python environment...${NC}"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DeepSpeed
echo "Installing DeepSpeed..."
pip install deepspeed

# Install other ML libraries
echo "Installing ML libraries..."
pip install transformers datasets accelerate tqdm

# Install development tools
pip install jupyter ipython black flake8

# Step 5: Verify installations
echo -e "${GREEN}[5/6] Verifying installations...${NC}"

echo "Python version:"
python3 --version

echo ""
echo "PyTorch version and CUDA:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')"

echo ""
echo "DeepSpeed report:"
ds_report

# Step 6: Setup directory structure
echo -e "${GREEN}[6/6] Setting up workspace...${NC}"

# Create common directories
mkdir -p ~/datasets
mkdir -p ~/checkpoints
mkdir -p ~/logs

# Setup git
echo ""
read -p "Do you want to configure git? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter your git username: " git_username
    read -p "Enter your git email: " git_email
    git config --global user.name "$git_username"
    git config --global user.email "$git_email"
    echo -e "${GREEN}✓ Git configured${NC}"
fi

# Clone repository (optional)
echo ""
read -p "Do you want to clone your repository now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "Enter repository URL: " repo_url
    git clone "$repo_url"
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

# Setup tmux configuration
echo -e "${GREEN}Setting up tmux...${NC}"
cat > ~/.tmux.conf << 'EOF'
# Tmux configuration for better usability
set -g mouse on
set -g history-limit 10000
set -g base-index 1
setw -g pane-base-index 1

# Status bar
set -g status-bg colour235
set -g status-fg colour136
set -g status-left '#[fg=green](#S) '
set -g status-right '#[fg=yellow]#(hostname) #[fg=white]%H:%M'
EOF

echo -e "${GREEN}✓ Tmux configured${NC}"

# Create helpful aliases
echo -e "${GREEN}Setting up bash aliases...${NC}"
cat >> ~/.bashrc << 'EOF'

# Helpful aliases for ML training
alias gpu='watch -n 1 nvidia-smi'
alias gpustat='nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
alias trains='tmux attach -t training || tmux new -s training'
alias checkpoints='ls -lh ~/checkpoints/'

# Quick Python check
alias pycheck='python3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPUs: {torch.cuda.device_count()}\")"'
EOF

source ~/.bashrc

# Final summary
echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "System Information:"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  Python: $(python3 --version)"
echo "  GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1) x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo ""
echo "Installed Packages:"
echo "  ✓ PyTorch (with CUDA)"
echo "  ✓ DeepSpeed"
echo "  ✓ Transformers"
echo "  ✓ Datasets"
echo "  ✓ Development tools"
echo ""
echo "Useful Commands:"
echo "  gpu          - Watch GPU usage"
echo "  gpustat      - Detailed GPU stats"
echo "  trains       - Attach to training tmux session"
echo "  checkpoints  - List saved checkpoints"
echo "  pycheck      - Check Python/PyTorch setup"
echo ""
echo "Next Steps:"
echo "  1. Navigate to your project: cd Template"
echo "  2. Install project dependencies: pip install -r requirements.txt"
echo "  3. Test GPU setup: ./run_on_g4dn.sh 1"
echo "  4. Start training: ./run_on_g4dn.sh 2"
echo ""
echo "Important: Use tmux for long-running jobs!"
echo "  tmux new -s training     # Create session"
echo "  Ctrl+B, then D           # Detach"
echo "  tmux attach -t training  # Reattach"
echo ""
echo -e "${YELLOW}Don't forget to STOP the instance when done to save money!${NC}"
echo "=================================="
