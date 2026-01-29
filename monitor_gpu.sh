#!/bin/bash
# GPU Monitoring Script
# Usage: ./monitor_gpu.sh

echo "GPU Memory Monitoring - Press Ctrl+C to stop"
echo "=========================================="
echo ""

while true; do
    clear
    echo "GPU Memory Usage - $(date)"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
               --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "GPU %s: %s\n", $1, $2
        printf "  Memory: %s MB / %s MB (%.1f%%)\n", $3, $4, ($3/$4)*100
        printf "  GPU Util: %s%%\n", $5
        printf "  Temp: %s°C\n\n", $6
    }'
    
    echo "=========================================="
    echo "Per-Process GPU Memory:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
               --format=csv,noheader | \
    awk -F', ' '{printf "  PID %s (%s): %s MB\n", $1, $2, $3}'
    
    sleep 2
done
