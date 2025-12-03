#!/bin/bash
# Quick start script for running optimized CIFAR-10 LBP encoder

echo "=================================================="
echo "DWN CIFAR-10 LBP+Thermometer Encoder - Optimized"
echo "=================================================="
echo ""

# Default configuration with ALL optimizations enabled
python cifar10_lbp_encoder.py \
    --encoding-type lbp+thermometer \
    --thermo-type distributive \
    --thermometer-bits 10 \
    --epochs 25 \
    --lr 1e-2 \
    --batch-size 256 \
    --hidden-size 8000 \
    --mixed-precision \
    --estimator ste \
    --fixed-thresholds

echo ""
echo "=================================================="
echo "Training complete! Check the output above."
echo "=================================================="
