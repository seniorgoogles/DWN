#!/bin/bash

# ==============================================================================
# Quick Start Script for JSC Complex Pipeline
# ==============================================================================

echo "==============================================================================="
echo "                   JSC COMPLEX - LUT PRUNING PIPELINE"
echo "==============================================================================="
echo ""
echo "This script will:"
echo "  1. Train a dense neural network"
echo "  2. Prune it down to 6 inputs per neuron (for 6-input LUTs)"
echo "  3. Fine-tune the pruned model"
echo "  4. Train a LUT-based model with fixed connections"
echo ""
echo "Expected duration: ~30-40 minutes"
echo "Expected final accuracy: ~70-72%"
echo ""
echo "==============================================================================="
echo ""

# Check if CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ ERROR: CUDA not available!"
    echo "   This script requires a GPU with CUDA support."
    exit 1
fi

echo "✅ CUDA available"
echo ""

# Check if torch_dwn is installed
if ! python3 -c "import torch_dwn" 2>/dev/null; then
    echo "❌ ERROR: torch_dwn not installed!"
    echo "   Please install it first:"
    echo "   pip install torch-dwn"
    exit 1
fi

echo "✅ torch_dwn installed"
echo ""

# Show current directory
echo "Working directory: $(pwd)"
echo ""

# Ask for confirmation
echo "Ready to start training?"
echo "Press Enter to continue, or Ctrl+C to cancel..."
read

echo ""
echo "==============================================================================="
echo "                          STARTING TRAINING"
echo "==============================================================================="
echo ""

# Run the main script
python3 jsc_complex.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================================================="
    echo "                          ✅ SUCCESS!"
    echo "==============================================================================="
    echo ""
    echo "Generated files:"
    echo "  ✓ dwn_lut_model_6inputs.pth         - Trained LUT model"
    echo "  ✓ pruned_connections_6inputs.pkl    - Connection mappings"
    echo "  ✓ pruned_models/jsc_pruned_final.pth - Pruned linear model"
    echo ""
    echo "Next steps:"
    echo "  • Load the LUT model for inference"
    echo "  • Export to FPGA synthesis tools"
    echo "  • Run hardware simulation"
    echo ""
else
    echo ""
    echo "==============================================================================="
    echo "                          ❌ TRAINING FAILED"
    echo "==============================================================================="
    echo ""
    echo "Check the error messages above for details."
    echo ""
    exit 1
fi
