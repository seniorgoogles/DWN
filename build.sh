#!/bin/bash
set -e  # Exit on error

# Use all CPU cores for parallel compilation of C++/CUDA extensions
export MAX_JOBS=$(nproc)

# Clean up (don't fail if dist doesn't exist)
rm -rf dist

# Uninstall existing package
pip3 uninstall -y torch_dwn || true

# Build with parallel jobs
python3 -m build --no-isolation

# Install with force reinstall
pip3 install dist/*whl --force-reinstall
