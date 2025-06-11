#!/bin/bash

# This script runs the experiments for the paper.
# It runs the experiments for the paper.
# Usage: ./run_experiments.sh
# Ensure the script is run from the root directory
# of the repository.
# Check if the script is run from the root directory

# List of thermometer encoders
THERMOMETER_ENCODERS=(
#    "uniform_thermometer"
#    "gaussian_thermometer"
    "distributive_thermometer"
#    "adaptive_dense_thermometer"
#    "adaptive_variance_thermometer"
)

THERMOMETER_OUTPUT_BITS=(
    "100"
    "120"
    "140"
    "160"
    "180"
    "200"
)

THERMOMETER_QUANT_BITS=(
     "-1"
    "6"
    "8"
    "10"
    "16"
    "32"
)

LUTS_NUM=(
    "10"
    "20"
    "30"
    "40"
    "50"
    "100"
    "200"
)

# Run experiments for each thermometer encoder
for encoder in "${THERMOMETER_ENCODERS[@]}"; do
    for luts in "${LUTS_NUM[@]}"; do
        for output_bits in "${THERMOMETER_OUTPUT_BITS[@]}"; do
            for quant_bits in "${THERMOMETER_QUANT_BITS[@]}"; do
                echo "Running experiment with $encoder, output bits: $output_bits, quant bits: $quant_bits"
                python3 primitive_jsc.py \
                    --luts-num $luts \
                    --thermometer $encoder \
                    --thermometer-bits $output_bits \
                    --quant-bits $quant_bits \
                    --epochs 10
            done
        done
    done
done