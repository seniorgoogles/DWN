#!/bin/bash

# Standard Thermometer
echo "Testing Standard Thermometer..."
python3 jsc.py \
  --experiment-mode grid \
  --thermometer_type standard \
  --thermometer-bits-list 10 20 40 80 100 120 140 160 180 200 \
  --output_folder experiments_standard_$(date +%Y%m%d_%H%M%S)

# Distributive Thermometer  
echo "Testing Distributive Thermometer..."
python3 jsc.py \
  --experiment-mode grid \
  --thermometer_type distributive \
  --thermometer-bits-list 10 20 40 80 100 120 140 160 180 200 \
  --output_folder experiments_distributive_$(date +%Y%m%d_%H%M%S)

echo "All experiments completed!"

python3 jsc.py \
  --experiment-mode grid \
  --thermometer_type distributive \
  --thermometer-bits 200 \
  --quant-bits-list 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --output_folder experiments_quant_distributive_$(date +%Y%m%d_%H%M%S)

python3 jsc.py \
  --experiment-mode grid \
  --thermometer_type standard \
  --thermometer-bits 200 \
  --quant-bits-list 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --output_folder experiments_quant_standard_$(date +%Y%m%d_%H%M%S)
