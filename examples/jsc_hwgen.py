import argparse
import os
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import datetime
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

import torch_dwn as dwn
import hw_gen as hwgen

def run_inference(model, x_test, y_test, device):
    pass

def parse_arguments():
    
    # Model related arguments
    parser = argparse.ArgumentParser(description="Process input files for jsc_hwgen.")
    parser.add_argument("--model_weights", "-w", required=True, help="Path to the model weights file.")
    parser.add_argument("--luts_data", "-l", required=True, help="Path to the LUTs data file.")
    parser.add_argument("--mapping", "-m", required=True, help="Path to the mapping file.")
    parser.add_argument("--model_config", "-c", required=True, help="Path to the model configuration file.")

    # Thermometer and thermometer bits
    parser.add_argument("--thermometer", action="store_true", help="Use thermometer encoding.")
    parser.add_argument("--thermometer_bits", type=int, default=8, help="Number of thermometer bits.")
    
    # Output folder
    parser.add_argument("--output_dir", default="output", help="Directory to save the output files.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Use the args normally, e.g.:
    print("Model Weights File:", args.model_weights)
    print("LUTs Data File:", args.luts_data)
    print("Mapping File:", args.mapping)
    print("Model Config File:", args.model_config)

if __name__ == '__main__':
    main()