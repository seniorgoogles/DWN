''' 
Code Generator for creating WNNs in Python
'''

import os
import shutil
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def read_lut_data(lut_data_filepath):
    lut_data = []
    
    with open(lut_data_filepath, 'r') as f:
        for line in f:
            lut_data.append(line.replace('\n', ''))
            
    return lut_data

def read_mapping(mapping_filepath):
    with open(mapping_filepath, 'r') as f:
        mapping_str_list = f.readline().replace('\n', '').split(';')
    
    mapping = [int(x) for x in mapping_str_list]
    return mapping

# argparse
parser = argparse.ArgumentParser(description='Export dataset to JSC format')
#parser.add_argument('-c', '--model_config', type=str, help='Model configuration')
parser.add_argument('-l', '--lut_data', type=str, help='LUT data file')
parser.add_argument('-m', '--mapping', type=str, help='Input mapping file')

parser.add_argument('-n', '--num_of_samples', type=int, help='Number of samples to export')
parser.add_argument('-d', '--dataset', type=str, help='Dataset train or test')
#parser.add_argument('--model_checkpoint_path', type=str, help='Model checkpoint path')

args = parser.parse_args()

model_config_filepath = args.model_config
input_mapping_filepath = args.mapping