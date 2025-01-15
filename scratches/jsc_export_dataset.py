import os
import shutil
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

dataset = openml.datasets.get_dataset(42468)
df_features, df_labels, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
features = df_features.values.astype(np.float32)
label_names = list(df_labels.unique())
labels = np.array(df_labels.map(lambda x : label_names.index(x)).values)
num_output = labels.max() + 1

x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=42)

thermometer = dwn.DistributiveThermometer(200).fit(x_train)
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)

y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)


# argparse
# -i input mapping file
# -n num_of_samples
# -d dataset train or test
# -o output_dir

parser = argparse.ArgumentParser(description='Export dataset to JSC format')
parser.add_argument('-i', '--input', type=str, help='Input mapping file')
parser.add_argument('-n', '--num_of_samples', type=int, help='Number of samples to export')
parser.add_argument('-d', '--dataset', type=str, help='Dataset train or test')
parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
args = parser.parse_args()

input_mapping_file = args.input
num_of_samples = args.num_of_samples
dataset = args.dataset
output_dir = args.output_dir

# Load mapping file
mapping = pd.read_csv(input_mapping_file, sep=';')

# Check if output folder exists, if so, delete it and create a new one
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

dataset_x = x_train if dataset == 'train' else x_test
dataset_y = y_train if dataset == 'train' else y_test

if num_of_samples > len(dataset_x):
    num_of_samples = len(dataset_x)
    
with open(f"{output_dir}/dataset.txt", 'a') as f:
    for i in range(num_of_samples):
        
        sample = dataset_x[i:i+1].cpu().flatten().numpy()
        result = dataset_y[i:i+1].cpu().flatten().numpy()
        result = dataset_y[i:i+1]
            
        # Convert mapping to list
        mapping_list = mapping.to_dict().keys()
        mapping_list = [int(key) for key in mapping_list]
        
        for index in range(len(mapping_list)):
            inp_map = mapping_list[index]
            
            # Write sample to file
            f.write(f"{int(sample[inp_map])};")
            
        f.write(f"{result.item()}")
        
        if i != num_of_samples - 1:
            f.write("\n")
