import torch
import os
from torch import nn
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_dwn as dwn

import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

    
def run_inference(model, X_dataset, y_dataset, samples_num, model_checkpoint_path = None):
    
    # Preload weights
    if model_checkpoint_path is not None:
        model.load_state_dict(torch.load(model_checkpoint_path))
    
    model.eval()
    
    acc = 0.0

    # Run inference    
    for sample_index in range(samples_num):
        sample = X_dataset[sample_index:sample_index+1].cuda(device)
        result = y_dataset[sample_index:sample_index+1].cuda(device)

        model_out = model(sample)
        
        # Compare result with ground truth
        #print(f"Result: {result} | Prediction: {model_out.argmax()}")
        
        acc += 1 if result == model_out.argmax() else 0
        
    print(f"Accuracy: {acc/samples_num*100}%")

device = 'cuda:0'

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

print(x_train.size(), x_test.size())

lut_layer = dwn.LUTLayer(x_train.size(1), 16, n=3, mapping='learnable')
group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)

# Load model
model = nn.Sequential(
    lut_layer,
    group_sum
).cuda(device)

run_inference(model, x_test, y_test, 60000)

#print(f"{lut_layer.luts}=")
#print(f"{lut_layer.mapping.weights.argmax(dim=0)}=")


