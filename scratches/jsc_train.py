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

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size):
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        
        for i in range(0, n_samples, batch_size):
            
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)
            
            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            pred_train = outputs.argmax(dim=1)
            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)
        
        print(f"Epochs: {epoch}/epochs")
        train_acc = correct_train / total_train
        #test_acc = evaluate(model, x_test, y_test)

        scheduler.step()
        
        test_acc = evaluate(model, x_test, y_test)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

############################################

device = 'cuda:0'
epochs = 2
luts_num = 5
luts_inp_num = 2

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

#print(x_train.size(), x_test.size())

lut_layer = dwn.LUTLayer(x_train.size(1), luts_num, n=luts_inp_num, mapping='learnable')
group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)

model = nn.Sequential(
    lut_layer,
    group_sum
).cuda(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=100)
#print(f"{lut_layer.luts}=")
#print(f"{lut_layer.mapping.weights.argmax(dim=0)}=")

# Save model
torch.save(model.state_dict(), 'model.pth')

# Print current working directory
print(os.getcwd())

# Load model
model = nn.Sequential(
    lut_layer,
    group_sum
).cuda(device)

model.load_state_dict(torch.load('model.pth'))

model.eval()

with open('model_config.txt', 'w') as f:
    f.write(f"{luts_num=}\n")
    f.write(f"{luts_inp_num=}")

# Write lut_layer.luts and lut_layer.mapping.weights.argmax(dim=0) to a file as csv
with open('luts_data.txt', 'w') as f:
    for index, lut in enumerate(lut_layer.luts):
        # Convert tensor to numpy
        lut = lut.cpu().detach().numpy()
        print(lut.shape)
        # Iterate over each element in the tensor
        for index in range(lut.size):
            f.write(f"{1 if lut[index] > 0 else 0}")
        
        if index == len(lut_layer.luts) - 1:
            f.write("\n")
                
with open('mapping.txt', 'w') as f:
    for mapping in lut_layer.mapping.weights.argmax(dim=0):
        f.write(f"{mapping}")
        if mapping != lut_layer.mapping.weights.argmax(dim=0)[-1]:
            f.write(";")
        
#print(f"{lut_layer.luts}=")
#print(f"{lut_layer.mapping.weights.argmax(dim=0)}=")
