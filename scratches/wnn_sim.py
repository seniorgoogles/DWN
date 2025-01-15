import os
import shutil
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch import nn

class HalfAdder():
    def __init__(self):
        pass
    
    def __call__(self, x, y):
        return x ^ y, x & y

class LookupTable():
    def __init__(self, luts_data, num_inputs):
        self.num_inputs = num_inputs
        self.luts_data = luts_data

        self.luts_output = dict()
        
        for i in range(2**num_inputs):
            self.luts_output[f"{i:0{num_inputs}b}"] = luts_data[i]
            
        
        #print(f"{self.luts_output=}")
        
    def __call__(self, x):
        return self.luts_output[x]
    
class GroupSum():
    
    def __init__(self, output_k):
        self.output_k = output_k
    
    def __call__(self, x):
        
        if len(x) > self.output_k and len(x) % self.output_k == 0:
            
            # Convert x to matrix tensor
            x = x.view(-1, int(len(x) / self.output_k))
            #print(x) 
            
            return x.sum(dim=1)
            
        elif len(x) == self.output_k:
            return x
        else:
            raise ValueError("Invalid number of inputs")
        
class WeightlessNeuralNetwork():   

    def __init__(self, num_luts_inp, luts_data):
        self.luts = []
        
        # Setup LUTs
        for lut_d in luts_data:
            self.luts.append(LookupTable(lut_d, num_luts_inp))
        

    def __call__(self, x):
        output: list = []
        offset = 0
        
        num_luts = len(self.luts)
        #print(f"{num_luts=}")
        
        for index in range(num_luts):
            tmp_inp = x[offset:offset+num_luts_inp]
            # Reverse tmp_inp
            tmp_inp = tmp_inp[::-1]

            output.append(int(self.luts[index](tmp_inp)))
            offset += num_luts_inp
        
        #print(f"{output=}")

        # Convert output to tensor  
        output = torch.tensor(output, dtype=torch.float32)      
        return GroupSum(5)(output)
    
def convert_to_wnn_data(sample: torch.Tensor, mapping):
    
    # Convert tensor to str
    output_str = ""
    
    # Convert real tensor to integer list
    sample_list = sample.cpu().flatten().numpy().tolist()
    sample_list = [int(x) for x in sample_list]
    
    for i in range(len(mapping)):
        output_str += str(sample_list[mapping[i]])
        
    return output_str

def run_inference(model, wnn, samples_num, model_checkpoint_path = None, device = 'cuda:0'):
    
    # Preload weights
    if model_checkpoint_path is not None:
        model.load_state_dict(torch.load(model_checkpoint_path))
    
    model.eval()
    
    acc = 0.0

    # Run inference    
    for sample_index in range(samples_num):
        sample = x_train[sample_index:sample_index+1].cuda(device)
        result = y_train[sample_index:sample_index+1].cuda(device)

        model_out = model(sample)
        
        # Compare result with ground truth
        #print(f"Result: {result} | Prediction: {model_out.argmax()}")
        
        acc += 1 if result == model_out.argmax() else 0
        
    print(f"Accuracy: {acc/samples_num*100}%")

def read_model_config(model_config_filepath):
    num_luts = -1
    num_luts_inp = -1
    
    with open(model_config_filepath, 'r') as f:
        num_luts = int((f.readline()).replace('luts_num=', '').replace('\n', ''))
        num_luts_inp = int((f.readline()).replace('luts_inp_num=', ''))
        
    return num_luts, num_luts_inp

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
# Path mapping file
# Path lut data file
# Path model configuratoin
# Number of inference samples
# Dataset train or test

device = "cuda:0"

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

parser = argparse.ArgumentParser(description='Export dataset to JSC format')
parser.add_argument('-c', '--model_config', type=str, help='Model configuration')
parser.add_argument('-l', '--lut_data', type=str, help='LUT data file')
parser.add_argument('-m', '--mapping', type=str, help='Input mapping file')

parser.add_argument('-n', '--num_of_samples', type=int, help='Number of samples to export')
parser.add_argument('-d', '--dataset', type=str, help='Dataset train or test')
parser.add_argument('--model_checkpoint_path', type=str, help='Model checkpoint path')

##### Argparser
args = parser.parse_args()

model_config_filepath = args.model_config
input_mapping_filepath = args.mapping
num_samples = args.num_of_samples
dataset = args.dataset
model_checkpoint_path = args.model_checkpoint_path

input_mapping = read_mapping(input_mapping_filepath)
num_luts, num_luts_inp = read_model_config(model_config_filepath)
luts_data = read_lut_data(args.lut_data)

##### Execution 
#print(f"{input_mapping=}")
#print(f"Num luts: {num_luts}, Num luts inp: {num_luts_inp}")
#print(f"Lut data: {luts_data}")

### Model
wnn = WeightlessNeuralNetwork(num_luts_inp, luts_data)

lut_layer = dwn.LUTLayer(x_train.size(1), num_luts, n=num_luts_inp, mapping='learnable')
group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)

model = nn.Sequential(
    lut_layer,
    group_sum
).cuda(device)

model.load_state_dict(torch.load(model_checkpoint_path))

model.eval()

### Inference

acc = 0.0

for sample_index in range(len(x_train)):
    
    sample_data = x_train[sample_index:sample_index+1]
    sample_output = y_train[sample_index:sample_index+1].item()
    
    sample_wnn_inp = convert_to_wnn_data(sample_data, input_mapping)
    # Reverse string
    
    print(f"{sample_wnn_inp=}")
    print(f"{sample_wnn_inp[::-1]=}")
    
    output_model = model(sample_data.cuda(device)).argmax().item()
    output_wnn = wnn(sample_wnn_inp)
    
    print(f"{output_wnn=}")
    
    # Print output_wnn tensor as binary string
    output_wnn = output_wnn.argmax().item() 

    #print(f"{output_model=}, {output_wnn=}")
    input()
    


    """
    # Clear the terminal
    os.system('clear')
    """
    # If the output of the model and the wnn are the same, the inference is correct, print "Correct" in green, else print "Incorrect" in red
    if output_model != output_wnn:
        print('\033[91m' + "Incorrect" + '\033[0m')
        raise ValueError("Incorrect inference @ sample: ", sample_index)
    
    acc += 1. if output_model == sample_output else 0


# Print in green that everything is ok
print('\033[92m' + "WNN Sim implementation and Pytorch Model are equal" + '\033[0m')
print(f"Accuracy: {acc/len(x_train)*100}%")