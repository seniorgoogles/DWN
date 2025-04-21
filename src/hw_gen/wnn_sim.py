import os
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch import nn


class LookupTable:
    def __init__(self, luts_data, num_inputs):
        self.num_inputs = num_inputs
        self.luts_data = luts_data

        self.luts_output = {
            f"{i:0{num_inputs}b}": luts_data[i]
            for i in range(2 ** num_inputs)
        }
        print(f"Lookup Table Outputs: {self.luts_output}")
        print(f"Lookup Table Data: {luts_data}")

    def __call__(self, x):
        return self.luts_output[x]


class GroupSum:
    def __init__(self, output_k):
        self.output_k = output_k

    def __call__(self, x):
        if len(x) > self.output_k and len(x) % self.output_k == 0:
            x = x.view(-1, int(len(x) / self.output_k))
            return x.sum(dim=1)
        elif len(x) == self.output_k:
            return x
        else:
            raise ValueError("Invalid number of inputs")


class WeightlessNeuralNetwork:
    def __init__(self, num_luts_inp, luts_data):
        self.num_luts_inp = num_luts_inp
        self.luts = [LookupTable(lut_d, num_luts_inp) for lut_d in luts_data]

    def __call__(self, x):
        output = []
        offset = 0
        num_luts = len(self.luts)

        for index in range(num_luts):
            tmp_inp = x[offset:offset + self.num_luts_inp][::-1]
            output.append(int(self.luts[index](tmp_inp)))
            offset += self.num_luts_inp
            print(f"Input to LUT {index}: {tmp_inp}")

        print(f"Weightless Neural Network output: {output}")
        output_tensor = torch.tensor(output, dtype=torch.float32)
        return GroupSum(5)(output_tensor)


def convert_to_wnn_data(sample: torch.Tensor, mapping):
    sample_list = [int(x) for x in sample.cpu().flatten().numpy().tolist()]
    
    print(f"{len(sample_list)=}")
    print(f"{len(mapping)=}")
    
    output_str = "".join(str(sample_list[i]) for i in mapping)
    
    return output_str


def run_inference(model, wnn, x_train, y_train, samples_num, device, model_checkpoint_path=None):
    if model_checkpoint_path is not None:
        model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    acc = 0.0

    for sample_index in range(samples_num):
        sample = x_train[sample_index:sample_index + 1].to(device)
        ground_truth = y_train[sample_index:sample_index + 1].to(device)
        model_out = model(sample)
        predicted = model_out.argmax()

        acc += 1 if ground_truth == predicted else 0

    print(f"Accuracy: {acc / samples_num * 100:.2f}%")
    return acc / samples_num * 100


def read_model_config(model_config_filepath):
    with open(model_config_filepath, 'r') as f:
        num_luts = int(f.readline().replace('luts_num=', '').strip())
        num_luts_inp = int(f.readline().replace('luts_inp_num=', '').strip())
    return num_luts, num_luts_inp


def read_lut_data(lut_data_filepath):
    with open(lut_data_filepath, 'r') as f:
        lut_data = [line.strip() for line in f]
    return lut_data


def read_mapping(mapping_filepath):
    with open(mapping_filepath, 'r') as f:
        mapping_str_list = f.readline().strip().split(';')
    mapping = [int(x) for x in mapping_str_list]
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Run WNN vs PyTorch model')
    parser.add_argument('-c', '--model_config', type=str, required=True, help='Model configuration file path')
    parser.add_argument('-l', '--lut_data', type=str, required=True, help='LUT data file path')
    parser.add_argument('-m', '--mapping', type=str, required=True, help='Input mapping file path')
    parser.add_argument('-n', '--num_of_samples', type=int, required=True, help='Number of samples to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['train', 'test'], required=True, help='Dataset: train or test')
    parser.add_argument('--model_checkpoint_path', type=str, help='Model checkpoint path')

    args = parser.parse_args()

    # Read config, mapping, LUTs
    input_mapping = read_mapping(args.mapping)
    num_luts, num_luts_inp = read_model_config(args.model_config)
    luts_data = read_lut_data(args.lut_data)

    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, _ = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute
    )
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    thermometer = dwn.DistributiveThermometer(200).fit(x_train)
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    device = 'cuda:0'

    # Instantiate WNN and PyTorch model
    wnn = WeightlessNeuralNetwork(num_luts_inp, luts_data)
    lut_layer = dwn.LUTLayer(x_train.size(1), num_luts, n=num_luts_inp, mapping='learnable')
    group_sum = dwn.GroupSum(k=len(label_names), tau=1)
    model = nn.Sequential(lut_layer, group_sum).to(device)

    if args.model_checkpoint_path:
        model.load_state_dict(torch.load(args.model_checkpoint_path))
    model.eval()

    data_samples, targets = (x_train, y_train) if args.dataset == 'train' else (x_test, y_test)
    num_samples = min(args.num_of_samples, len(data_samples))

    # Lists to store outputs
    wnn_input_list = []
    wnn_output_list = []
    lut_output_list = []

    # Full output with all data
    with open('_sim_output_full.txt', 'w') as f:
        for sample_index in range(num_samples):
            sample_data = data_samples[sample_index:sample_index + 1]
            sample_output = targets[sample_index].item()

            sample_wnn_inp = convert_to_wnn_data(sample_data, input_mapping)
            wnn_input_list.append(sample_wnn_inp[::-1])

            # PyTorch model output
            model_out = model(sample_data.to(device))[0].detach().cpu().numpy()
            # WNN output
            wnn_out = wnn(sample_wnn_inp).numpy()

            # LUT layer intermediate output
            lut_out = lut_layer(sample_data.to(device))[0].detach().cpu().numpy()
            lut_output_list.append(lut_out.tolist())

            f.write(f"{sample_wnn_inp} | model={model_out} | wnn={wnn_out} | lut={lut_out} | expected={sample_output}\n")

            # Check consistency
            if not np.array_equal(model_out, wnn_out):
                print('Incorrect LUT/GroupSum mapping')
                raise ValueError(f"Mismatch at sample {sample_index}")

            acc_flag = 'Correct' if model_out.argmax() == sample_output else 'Incorrect'
            print(f"Sample {sample_index}: {acc_flag} | model={model_out.argmax()} wnn={wnn_out.argmax()} expected={sample_output}")

    # Write simple outputs
    with open('_sim_output.txt', 'w') as f:
        for sample_index in range(num_samples):
            sample_data = data_samples[sample_index:sample_index + 1]
            sample_output = targets[sample_index].item()
            model_out = model(sample_data.to(device))[0].detach().cpu().numpy()
            f.write(f"{model_out.argmax()}\n")

    # Write WNN inputs
    with open('_sim_wnn_input.txt', 'w') as f:
        for inp in wnn_input_list:
            f.write(f'"{inp}",\n')

    # Write WNN outputs
    with open('_sim_wnn_output.txt', 'w') as f:
        for out in wnn_output_list:
            f.write(f'"{"".join(str(int(x)) for x in out)}",\n')

    # Write LUT layer outputs
    with open('_sim_lut_layer_output.txt', 'w') as f:
        for lut_vec in lut_output_list:
            f.write(f"{lut_vec}\n")


if __name__ == '__main__':
    main()
import os
import torch
import torch_dwn as dwn
import openml
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch import nn


class LookupTable:
    def __init__(self, luts_data, num_inputs):
        self.num_inputs = num_inputs
        self.luts_data = luts_data

        self.luts_output = {
            f"{i:0{num_inputs}b}": luts_data[i]
            for i in range(2 ** num_inputs)
        }
        print(f"Lookup Table Outputs: {self.luts_output}")
        print(f"Lookup Table Data: {luts_data}")

    def __call__(self, x):
        return self.luts_output[x]


class GroupSum:
    def __init__(self, output_k):
        self.output_k = output_k

    def __call__(self, x):
        if len(x) > self.output_k and len(x) % self.output_k == 0:
            x = x.view(-1, int(len(x) / self.output_k))
            return x.sum(dim=1)
        elif len(x) == self.output_k:
            return x
        else:
            raise ValueError("Invalid number of inputs")


class WeightlessNeuralNetwork:
    def __init__(self, num_luts_inp, luts_data):
        self.num_luts_inp = num_luts_inp
        self.luts = [LookupTable(lut_d, num_luts_inp) for lut_d in luts_data]

    def __call__(self, x):
        output = []
        offset = 0
        num_luts = len(self.luts)

        for index in range(num_luts):
            tmp_inp = x[offset:offset + self.num_luts_inp][::-1]
            output.append(int(self.luts[index](tmp_inp)))
            offset += self.num_luts_inp
            print(f"Input to LUT {index}: {tmp_inp}")

        print(f"Weightless Neural Network output: {output}")
        output_tensor = torch.tensor(output, dtype=torch.float32)
        return GroupSum(5)(output_tensor)


def convert_to_wnn_data(sample: torch.Tensor, mapping):
    sample_list = [int(x) for x in sample.cpu().flatten().numpy().tolist()]
    output_str = "".join(str(sample_list[i]) for i in mapping)
    return output_str


def run_inference(model, wnn, x_train, y_train, samples_num, device, model_checkpoint_path=None):
    if model_checkpoint_path is not None:
        model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    acc = 0.0

    for sample_index in range(samples_num):
        sample = x_train[sample_index:sample_index + 1].to(device)
        ground_truth = y_train[sample_index:sample_index + 1].to(device)
        model_out = model(sample)
        predicted = model_out.argmax()

        acc += 1 if ground_truth == predicted else 0

    print(f"Accuracy: {acc / samples_num * 100:.2f}%")
    return acc / samples_num * 100


def read_model_config(model_config_filepath):
    with open(model_config_filepath, 'r') as f:
        num_luts = int(f.readline().replace('luts_num=', '').strip())
        num_luts_inp = int(f.readline().replace('luts_inp_num=', '').strip())
    return num_luts, num_luts_inp


def read_lut_data(lut_data_filepath):
    with open(lut_data_filepath, 'r') as f:
        lut_data = [line.strip() for line in f]
    return lut_data


def read_mapping(mapping_filepath):
    with open(mapping_filepath, 'r') as f:
        mapping_str_list = f.readline().strip().split(';')
    mapping = [int(x) for x in mapping_str_list]
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Run WNN vs PyTorch model')
    parser.add_argument('-c', '--model_config', type=str, required=True, help='Model configuration file path')
    parser.add_argument('-l', '--lut_data', type=str, required=True, help='LUT data file path')
    parser.add_argument('-m', '--mapping', type=str, required=True, help='Input mapping file path')
    parser.add_argument('-n', '--num_of_samples', type=int, required=True, help='Number of samples to use')
    parser.add_argument('-d', '--dataset', type=str, choices=['train', 'test'], required=True, help='Dataset: train or test')
    parser.add_argument('--model_checkpoint_path', type=str, help='Model checkpoint path')

    args = parser.parse_args()

    # Read config, mapping, LUTs
    input_mapping = read_mapping(args.mapping)
    num_luts, num_luts_inp = read_model_config(args.model_config)
    luts_data = read_lut_data(args.lut_data)

    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, _ = dataset.get_data(
        dataset_format='dataframe', target=dataset.default_target_attribute
    )
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    thermometer = dwn.DistributiveThermometer(200).fit(x_train)
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)

    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    device = 'cuda:0'

    # Instantiate WNN and PyTorch model
    wnn = WeightlessNeuralNetwork(num_luts_inp, luts_data)
    lut_layer = dwn.LUTLayer(x_train.size(1), num_luts, n=num_luts_inp, mapping='learnable')
    group_sum = dwn.GroupSum(k=len(label_names), tau=1)
    model = nn.Sequential(lut_layer, group_sum).to(device)

    if args.model_checkpoint_path:
        model.load_state_dict(torch.load(args.model_checkpoint_path))
    model.eval()

    data_samples, targets = (x_train, y_train) if args.dataset == 'train' else (x_test, y_test)
    num_samples = min(args.num_of_samples, len(data_samples))

    # Lists to store outputs
    wnn_input_list = []
    wnn_output_list = []
    lut_output_list = []

    # Full output with all data
    with open('_sim_output_full.txt', 'w') as f:
        for sample_index in range(num_samples):
            sample_data = data_samples[sample_index:sample_index + 1]
            sample_output = targets[sample_index].item()

            sample_wnn_inp = convert_to_wnn_data(sample_data, input_mapping)
            wnn_input_list.append(sample_wnn_inp[::-1])

            # PyTorch model output
            model_out = model(sample_data.to(device))[0].detach().cpu().numpy()
            # WNN output
            wnn_out = wnn(sample_wnn_inp).numpy()

            # LUT layer intermediate output
            lut_out = lut_layer(sample_data.to(device))[0].detach().cpu().numpy()
            lut_output_list.append(lut_out.tolist())

            f.write(f"{sample_wnn_inp} | model={model_out} | wnn={wnn_out} | lut={lut_out} | expected={sample_output}\n")

            # Check consistency
            if not np.array_equal(model_out, wnn_out):
                print('Incorrect LUT/GroupSum mapping')
                raise ValueError(f"Mismatch at sample {sample_index}")

            acc_flag = 'Correct' if model_out.argmax() == sample_output else 'Incorrect'
            print(f"Sample {sample_index}: {acc_flag} | model={model_out.argmax()} wnn={wnn_out.argmax()} expected={sample_output}")

    # Write simple outputs
    with open('_sim_output.txt', 'w') as f:
        for sample_index in range(num_samples):
            sample_data = data_samples[sample_index:sample_index + 1]
            sample_output = targets[sample_index].item()
            model_out = model(sample_data.to(device))[0].detach().cpu().numpy()
            f.write(f"{model_out.argmax()}\n")

    # Write WNN inputs
    with open('_sim_wnn_input.txt', 'w') as f:
        for inp in wnn_input_list:
            f.write(f'"{inp}",\n')

    # Write WNN outputs
    with open('_sim_wnn_output.txt', 'w') as f:
        for out in wnn_output_list:
            f.write(f'"{"".join(str(int(x)) for x in out)}",\n')

    # Write LUT layer outputs
    with open('_sim_lut_layer_output.txt', 'w') as f:
        for lut_vec in lut_output_list:
            for i in range(len(lut_vec)):
                f.write(str(int(lut_vec[i])))
            f.write('\n')
                


if __name__ == '__main__':
    main()
