import argparse
import os
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
import datetime
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, device, skip_batches=500):
    test_acc = 0.0
    best_acc = 0.0
    
    n_samples = x_train.shape[0]
    print(f"{epochs=}, {batch_size=}, {n_samples=}")
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        
        for batch_index, i in enumerate(range(0, n_samples, batch_size), start=1):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)
            
            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            pred_train = outputs.argmax(dim=1)
            batch_correct = (pred_train == batch_y).sum().item()
            correct_train += batch_correct
            batch_total = batch_y.size(0)
            total_train += batch_total
            
            batch_acc = batch_correct / batch_total
            
            if batch_index % skip_batches == 0 and batch_index > 0:
                print(f'Epoch {epoch + 1}, Batch {batch_index}: Loss: {loss.item():.4f}, Accuracy: {batch_acc:.4f}')
        
        scheduler.step()
        
        train_acc = correct_train / total_train
        test_acc = evaluate(model, x_test, y_test, device)
        print(f'Epoch {epoch + 1} Summary: Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"Best Test Accuracy: {best_acc:.4f}")
            torch.save(model.state_dict(), os.path.join(args.output_folder, 'best.pth'))
    
    print("Best Test Accuracy: ", best_acc)
    print("Final Test Accuracy: ", test_acc)
    torch.save(model.state_dict(), os.path.join(args.output_folder, 'last.pth'))

    return best_acc

def main(args):
    # Set device und Hyperparameter
    device = 'cuda:0'
    epochs = args.epochs
    luts_num = args.luts_num
    luts_inp_num = args.luts_inp_num
    batch_size = args.batch_size
    
    thermometer = None
    thermometer_bits = args.thermometer_bits
    thermometer_type = args.thermometer

    # Erstelle Ausgabeverzeichnis falls nicht vorhanden
    os.makedirs(args.output_folder, exist_ok=True)

    # Dataset laden und aufteilen
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x : label_names.index(x)).values)
    num_output = labels.max() + 1

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=42)
    
    # Begrenze Anzahl der Trainingsbeispiele, falls angegeben
    if args.num_examples is not None:
        num = args.num_examples
        x_train = x_train[:num]
        y_train = y_train[:num]

    if thermometer_type == "distributive_thermometer":
        thermometer = dwn.DistributiveThermometer(thermometer_bits).fit(x_train)
    elif thermometer_type == "thermometer": 
        thermometer = dwn.Thermometer(thermometer_bits).fit(x_train)
    else:
        raise ValueError(f"Unknown thermometer type: {thermometer_type}")
    
    #print(f"Thermometer thresholds: {thermometer.thresholds=}")
    
    # Print colorful that the dataset is fitted to the thermometer
    print(f"\033[92mThermometer fitted to dataset with {x_train.shape[0]} samples and {x_train.shape[1]} features.\033[0m")

    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)
    
    #print(thermometer.get_thresholds(x_train))
    
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    lut_layer = dwn.LUTLayer(x_train.size(1), luts_num, n=luts_inp_num, mapping='learnable') 
    group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)

    model = nn.Sequential(
        lut_layer,
        group_sum
    ).cuda(device)

    # 1e-2(30), 1e-3(30), 1e-4(30), 1e-5(10)
    epochs = 14
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012123)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size, device=device)
    
    epochs = 14
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size, device=device)
    
    epochs = 4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
    train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size, device=device)
    
    # Modell speichern
    model_path = os.path.join(args.output_folder, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Aktuelles Arbeitsverzeichnis ausgeben
    print(os.getcwd())

    
    # Modell neu laden
    model = nn.Sequential(
        lut_layer,
        group_sum
    ).cuda(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Speichere Konfiguration
    config_path = os.path.join(args.output_folder, 'model_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"{luts_num=}\n")
        f.write(f"{luts_inp_num=}\n")
        f.write(f"{thermometer_type=}")

    # Speichere LUT-Daten als CSV
    luts_data_path = os.path.join(args.output_folder, 'luts_data.txt')
    with open(luts_data_path, 'w') as f:
        for lut_index in range(len(lut_layer.luts)):
            lut = lut_layer.luts[lut_index].cpu().detach().numpy()
            print(lut.shape)
            for index in range(lut.size):
                f.write(f"{1 if lut[index] > 0 else 0}")
                if index == lut.size - 1:
                    if lut_index != len(lut_layer.luts) - 1:
                        f.write("\n")

    # Speichere Mapping-Daten
    mapping_path = os.path.join(args.output_folder, 'mapping.txt')
    with open(mapping_path, 'w') as f:
        argmax_vals = lut_layer.mapping.weights.argmax(dim=0)
        for i, mapping in enumerate(argmax_vals):
            f.write(f"{mapping}")
            if i != len(argmax_vals) - 1:
                f.write(";")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate model with configurable output folder, example limit and LUT parameters.')
    parser.add_argument('--output_folder', type=str, default='output', help='Folder to store outputs')
    parser.add_argument('--num-examples', type=int, default=None, help='Number of examples to use for training')
    parser.add_argument('--luts-num', type=int, default=15, help='Number of LUTs')
    parser.add_argument('--luts-inp-num', type=int, default=6, help='Number of inputs per LUT')
    parser.add_argument('--thermometer', '-t', type=str, default="thermometer", help='Thermometer type, thermometer or distributive_thermometer')
    parser.add_argument('--thermometer_bits', '-b', type=int, default=200, help='Number of thermometer bits')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
    args = parser.parse_args()
    
    # Print command 
    str_cmd = f"python3 jsc.py --output_folder {args.output_folder} --num-examples {args.num_examples} --luts-num  {args.luts_num} --luts-inp-num {args.luts_inp_num} --epochs {args.epochs} --batch-size {args.batch_size}"
    print(str_cmd)
    
    # If outputfolder exists, ask for removement, if not, create it
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)

    os.makedirs(args.output_folder) 
        
    # Create subfolder for this run with the datetime format YYYYMMDD_HHMMSS_run    
    #now = datetime.datetime.now()
    #run_folder = now.strftime("%Y%m%d_%H%M%S_run")
    #args.output_folder = os.path.join(args.output_folder, run_folder)
    #os.makedirs(args.output_folder)
    
    # Write call to output folder als txt file
    with open(os.path.join(args.output_folder, 'call.txt'), 'w') as f:
        f.write(str_cmd)
    
    main(args)