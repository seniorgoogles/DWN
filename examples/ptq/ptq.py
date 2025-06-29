import argparse
import os
import torch
import datetime
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

import torch_dwn as dwn
from torch import nn
from brevitas.nn import QuantIdentity
from brevitas.quant import Int8ActPerTensorFixedPointMSE

def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def evaluate_single_quantization(args, original_data, quant_bits, luts_num):
    """Evaluate model with a specific quantization bit width and LUT number"""
    QUANTIZER = Int8ActPerTensorFixedPointMSE
    device = 'cuda:0'
    
    x_train_orig, x_test_orig, y_train, y_test, num_output = original_data
    
    # Create quantized versions
    x_test_quant = x_test_orig.copy()
    x_train_quant = x_train_orig.copy()
    
    if quant_bits > 0:
        print(f"   • Applying quantization ({quant_bits} bits)...")
        quant_identity = QuantIdentity(return_quant_tensor=True, bit_width=quant_bits, act_quant=QUANTIZER)
        for i in range(x_test_orig.shape[1]):
            x_test_quant[:, i] = quant_identity(torch.tensor(x_test_orig[:, i])).tensor.detach().numpy()
            x_train_quant[:, i] = quant_identity(torch.tensor(x_train_orig[:, i])).tensor.detach().numpy()
    
    # Convert to tensors
    x_train_orig_tensor = torch.tensor(x_train_orig)
    x_test_orig_tensor = torch.tensor(x_test_orig)
    x_train_quant_tensor = torch.tensor(x_train_quant)
    x_test_quant_tensor = torch.tensor(x_test_quant)
    
    # Thermometer encoding - fit on ORIGINAL data
    if args.thermometer == "uniform_thermometer":
        thermometer = dwn.Thermometer(args.thermometer_bits).fit(x_train_orig_tensor)
    elif args.thermometer == "gaussian_thermometer":
        thermometer = dwn.GaussianThermometer(args.thermometer_bits).fit(x_train_orig_tensor)
    elif args.thermometer == "distributive_thermometer":
        thermometer = dwn.DistributiveThermometer(args.thermometer_bits).fit(x_train_orig_tensor)
    else:
        raise ValueError(f"Unknown thermometer type: {args.thermometer}")
    
    # Apply thermometer to both original and quantized data
    x_test_fp = thermometer.binarize(x_test_orig_tensor).flatten(start_dim=1)
    x_test_quant_binarized = thermometer.binarize(x_test_quant_tensor).flatten(start_dim=1)
    
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)
    
    # Load model (create new model for each combination)
    model_key = f"model_{luts_num}_{quant_bits}"
    if not hasattr(evaluate_single_quantization, 'models'):
        evaluate_single_quantization.models = {}
    
    if model_key not in evaluate_single_quantization.models:
        lut_layer = dwn.LUTLayer(x_test_fp.size(1), luts_num, n=args.luts_inp_num, mapping='learnable')
        group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)
        
        model = nn.Sequential(
            lut_layer,
            group_sum
        ).cuda(device)
        
        if args.weights is None:
            raise ValueError("No weights file provided. Use --weights <file> to load trained weights.")
        
        model.load_state_dict(torch.load(args.weights))
        evaluate_single_quantization.models[model_key] = model
        print(f"Loaded weights from {args.weights} for LUTs: {luts_num}")
    
    model = evaluate_single_quantization.models[model_key]
    
    # Evaluate on floating point (original) data
    fp_acc = evaluate(model, x_test_fp, y_test_tensor, device)
    
    # Evaluate on quantized data
    quant_acc = evaluate(model, x_test_quant_binarized, y_test_tensor, device)
    
    print(f"LUTs: {luts_num:3d} | Bits: {quant_bits:2d} | FP Accuracy: {fp_acc*100:5.2f}% | Quantized Accuracy: {quant_acc*100:5.2f}%")
    
    return fp_acc, quant_acc

def main(args):
    # Load dataset once
    print("Loading dataset...")
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)
    num_output = labels.max() + 1
    
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=42)
    
    if args.num_examples is not None:
        x_train = x_train[:args.num_examples]
        y_train = y_train[:args.num_examples]
    
    original_data = (x_train, x_test, y_train, y_test, num_output)
    
    # Define quantization bit range and LUT range
    bit_range = list(range(args.max_bits, args.min_bits - 1, -1))  # 32 down to 3
    lut_range = args.luts_num if isinstance(args.luts_num, list) else [args.luts_num]
    
    results = []
    
    print(f"\nRunning quantization sweep from {args.max_bits} to {args.min_bits} bits...")
    print(f"LUT configurations: {lut_range}")
    print("="*80)
    
    for luts_num in lut_range:
        print(f"\n--- Testing with {luts_num} LUTs ---")
        for quant_bits in bit_range:
            fp_acc, quant_acc = evaluate_single_quantization(args, original_data, quant_bits, luts_num)
            results.append({
                'luts_num': luts_num,
                'bits': quant_bits,
                'fp_accuracy': fp_acc * 100,
                'quant_accuracy': quant_acc * 100,
                'accuracy_drop': (fp_acc - quant_acc) * 100
            })
    
    # Save results to CSV with floating point accuracy included
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_folder, 'quantization_results.csv')
    
    # Add a separate row at the top with just the floating point baseline
    if len(results) > 0:
        # Get the floating point baseline (should be the same across all configurations)
        fp_baseline = results[0]['fp_accuracy']
        
        # Create header row with FP baseline
        header_row = pd.DataFrame([{
            'luts_num': 'FP_BASELINE',
            'bits': 'FP32', 
            'fp_accuracy': fp_baseline,
            'quant_accuracy': fp_baseline,
            'accuracy_drop': 0.0
        }])
        
        # Combine header row with results
        final_df = pd.concat([header_row, results_df], ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print(f"CSV includes FP32 baseline row (FP Accuracy: {fp_baseline:.2f}%)")
    else:
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    
    # Print summary statistics
    print_summary(results_df, lut_range)
    
    return results_df

def print_summary(results_df, lut_range):
    """Print comprehensive summary statistics"""
    multiple_luts = len(lut_range) > 1
    
    print("\n" + "="*80)
    print("QUANTIZATION ANALYSIS SUMMARY")
    print("="*80)
    
    if multiple_luts:
        for luts_num in lut_range:
            lut_data = results_df[results_df['luts_num'] == luts_num]
            print(f"\n--- {luts_num} LUTs ---")
            print(f"FP32 Baseline Accuracy: {lut_data['fp_accuracy'].iloc[0]:.2f}%")
            print(f"Best Quantized Accuracy: {lut_data['quant_accuracy'].max():.2f}% at {lut_data.loc[lut_data['quant_accuracy'].idxmax(), 'bits']} bits")
            print(f"Worst Quantized Accuracy: {lut_data['quant_accuracy'].min():.2f}% at {lut_data.loc[lut_data['quant_accuracy'].idxmin(), 'bits']} bits")
            print(f"Max Accuracy Drop: {lut_data['accuracy_drop'].max():.2f}% at {lut_data.loc[lut_data['accuracy_drop'].idxmax(), 'bits']} bits")
            
            # Find acceptable bit widths
            acceptable = lut_data[lut_data['accuracy_drop'] < 5.0]
            if not acceptable.empty:
                min_acceptable_bits = acceptable['bits'].min()
                print(f"Minimum bits for <5% accuracy drop: {min_acceptable_bits} bits")
            else:
                print("No quantization level achieves <5% accuracy drop")
        
        # Overall best configurations
        print(f"\n--- OVERALL BEST CONFIGURATIONS ---")
        best_overall = results_df.loc[results_df['quant_accuracy'].idxmax()]
        print(f"Best overall: {best_overall['quant_accuracy']:.2f}% with {int(best_overall['luts_num'])} LUTs at {int(best_overall['bits'])} bits")
        
        # Best efficiency (accuracy per complexity unit)
        results_df['complexity'] = results_df['luts_num'] * results_df['bits']
        results_df['efficiency'] = results_df['quant_accuracy'] / results_df['complexity']
        best_efficiency = results_df.loc[results_df['efficiency'].idxmax()]
        print(f"Most efficient: {best_efficiency['quant_accuracy']:.2f}% with {int(best_efficiency['luts_num'])} LUTs at {int(best_efficiency['bits'])} bits")
        
        # Print top 5 configurations by accuracy
        print(f"\n--- TOP 5 CONFIGURATIONS BY ACCURACY ---")
        top_5 = results_df.nlargest(5, 'quant_accuracy')
        for idx, row in top_5.iterrows():
            print(f"{row['quant_accuracy']:.2f}% | {int(row['luts_num'])} LUTs | {int(row['bits'])} bits | Drop: {row['accuracy_drop']:.2f}%")
        
        # Print top 5 most efficient configurations
        print(f"\n--- TOP 5 MOST EFFICIENT CONFIGURATIONS ---")
        top_5_eff = results_df.nlargest(5, 'efficiency')
        for idx, row in top_5_eff.iterrows():
            print(f"{row['quant_accuracy']:.2f}% | {int(row['luts_num'])} LUTs | {int(row['bits'])} bits | Efficiency: {row['efficiency']:.4f}")
        
    else:
        print(f"FP32 Baseline Accuracy: {results_df['fp_accuracy'].iloc[0]:.2f}%")
        print(f"Best Quantized Accuracy: {results_df['quant_accuracy'].max():.2f}% at {results_df.loc[results_df['quant_accuracy'].idxmax(), 'bits']} bits")
        print(f"Worst Quantized Accuracy: {results_df['quant_accuracy'].min():.2f}% at {results_df.loc[results_df['quant_accuracy'].idxmin(), 'bits']} bits")
        print(f"Max Accuracy Drop: {results_df['accuracy_drop'].max():.2f}% at {results_df.loc[results_df['accuracy_drop'].idxmax(), 'bits']} bits")
        
        # Find acceptable bit widths (e.g., <5% accuracy drop)
        acceptable = results_df[results_df['accuracy_drop'] < 5.0]
        if not acceptable.empty:
            min_acceptable_bits = acceptable['bits'].min()
            print(f"Minimum bits for <5% accuracy drop: {min_acceptable_bits} bits")
        else:
            print("No quantization level achieves <5% accuracy drop")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model across quantization bit range.')
    parser.add_argument('--output_folder', type=str, default='quantization_sweep', help='Folder to store outputs')
    parser.add_argument('--num-examples', type=int, default=None, help='Number of examples to use for fitting thermometer')
    parser.add_argument('--luts-num', type=str, default='50', help='Number of LUTs (single value or comma-separated list, e.g., "25,50,100")')
    parser.add_argument('--luts-inp-num', type=int, default=6, help='Number of inputs per LUT')
    parser.add_argument('--thermometer', '-t', type=str, default="uniform_thermometer", help='Thermometer type')
    parser.add_argument('--thermometer-bits', '-b', type=int, default=200, help='Number of thermometer bits')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights file')
    parser.add_argument('--min-bits', type=int, default=3, help='Minimum quantization bits')
    parser.add_argument('--max-bits', type=int, default=32, help='Maximum quantization bits')
    args = parser.parse_args()
    
    # Parse LUTs argument
    if ',' in args.luts_num:
        args.luts_num = [int(x.strip()) for x in args.luts_num.split(',')]
    else:
        args.luts_num = [int(args.luts_num)]
    
    # Clear output folder if exists
    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)
    
    results = main(args)