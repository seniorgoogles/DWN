import argparse
import os
import sys
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
from brevitas.nn import QuantIdentity
from brevitas.quant import Int8ActPerTensorFixedPoint
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import json


def evaluate(model, x_test, y_test, device):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, device, experiment_id=None, skip_batches=1000):
    n_samples = x_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"📊 Training Configuration:")
    print(f"   • Training samples: {n_samples:,}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Batches per epoch: {n_batches}")
    print(f"   • Total epochs: {epochs}")
    print(f"   • Device: {device}")
    print(f"   • Optimizer: {type(optimizer).__name__}")
    print(f"   • Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    epoch_start_time = datetime.datetime.now()
    
    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{epochs} - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print("─" * 60)
        
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        epoch_loss = 0.0
        batch_times = []
        
        epoch_start = datetime.datetime.now()
        
        for batch_index, i in enumerate(range(0, n_samples, batch_size), start=1):
            batch_start = datetime.datetime.now()
            
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
            epoch_loss += loss.item()
            
            batch_time = (datetime.datetime.now() - batch_start).total_seconds()
            batch_times.append(batch_time)
            
            # Show progress for every batch
            avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)  # Average of last 10 batches
            samples_per_sec = batch_size / avg_batch_time
            remaining_batches = n_batches - batch_index
            eta_seconds = remaining_batches * avg_batch_time
            eta_time = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
            
            progress = batch_index / n_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            if batch_index % skip_batches == 0 or batch_index == n_batches:
                print(f"   Batch {batch_index:4d}/{n_batches} [{bar}] {progress*100:5.1f}% | "
                    f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f} | "
                    f"{samples_per_sec:.0f} samples/s | ETA: {eta_time.strftime('%H:%M:%S')}")
        
        # End of epoch
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()
        train_acc = correct_train / total_train
        avg_loss = epoch_loss / n_batches
        test_acc = evaluate(model, x_test, y_test, device)
        
        print(f"\n✅ Epoch {epoch + 1} Complete:")
        print(f"   • Duration: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        print(f"   • Avg batch time: {sum(batch_times)/len(batch_times):.3f}s")
        print(f"   • Samples/second: {n_samples/epoch_time:.0f}")
        print(f"   • Average loss: {avg_loss:.4f}")
        print(f"   • Train accuracy: {train_acc:.4f} ({correct_train}/{total_train})")
        print(f"   • Test accuracy: {test_acc:.4f}")
        print(f"   • Learning rate: {old_lr:.2e} → {new_lr:.2e}")
        
        # Memory usage if CUDA
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"   • GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    
    # Return final test accuracy after all epochs
    final_test_acc = evaluate(model, x_test, y_test, device)
    
    total_training_time = (datetime.datetime.now() - epoch_start_time).total_seconds()
    print(f"\n🎯 Training Phase Complete:")
    print(f"   • Total time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
    print(f"   • Final test accuracy: {final_test_acc:.4f}")
    print(f"   • Avg time per epoch: {total_training_time/epochs:.1f}s")
    
    return final_test_acc

def prepare_dataset(quant_bits, thermometer_bits, thermometer_type, num_examples=None, random_state=42):
    """Prepare dataset with fresh initialization for each experiment"""
    print(f"\n📁 Dataset Preparation:")
    print(f"   • Loading OpenML dataset 42468...")
    
    # Dataset laden und aufteilen
    dataset_start = datetime.datetime.now()
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
    
    print(f"   • Dataset loaded in {(datetime.datetime.now() - dataset_start).total_seconds():.2f}s")
    print(f"   • Features shape: {df_features.shape}")
    print(f"   • Feature names: {list(df_features.columns)[:5]}{'...' if len(df_features.columns) > 5 else ''}")
    print(f"   • Target: {dataset.default_target_attribute}")
    
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x : label_names.index(x)).values)
    num_output = labels.max() + 1

    print(f"   • Unique classes: {len(label_names)} {label_names}")
    print(f"   • Class distribution: {dict(zip(label_names, np.bincount(labels)))}")
    print(f"   • Features dtype: {features.dtype}, shape: {features.shape}")
    print(f"   • Labels dtype: {labels.dtype}, shape: {labels.shape}")

    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=random_state)
    
    print(f"   • Train/test split (80/20): {x_train.shape[0]:,} / {x_test.shape[0]:,} samples")
    
    # Begrenze Anzahl der Trainingsbeispiele, falls angegeben
    original_train_size = x_train.shape[0]
    if num_examples is not None:
        x_train = x_train[:num_examples]
        y_train = y_train[:num_examples]
        print(f"   • Limited training samples: {original_train_size:,} → {x_train.shape[0]:,}")
    
    # Quantize dataset if quantization is enabled
    if quant_bits > 0:
        print(f"   • Applying quantization ({quant_bits} bits)...")
        quant_start = datetime.datetime.now()
        quant_identity = QuantIdentity(return_quant_tensor=True, bit_width=quant_bits, act_quant=Int8ActPerTensorFixedPoint)
        x_test = quant_identity(torch.tensor(x_test)).tensor
        x_train = quant_identity(torch.tensor(x_train)).tensor
        quant_time = (datetime.datetime.now() - quant_start).total_seconds()
        print(f"   • Quantization completed in {quant_time:.2f}s")
    else:
        x_train = torch.tensor(x_train)
        x_test = torch.tensor(x_test)
        print(f"   • No quantization applied")

    # Thermometer encoding
    print(f"   • Applying {thermometer_type} thermometer encoding ({thermometer_bits} bits)...")
    
    therm_start = datetime.datetime.now()
    if thermometer_type == 'distributive':
        thermometer = dwn.DistributiveThermometer(thermometer_bits).fit(x_train)
    else:
        thermometer = dwn.Thermometer(thermometer_bits).fit(x_train)

    # Convert dataset to thermometer encoding
    x_train_orig_shape = x_train.shape
    x_test_orig_shape = x_test.shape
    
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)
        
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)
    
    therm_time = (datetime.datetime.now() - therm_start).total_seconds()
    print(f"   • Thermometer encoding completed in {therm_time:.2f}s")
    print(f"   • Shape transformation: {x_train_orig_shape} → {x_train.shape}")
    print(f"   • Test shape transformation: {x_test_orig_shape} → {x_test.shape}")
    print(f"   • Binary features per sample: {x_train.shape[1]:,}")
    print(f"   • Memory usage - Train: {x_train.numel() * 4 / 1024**2:.1f} MB")
    print(f"   • Memory usage - Test: {x_test.numel() * 4 / 1024**2:.1f} MB")
    
    return x_train, x_test, y_train, y_test, num_output

def run_single_experiment(config, device):
    """Run a single experiment with given configuration"""
    experiment_start = datetime.datetime.now()
    
    print(f"\n🧪 Experiment {config['experiment_id']} Configuration:")
    print("─" * 50)
    for key, value in config.items():
        if key != 'experiment_id':
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    print("─" * 50)
    
    # Prepare fresh dataset for this experiment
    x_train, x_test, y_train, y_test, num_output = prepare_dataset(
        config['quant_bits'], 
        config['thermometer_bits'], 
        config['thermometer_type'],
        config.get('num_examples', None)
    )
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   • Input features: {x_train.size(1):,}")
    print(f"   • Number of LUTs: {config['luts_num']}")
    print(f"   • LUT input size: {config['luts_inp_num']}")
    print(f"   • Output classes: {num_output}")
    print(f"   • Group sum tau: {1/0.3:.3f}")
    
    # Setup model
    model_start = datetime.datetime.now()
    lut_layer = dwn.LUTLayer(x_train.size(1), config['luts_num'], n=config['luts_inp_num'], mapping='learnable') 
    group_sum = dwn.GroupSum(k=num_output, tau=1/0.3)

    model = nn.Sequential(
        lut_layer,
        group_sum
    ).cuda(device)
    
    model_setup_time = (datetime.datetime.now() - model_start).total_seconds()
    print(f"   • Model created in {model_setup_time:.3f}s")
    
    # Detailed model information
    print(f"\n🔧 Model Details:")
    print(f"   • LUT Layer:")
    print(f"     - Total LUTs: {len(lut_layer.luts)}")
    print(f"     - LUT shape: {lut_layer.luts[0].shape if len(lut_layer.luts) > 0 else 'N/A'}")
    print(f"     - Mapping type: learnable")
    print(f"     - Mapping shape: {lut_layer.mapping.weights.shape}")
    print(f"   • Group Sum Layer:")
    print(f"     - Output groups: {num_output}")
    print(f"     - Temperature (tau): {group_sum.tau}")
    
    # Memory estimation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # float32
    print(f"   • Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"   • Model size: {model_size_mb:.2f} MB")
    
    # Variables to track best model and final model
    best_accuracy = 0.0
    best_model_state = None
    final_model_state = None
    best_phase_info = ""
    final_phase_info = ""
    
    if config['epochs'] == -1:
        lr_list = [1e-2, 1e-3, 1e-4]
        epochs_list = [14, 14, 4]

        print(f"\n🎯 Multi-Phase Training Schedule:")
        for i, (lr, epochs) in enumerate(zip(lr_list, epochs_list), 1):
            print(f"   Phase {i}: {epochs} epochs @ lr={lr:.1e}")
        print("─" * 60)

        for phase, (lr, epochs) in enumerate(zip(lr_list, epochs_list), 1):
            phase_start = datetime.datetime.now()
            
            print(f"\n🚀 Training Phase {phase}/3")
            print(f"   • Learning Rate: {lr:.1e}")
            print(f"   • Epochs: {epochs}")
            print(f"   • Started: {phase_start.strftime('%H:%M:%S')}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
            
            phase_accuracy = train_and_evaluate(
                model, optimizer, scheduler, x_train, y_train, x_test, y_test, 
                epochs=epochs, batch_size=config['batch_size'], device=device, 
                experiment_id=config['experiment_id']
            )
            
            phase_time = (datetime.datetime.now() - phase_start).total_seconds()
            
            print(f"\n📈 Phase {phase} Results:")
            print(f"   • Duration: {phase_time:.1f}s ({phase_time/60:.1f}m)")
            print(f"   • Final Test Accuracy: {phase_accuracy:.4f}")
            
            # Check if this is the best accuracy so far
            if phase_accuracy > best_accuracy:
                best_accuracy = phase_accuracy
                best_model_state = copy.deepcopy(model.state_dict())
                best_phase_info = f"Phase {phase} (lr={lr:.1e}, epochs={epochs})"
                print(f"   🏆 NEW BEST ACCURACY: {best_accuracy:.4f}")
            else:
                print(f"   📊 Current best remains: {best_accuracy:.4f}")
            
            # Always save the state from the final phase
            if phase == len(lr_list):
                final_model_state = copy.deepcopy(model.state_dict())
                final_phase_info = f"Final Phase {phase} (lr={lr:.1e}, epochs={epochs})"
            
            if phase < len(lr_list):
                print(f"\n⏭️  Proceeding to Phase {phase + 1}...")
    else:    
        print(f"\n🚀 Single Phase Training:")
        print(f"   • Epochs: {config['epochs']}")
        print(f"   • Learning Rate: 1e-2")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)
        
        final_accuracy = train_and_evaluate(
            model, optimizer, scheduler, x_train, y_train, x_test, y_test, 
            epochs=config['epochs'], batch_size=config['batch_size'], device=device, 
            experiment_id=config['experiment_id']
        )
        
        # For single phase, best and final are the same
        best_accuracy = final_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        final_model_state = copy.deepcopy(model.state_dict())
        best_phase_info = f"Single phase ({config['epochs']} epochs)"
        final_phase_info = f"Single phase ({config['epochs']} epochs)"

    experiment_time = (datetime.datetime.now() - experiment_start).total_seconds()
    
    print(f"\n🎉 Experiment {config['experiment_id']} Complete!")
    print(f"   • Total Duration: {experiment_time:.1f}s ({experiment_time/60:.1f}m)")
    print(f"   • Best Accuracy: {best_accuracy:.4f}")
    print(f"   • Best Model from: {best_phase_info}")
    print(f"   • Final Model from: {final_phase_info}")
    print(f"   • Samples processed: {x_train.shape[0] * (32 if config['epochs'] == -1 else config['epochs']):,}")
    
    # Performance metrics
    if experiment_time > 0:
        throughput = x_train.shape[0] * (32 if config['epochs'] == -1 else config['epochs']) / experiment_time
        print(f"   • Training throughput: {throughput:.0f} samples/second")

    # Add experiment results to config
    config['best_accuracy'] = best_accuracy
    config['best_phase_info'] = best_phase_info
    config['final_phase_info'] = final_phase_info
    config['experiment_duration'] = experiment_time
    config['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return config, best_model_state, final_model_state

def generate_experiment_configs(args):
    """Generate all experiment configurations based on provided parameters"""
    
    # Define parameter grids for experiments
    if args.experiment_mode == 'grid':
        # Grid search over multiple parameters
        luts_nums = args.luts_num_list if args.luts_num_list else [args.luts_num]
        luts_inp_nums = args.luts_inp_num_list if args.luts_inp_num_list else [args.luts_inp_num]
        thermometer_bits_list = args.thermometer_bits_list if args.thermometer_bits_list else [args.thermometer_bits]
        quant_bits_list = args.quant_bits_list if args.quant_bits_list else [args.quant_bits]
        
        configs = []
        for luts_num, luts_inp_num, therm_bits, quant_bits in product(
            luts_nums, luts_inp_nums, thermometer_bits_list, quant_bits_list
        ):
            config = {
                'luts_num': luts_num,
                'luts_inp_num': luts_inp_num,
                'thermometer_bits': therm_bits,
                'thermometer_type': args.thermometer_type,
                'quant_bits': quant_bits,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'num_examples': args.num_examples,
                'experiment_id': len(configs) + 1
            }
            configs.append(config)
            
    else:
        # Single experiment
        configs = [{
            'luts_num': args.luts_num,
            'luts_inp_num': args.luts_inp_num,
            'thermometer_bits': args.thermometer_bits,
            'thermometer_type': args.thermometer_type,
            'quant_bits': args.quant_bits,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_examples': args.num_examples,
            'experiment_id': 1
        }]
    
    return configs

def save_experiment_results(configs, base_folder):
    """Save experiment results to CSV and create visualizations"""
    
    print(f"💾 Saving experiment results...")
    
    # Create results DataFrame
    results_df = pd.DataFrame(configs)
    
    # Save to CSV
    csv_path = os.path.join(base_folder, 'experiment_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"   • CSV saved: {csv_path}")
    
    # Create visualizations
    plt.style.use('default')
    plots_created = []
    
    print(f"🎨 Creating visualizations...")
    
    # 1. Bar plot of all experiments
    plt.figure(figsize=(12, 6))
    bars = plt.bar(results_df['experiment_id'], results_df['best_accuracy'])
    
    # Color bars based on performance
    accuracies = results_df['best_accuracy']
    max_acc = accuracies.max()
    min_acc = accuracies.min()
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if acc == max_acc:
            bar.set_color('gold')
        elif acc >= max_acc - 0.01:  # Top performers
            bar.set_color('lightgreen')
        elif acc <= min_acc + 0.01:  # Bottom performers
            bar.set_color('lightcoral')
        else:
            bar.set_color('lightblue')
    
    plt.xlabel('Experiment ID')
    plt.ylabel('Best Accuracy')
    plt.title(f'Best Accuracy per Experiment (Range: {min_acc:.3f} - {max_acc:.3f})')
    plt.xticks(results_df['experiment_id'])
    
    # Add value labels on bars
    for i, v in enumerate(results_df['best_accuracy']):
        plt.text(i+1, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(base_folder, 'accuracy_by_experiment.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append('accuracy_by_experiment.png')
    print(f"   • Bar plot: {plot_path}")
    
    # 2. If multiple values for luts_num, create line plot
    if len(results_df['luts_num'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        for luts_inp_num in sorted(results_df['luts_inp_num'].unique()):
            subset = results_df[results_df['luts_inp_num'] == luts_inp_num]
            subset = subset.sort_values('luts_num')
            plt.plot(subset['luts_num'], subset['best_accuracy'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f'LUT inputs: {luts_inp_num}')
        plt.xlabel('Number of LUTs')
        plt.ylabel('Best Accuracy')
        plt.title('Accuracy vs Number of LUTs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(base_folder, 'accuracy_vs_luts_num.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append('accuracy_vs_luts_num.png')
        print(f"   • Line plot (LUTs): {plot_path}")
    
    # 3. If multiple values for thermometer_bits, create line plot
    if len(results_df['thermometer_bits'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        for luts_num in sorted(results_df['luts_num'].unique()):
            subset = results_df[results_df['luts_num'] == luts_num]
            subset = subset.sort_values('thermometer_bits')
            plt.plot(subset['thermometer_bits'], subset['best_accuracy'], 
                    marker='s', linewidth=2, markersize=8,
                    label=f'LUTs: {luts_num}')
        plt.xlabel('Thermometer Bits')
        plt.ylabel('Best Accuracy')
        plt.title('Accuracy vs Thermometer Bits')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(base_folder, 'accuracy_vs_thermometer_bits.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append('accuracy_vs_thermometer_bits.png')
        print(f"   • Line plot (Thermometer): {plot_path}")
    
    # 4. Heatmap if we have a 2D grid
    if len(results_df['luts_num'].unique()) > 1 and len(results_df['luts_inp_num'].unique()) > 1:
        pivot_table = results_df.pivot_table(values='best_accuracy', 
                                           index='luts_inp_num', 
                                           columns='luts_num', 
                                           aggfunc='mean')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', 
                   cbar_kws={'label': 'Best Accuracy'})
        plt.title('Accuracy Heatmap: LUT Inputs vs Number of LUTs')
        plt.ylabel('Number of LUT Inputs')
        plt.xlabel('Number of LUTs')
        plt.tight_layout()
        plot_path = os.path.join(base_folder, 'accuracy_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append('accuracy_heatmap.png')
        print(f"   • Heatmap: {plot_path}")
    
    # 5. Distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['best_accuracy'], bins=min(20, len(results_df)), 
             alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(results_df['best_accuracy'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["best_accuracy"].mean():.3f}')
    plt.axvline(results_df['best_accuracy'].median(), color='green', linestyle='--', 
                label=f'Median: {results_df["best_accuracy"].median():.3f}')
    plt.xlabel('Best Accuracy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(base_folder, 'accuracy_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plots_created.append('accuracy_distribution.png')
    print(f"   • Distribution plot: {plot_path}")
    
    # 6. If experiment duration is available, create time vs accuracy plot
    if 'experiment_duration' in results_df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['experiment_duration'] / 60, results_df['best_accuracy'], 
                   alpha=0.7, s=60, c=results_df['best_accuracy'], cmap='viridis')
        plt.colorbar(label='Best Accuracy')
        plt.xlabel('Experiment Duration (minutes)')
        plt.ylabel('Best Accuracy')
        plt.title('Accuracy vs Experiment Duration')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(base_folder, 'accuracy_vs_duration.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots_created.append('accuracy_vs_duration.png')
        print(f"   • Duration plot: {plot_path}")
    
    # 7. Summary statistics
    summary_stats = {
        'total_experiments': len(results_df),
        'best_accuracy': float(results_df['best_accuracy'].max()),
        'worst_accuracy': float(results_df['best_accuracy'].min()),
        'mean_accuracy': float(results_df['best_accuracy'].mean()),
        'median_accuracy': float(results_df['best_accuracy'].median()),
        'std_accuracy': float(results_df['best_accuracy'].std()),
        'best_config': results_df.loc[results_df['best_accuracy'].idxmax()].to_dict(),
        'worst_config': results_df.loc[results_df['best_accuracy'].idxmin()].to_dict(),
        'plots_created': plots_created
    }
    
    # Save summary
    summary_path = os.path.join(base_folder, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=4, default=str)
    print(f"   • Summary: {summary_path}")
    
    print(f"\n📈 Experiment Analysis:")
    print(f"   • Total experiments: {summary_stats['total_experiments']}")
    print(f"   • Best accuracy: {summary_stats['best_accuracy']:.4f}")
    print(f"   • Worst accuracy: {summary_stats['worst_accuracy']:.4f}")
    print(f"   • Mean accuracy: {summary_stats['mean_accuracy']:.4f} ± {summary_stats['std_accuracy']:.4f}")
    print(f"   • Median accuracy: {summary_stats['median_accuracy']:.4f}")
    print(f"   • Accuracy range: {summary_stats['best_accuracy'] - summary_stats['worst_accuracy']:.4f}")
    print(f"   • Plots created: {len(plots_created)}")
    
    # Best configuration details
    best_config = summary_stats['best_config']
    print(f"\n🏆 Best Configuration:")
    for key, value in best_config.items():
        if key not in ['experiment_id', 'timestamp', 'best_phase_info']:
            print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    return results_df, summary_stats

def main(args):
    device = 'cuda:0'
    
    # System information
    print(f"🖥️  System Information:")
    print(f"   • Python version: {sys.version.split()[0]}")
    print(f"   • PyTorch version: {torch.__version__}")
    print(f"   • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   • CUDA version: {torch.version.cuda}")
        print(f"   • GPU device: {torch.cuda.get_device_name(device)}")
        print(f"   • GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    print(f"   • Process ID: {os.getpid()}")
    print(f"   • Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create base output folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_folder = f"{args.output_folder}_{timestamp}"
    os.makedirs(base_folder, exist_ok=True)
    
    # Generate experiment configurations
    configs = generate_experiment_configs(args)
    
    print(f"\n🔬 Experiment Setup:")
    print(f"   • Total experiments: {len(configs)}")
    print(f"   • Experiment mode: {args.experiment_mode}")
    print(f"   • Results folder: {base_folder}")
    
    if len(configs) > 1:
        print(f"\n📋 Parameter Variations:")
        param_variations = {}
        for config in configs:
            for key, value in config.items():
                if key not in ['experiment_id', 'timestamp']:
                    if key not in param_variations:
                        param_variations[key] = set()
                    param_variations[key].add(str(value))
        
        for param, values in param_variations.items():
            if len(values) > 1:
                print(f"   • {param.replace('_', ' ').title()}: {', '.join(sorted(values))}")
            else:
                print(f"   • {param.replace('_', ' ').title()}: {list(values)[0]} (fixed)")
    
    # Estimate total training time
    if args.epochs == -1:
        estimated_epochs_per_exp = 32  # 14 + 14 + 4
    else:
        estimated_epochs_per_exp = args.epochs
    
    # Rough estimate: assume 1-3 seconds per epoch depending on dataset size
    estimated_time_per_exp = estimated_epochs_per_exp * 2  # seconds
    total_estimated_time = estimated_time_per_exp * len(configs)
    
    if total_estimated_time > 60:
        time_str = f"{total_estimated_time/60:.1f} minutes"
        if total_estimated_time > 3600:
            time_str = f"{total_estimated_time/3600:.1f} hours"
    else:
        time_str = f"{total_estimated_time:.0f} seconds"
    
    print(f"   • Estimated duration: ~{time_str}")
    print(f"   • Expected completion: {(datetime.datetime.now() + datetime.timedelta(seconds=total_estimated_time)).strftime('%H:%M:%S')}")
    
    # Create progress tracking file
    progress_file = os.path.join(base_folder, 'progress.txt')
    status_file = os.path.join(base_folder, 'status.json')
    
    # Initialize status
    status = {
        'total_experiments': len(configs),
        'completed_experiments': 0,
        'current_experiment': 0,
        'start_time': datetime.datetime.now().isoformat(),
        'estimated_completion': None,
        'status': 'running',
        'best_accuracy_so_far': 0.0,
        'pid': os.getpid(),
        'args': vars(args)
    }
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    # Run all experiments
    all_results = []
    start_time = datetime.datetime.now()
    
    for i, config in enumerate(configs, 1):
        # Update status
        status['current_experiment'] = i
        status['completed_experiments'] = i - 1
        
        # Estimate completion time
        if i > 1:
            elapsed = datetime.datetime.now() - start_time
            avg_time_per_exp = elapsed / (i - 1)
            remaining_time = avg_time_per_exp * (len(configs) - i + 1)
            estimated_completion = datetime.datetime.now() + remaining_time
            status['estimated_completion'] = estimated_completion.isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"🧪 EXPERIMENT {i}/{len(configs)} - {datetime.datetime.now().strftime('%H:%M:%S')}")
        if status['estimated_completion']:
            print(f"⏰ Estimated completion: {estimated_completion.strftime('%H:%M:%S')} ({str(remaining_time).split('.')[0]} remaining)")
        if status['best_accuracy_so_far'] > 0:
            print(f"🏆 Best accuracy so far: {status['best_accuracy_so_far']:.4f}")
        print(f"{'='*80}")
        
        # Update progress file
        with open(progress_file, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting experiment {i}/{len(configs)}\n")
        
        # Create individual experiment folder
        exp_folder = os.path.join(base_folder, f"experiment_{config['experiment_id']}")
        os.makedirs(exp_folder, exist_ok=True)
        
        # Run experiment
        result_config, best_model_state, final_model_state = run_single_experiment(config, device)
        
        # Save both models
        best_model_path = os.path.join(exp_folder, 'best_model.pth')
        final_model_path = os.path.join(exp_folder, 'final_model.pth')
        torch.save(best_model_state, best_model_path)
        torch.save(final_model_state, final_model_path)
        
        config_path = os.path.join(exp_folder, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(result_config, f, indent=4, default=str)
        
        all_results.append(result_config)
        
        # Update best accuracy
        if result_config['best_accuracy'] > status['best_accuracy_so_far']:
            status['best_accuracy_so_far'] = result_config['best_accuracy']
        
        # Show progress summary
        accuracies = [r['best_accuracy'] for r in all_results]
        avg_acc = sum(accuracies) / len(accuracies)
        
        progress_msg = f"✅ Experiment {i} completed - Accuracy: {result_config['best_accuracy']:.4f}"
        print(f"\n{progress_msg}")
        print(f"📊 Progress summary:")
        print(f"   • Best so far: {status['best_accuracy_so_far']:.4f}")
        print(f"   • Average: {avg_acc:.4f}")
        print(f"   • Range: {min(accuracies):.4f} - {max(accuracies):.4f}")
        print(f"   • Completed: {len(all_results)}/{len(configs)}")
        
        with open(progress_file, 'a') as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {progress_msg}\n")
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Best so far: {status['best_accuracy_so_far']:.4f}, Average: {avg_acc:.4f}\n")
    
    # Final status update
    total_duration = datetime.datetime.now() - start_time
    status['completed_experiments'] = len(configs)
    status['current_experiment'] = len(configs)
    status['status'] = 'completed'
    status['end_time'] = datetime.datetime.now().isoformat()
    status['total_duration'] = str(total_duration)
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)
    
    # Save aggregated results and create visualizations
    print(f"\n{'='*80}")
    print(f"📊 GENERATING RESULTS AND VISUALIZATIONS")
    print(f"{'='*80}")
    
    results_df, summary_stats = save_experiment_results(all_results, base_folder)
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"   • Total duration: {str(total_duration).split('.')[0]}")
    print(f"   • Results saved to: {base_folder}")
    print(f"   • Experiments completed: {len(all_results)}")
    print(f"   • Best accuracy achieved: {summary_stats['best_accuracy']:.4f}")
    print(f"   • Average accuracy: {summary_stats['mean_accuracy']:.4f} ± {summary_stats['std_accuracy']:.4f}")
    
    with open(progress_file, 'a') as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All experiments completed! Best: {summary_stats['best_accuracy']:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-experiment training with result analysis')
    
    # Basic parameters
    parser.add_argument('--output_folder', type=str, default='experiments', help='Base folder for experiment outputs')
    parser.add_argument('--experiment-mode', type=str, default='single', choices=['single', 'grid'], 
                       help='Single experiment or grid search')
    
    # Dataset parameters
    parser.add_argument('--num-examples', type=int, default=None, help='Number of examples to use for training')
    
    # Model parameters (single values)
    parser.add_argument('--luts-num', type=int, default=50, help='Number of LUTs (default for single experiment)')
    parser.add_argument('--luts-inp-num', type=int, default=6, help='Number of inputs per LUT (default for single experiment)')
    parser.add_argument('--thermometer_bits', '-b', type=int, default=200, help='Number of thermometer bits (default for single experiment)')
    parser.add_argument('--quant-bits', type=int, default=-1, help='Number of bits for quantization (default for single experiment)')
    
    # Model parameters (lists for grid search)
    parser.add_argument('--luts-num-list', type=int, nargs='+', default=None, 
                       help='List of LUT numbers for grid search (e.g., --luts-num-list 25 50 100)')
    parser.add_argument('--luts-inp-num-list', type=int, nargs='+', default=None,
                       help='List of LUT input numbers for grid search (e.g., --luts-inp-num-list 4 6 8)')
    parser.add_argument('--thermometer-bits-list', type=int, nargs='+', default=None,
                       help='List of thermometer bits for grid search (e.g., --thermometer-bits-list 100 200 400)')
    parser.add_argument('--quant-bits-list', type=int, nargs='+', default=None,
                       help='List of quantization bits for grid search (e.g., --quant-bits-list -1 8 16)')
    
    # Training parameters
    parser.add_argument('--thermometer_type', type=str, default='distributive', choices=['distributive', 'standard'], 
                       help='Type of thermometer encoding')
    parser.add_argument('--epochs', type=int, default=-1, help='Number of epochs for training, for -1 use default epochs for each learning rate')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Print command 
    str_cmd = f"python {__file__} " + ' '.join(f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items() if v is not None)
    print(f"Command: {str_cmd}")
    
    main(args)