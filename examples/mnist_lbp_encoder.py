"""
MNIST Digit Classification using DWN (Deep Weight-sharing Network)
This example demonstrates using EncoderLayer with thermometer encoding for MNIST classification.
"""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
from torch_dwn import EncoderLayer
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets, transforms


# ============================================================================
# Utility Functions
# ============================================================================

def plot_dataset(dataset, filename="dataset.png"):
    """Plot histograms of first 16 features in a 4x4 grid"""
    num_features = dataset.shape[1]
    fig, axes = plt.subplots(4, 4, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(num_features, 16)):
        axes[i].hist(dataset[:, i].cpu().numpy(), bins=30, edgecolor='black')
        axes[i].set_title(f'Feature {i}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_thresholds_comparison(initial_thresholds, final_thresholds, train_data, features_to_plot=range(16),
                                input_min=-1.0, input_max=0.999999999999999):
    """Plot thresholds and spacings before/after training for specified features"""
    initial_thresholds = initial_thresholds.cpu().numpy()
    final_thresholds = final_thresholds.cpu().numpy()
    train_data_np = train_data.cpu().numpy()

    for feature_idx in features_to_plot:
        if feature_idx >= initial_thresholds.shape[0]:
            break

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])  # Top row - full width for histogram
        ax2 = fig.add_subplot(gs[1, 0])  # Middle left - thresholds
        ax3 = fig.add_subplot(gs[1, 1])  # Middle right - spacings
        ax4 = fig.add_subplot(gs[2, 0])  # Bottom left - initial thresholds on histogram
        ax5 = fig.add_subplot(gs[2, 1])  # Bottom right - final thresholds on histogram

        fig.suptitle(f'Feature {feature_idx}: Data Distribution and Thresholds Analysis', fontsize=14, fontweight='bold')

        # Top: Data histogram
        feature_data = train_data_np[:, feature_idx]
        ax1.hist(feature_data, bins=100, color='lightblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Training Data Distribution')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Middle left: Combined thresholds plot
        ax2.plot(initial_thresholds[feature_idx], '-', label='Initial', color='blue', linewidth=2)
        ax2.plot(final_thresholds[feature_idx], '-', label='Final', color='green', linewidth=2, alpha=0.7)
        ax2.set_title('Thresholds: Initial vs Final')
        ax2.set_xlabel('Threshold Index')
        ax2.set_ylabel('Threshold Value')
        ax2.axhline(y=input_min, color='r', linestyle='--', alpha=0.3, label=f'Range: [{input_min:.1f}, {input_max:.2f}]')
        ax2.axhline(y=input_max, color='r', linestyle='--', alpha=0.3)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Middle right: Bar plot of spacing changes (final - initial)
        initial_with_bounds = np.concatenate([[input_min], initial_thresholds[feature_idx], [input_max]])
        initial_spacings = np.diff(initial_with_bounds)
        final_with_bounds = np.concatenate([[input_min], final_thresholds[feature_idx], [input_max]])
        final_spacings = np.diff(final_with_bounds)

        spacing_changes = final_spacings - initial_spacings
        x = np.arange(len(spacing_changes))

        # Color bars: red for negative (spacing decreased), green for positive (spacing increased)
        colors = ['red' if change < 0 else 'green' for change in spacing_changes]
        ax3.bar(x, spacing_changes, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_title('Spacing Changes (Final - Initial)\nIf thresholds unchanged, all bars = 0')
        ax3.set_xlabel('Spacing Index')
        ax3.set_ylabel('Spacing Change')
        ax3.grid(True, alpha=0.3, axis='y')

        # Bottom left: Histogram with initial thresholds
        ax4.hist(feature_data, bins=100, color='lightblue', edgecolor='black', alpha=0.5, label='Data')
        for i, thresh in enumerate(initial_thresholds[feature_idx]):
            ax4.axvline(x=thresh, color='blue', alpha=0.3, linewidth=0.5)
        ax4.set_title('Initial Thresholds on Data Distribution')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)

        # Bottom right: Histogram with final thresholds
        ax5.hist(feature_data, bins=100, color='lightgreen', edgecolor='black', alpha=0.5, label='Data')
        for i, thresh in enumerate(final_thresholds[feature_idx]):
            ax5.axvline(x=thresh, color='green', alpha=0.3, linewidth=0.5)
        ax5.set_title('Final Thresholds on Data Distribution')
        ax5.set_xlabel('Value')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)

        plt.savefig(f'mnist_feature{feature_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved plot: mnist_feature{feature_idx}.png')


# ============================================================================
# Model Functions
# ============================================================================

def evaluate(model, x_test, y_test, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc


def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, device):
    """Train model and evaluate after each epoch"""
    n_samples = x_train.shape[0]
    best_test_acc = 0.0
    best_epoch = 0

    print("\n" + "="*70)
    print(f"Training for {epochs} epochs (batch_size={batch_size})")
    print("="*70)

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

        train_acc = correct_train / total_train
        scheduler.step()

        test_acc = evaluate(model, x_test, y_test, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1

        print(f'Epoch {epoch + 1:2d}/{epochs} | '
              f'Loss: {loss.item():.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Test Acc: {test_acc:.4f}')

    print("="*70)
    print("Training complete!")
    print(f"Best Test Accuracy: {best_test_acc:.4f} (achieved at epoch {best_epoch})")
    print("="*70)

    return best_test_acc, best_epoch


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MNIST Digit Classification with DWN EncoderLayer')
    parser.add_argument('--estimator', type=str, default='glt', choices=['ste', 'finite_difference', 'glt'],
                        help='Estimator type for EncoderLayer: ste, finite_difference, or glt (default: glt)')
    parser.add_argument('--fixed-thresholds', action='store_true',
                        help='Keep thresholds fixed (non-trainable) during training')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs (default: 25)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--hidden-size', type=int, default=10,
                        help='Hidden layer size (default: 10)')
    parser.add_argument('--thermometer-bits', type=int, default=200,
                        help='Number of thermometer bits per feature (default: 200)')
    parser.add_argument('--encoding-type', type=str, default='thermometer',
                        choices=['thermometer', 'lbp', 'lbp+thermometer', 'lbp_distributive'],
                        help='Encoding type: thermometer, lbp, lbp+thermometer, or lbp_distributive (default: thermometer)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store MNIST data (default: ./data)')
    args = parser.parse_args()

    print("="*70)
    print("MNIST Digit Classification with DWN EncoderLayer")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Encoding type: {args.encoding_type}")
    print(f"  - Estimator: {args.estimator}")
    print(f"  - Thresholds: {'Fixed (non-trainable)' if args.fixed_thresholds else 'Trainable'}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Hidden size: {args.hidden_size}")
    print(f"  - Thermometer bits: {args.thermometer_bits}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    # Convert to numpy arrays and flatten images
    x_train = train_dataset.data.numpy().astype(np.float32).reshape(len(train_dataset.data), -1)  # Shape: (60000, 784)
    y_train = train_dataset.targets.numpy().astype(np.int64)
    x_test = test_dataset.data.numpy().astype(np.float32).reshape(len(test_dataset.data), -1)  # Shape: (10000, 784)
    y_test = test_dataset.targets.numpy().astype(np.int64)

    num_classes = 10
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    print(f"Dataset loaded:")
    print(f"  - Features shape: {x_train.shape} (flattened from 28x28 grayscale images)")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {class_names}")
    print(f"  - Training samples: {len(x_train)}")
    print(f"  - Test samples: {len(x_test)}")

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    plot_dataset(x_train, filename="before_mnist_train_dataset.png")
    plot_dataset(x_test, filename="before_mnist_test_dataset.png")

    sigma = 3.5

    # Clip to ±sigma standard deviations to reduce outliers
    print(f"\nClipping features to ±{sigma} standard deviations...")
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    clip_min = mean - sigma * std
    clip_max = mean + sigma * std

    print(f"  - Min value before clipping: {x_train.min().item():.4f}")
    print(f"  - Max value before clipping: {x_train.max().item():.4f}")

    x_train = torch.max(torch.min(x_train, clip_max), clip_min)
    x_test = torch.max(torch.min(x_test, clip_max), clip_min)

    # Get Min/Max after clipping
    print(f"  - Min value after clipping: {x_train.min().item():.4f}")
    print(f"  - Max value after clipping: {x_train.max().item():.4f}")

    # Normalize features to [-1, 1)
    print("\nNormalizing features to [-1, 1)...")
    x_min = x_train.min(dim=0, keepdim=True)[0]
    x_max = x_train.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    # Avoid division by zero for constant features
    x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)

    # Normalize: scale to [-1, 1)
    x_train = 2 * (x_train - x_min) / x_range - 1
    x_test = 2 * (x_test - x_min) / x_range - 1

    plot_dataset(x_train, filename="after_mnist_train_dataset.png")
    plot_dataset(x_test, filename="after_mnist_test_dataset.png")

    print(f"  - Min value: {x_train.min().item():.4f}")
    print(f"  - Max value: {x_train.max().item():.4f}")

    print(f"\nUsing EncoderLayer with encoding_type={args.encoding_type} ({args.thermometer_bits} bits)...")
    print(f"Original features shape: {x_train.shape}")

    # Build model with EncoderLayer
    print("\nBuilding DWN model with EncoderLayer...")
    num_features = x_train.size(1)
    image_shape = (28, 28)  # MNIST image dimensions

    # Determine encoder output size based on encoding type
    if args.encoding_type == 'thermometer':
        encoder_output_size = num_features * args.thermometer_bits
        encoder = EncoderLayer(num_features, args.thermometer_bits, input_dataset=x_train,
                               estimator_type=args.estimator, encoding_type=args.encoding_type)
    elif args.encoding_type == 'lbp':
        encoder_output_size = num_features  # LBP outputs one code per pixel
        encoder = EncoderLayer(num_features, args.thermometer_bits, input_dataset=x_train,
                               estimator_type=args.estimator, encoding_type=args.encoding_type,
                               image_shape=image_shape)
    elif args.encoding_type == 'lbp+thermometer':
        encoder_output_size = num_features * (8 + args.thermometer_bits)  # 8 LBP bits + thermometer bits
        encoder = EncoderLayer(num_features, args.thermometer_bits, input_dataset=x_train,
                               estimator_type=args.estimator, encoding_type=args.encoding_type,
                               image_shape=image_shape)
    elif args.encoding_type == 'lbp_distributive':
        encoder_output_size = num_features * args.thermometer_bits  # thermometer on LBP values
        encoder = EncoderLayer(num_features, args.thermometer_bits, input_dataset=x_train,
                               estimator_type=args.estimator, encoding_type=args.encoding_type,
                               image_shape=image_shape)

    print(f"  - Encoder output size: {encoder_output_size}")

    model = nn.Sequential(
        encoder,
        nn.Flatten(start_dim=1),
        dwn.LUTLayer(encoder_output_size, args.hidden_size, n=6, mapping='learnable'),
        dwn.GroupSum(k=num_classes, tau=1/0.3)
    )

    model = model.cuda()

    # Set threshold learning based on argument
    if args.fixed_thresholds:
        model[0].thresholds.requires_grad = False
        print("\nThresholds are FIXED (non-trainable)")
    else:
        model[0].thresholds.requires_grad = True
        print("\nThresholds are TRAINABLE")

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

    print(f"\nOptimizer configuration:")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Save initial thresholds for comparison (sorted)
    print("\nSaving initial thresholds for plotting...")
    if model[0].thresholds.data.ndim == 1:
        initial_thresholds = torch.sort(model[0].thresholds.data)[0].clone()
    else:
        initial_thresholds = torch.sort(model[0].thresholds.data, dim=1)[0].clone()

    # Train the model
    best_test_acc, best_epoch = train_and_evaluate(
        model, optimizer, scheduler, x_train, y_train, x_test, y_test,
        epochs=args.epochs, batch_size=128, device=device
    )

    # Plot threshold comparisons for first and last features (sorted)
    print("\nGenerating threshold comparison plots for first and last features...")
    if model[0].thresholds.data.ndim == 1:
        final_thresholds = torch.sort(model[0].thresholds.data)[0]
        print(f"  - Global thresholds (1D): {final_thresholds}")
        print("  - Skipping per-feature plots for 1D thresholds")
    else:
        final_thresholds = torch.sort(model[0].thresholds.data, dim=1)[0]
        num_features = x_train.size(1)
        plot_thresholds_comparison(initial_thresholds, final_thresholds, x_train, features_to_plot=[0, num_features - 1])
        print("All plots saved!")

    # Mapping and Truth Table Overview
    print("\n" + "="*70)
    print("MAPPING AND TRUTH TABLE OVERVIEW")
    print("="*70)

    lut_layer = model[2]

    # Mapping statistics
    print(f"\nLUT Layer Configuration:")
    print(f"  - Input size: {lut_layer.input_size}")
    print(f"  - Output size: {lut_layer.output_size}")
    print(f"  - Inputs per LUT (n): {lut_layer.n}")
    print(f"  - Truth table size per LUT: 2^{lut_layer.n} = {2**lut_layer.n}")
    print(f"  - Total parameters: {lut_layer.output_size * (2**lut_layer.n):,}")

    # Get mapping
    if hasattr(lut_layer, 'mapping'):
        mapping = lut_layer.mapping
        if hasattr(mapping, 'weights'):
            # LearnableMapping - extract discrete mapping from weights
            discrete_mapping = mapping.weights.argmax(dim=0)
            print(f"\nMapping type: LearnableMapping (soft routing)")
            print(f"  - Discrete mapping shape: {discrete_mapping.shape}")
            print(f"  - Weights shape: {mapping.weights.shape}")
        elif torch.is_tensor(mapping):
            # Fixed mapping (tensor)
            discrete_mapping = mapping
            print(f"\nMapping type: Fixed mapping (tensor)")
            print(f"  - Mapping shape: {discrete_mapping.shape}")
        else:
            # Other mapping type
            print(f"\nMapping type: {type(mapping).__name__}")
            discrete_mapping = mapping

        # Mapping statistics
        print(f"\nMapping Statistics:")
        unique_inputs = torch.unique(discrete_mapping)
        print(f"  - Unique inputs used: {len(unique_inputs)} / {lut_layer.input_size}")
        print(f"  - Coverage: {100 * len(unique_inputs) / lut_layer.input_size:.1f}%")

        # Count connections per input
        input_connections = torch.zeros(lut_layer.input_size)
        for inp in unique_inputs:
            input_connections[inp.item()] = (discrete_mapping == inp).sum().item()

        connected_inputs = (input_connections > 0).sum().item()
        print(f"  - Connected inputs: {connected_inputs} / {lut_layer.input_size}")
        print(f"  - Average connections per input: {input_connections[input_connections > 0].mean():.2f}")
        print(f"  - Max connections to single input: {input_connections.max().item():.0f}")
        print(f"  - Min connections (non-zero): {input_connections[input_connections > 0].min().item():.0f}")

        # Show first 3 LUTs mapping
        print(f"\nFirst 3 LUTs - Input Mappings:")
        for i in range(min(3, lut_layer.output_size)):
            inputs = discrete_mapping[i].cpu().numpy()
            print(f"  LUT {i}: inputs = {inputs}")

    # Truth table statistics
    print(f"\nTruth Table Statistics:")
    luts = lut_layer.luts.data
    print(f"  - Shape: {luts.shape}")
    print(f"  - Mean: {luts.mean().item():.4f}")
    print(f"  - Std: {luts.std().item():.4f}")
    print(f"  - Min: {luts.min().item():.4f}")
    print(f"  - Max: {luts.max().item():.4f}")

    # Sparsity analysis
    threshold = 0.01
    sparse_entries = (luts.abs() < threshold).sum().item()
    total_entries = luts.numel()
    print(f"  - Entries < {threshold}: {sparse_entries} / {total_entries} ({100*sparse_entries/total_entries:.1f}%)")

    # Show first 2 truth tables
    print(f"\nFirst 2 LUTs - Truth Tables:")
    for i in range(min(2, lut_layer.output_size)):
        tt = luts[i].cpu().numpy()
        print(f"  LUT {i}: {tt}")

    # Generate detailed LUT report with binarized truth tables
    print("\n" + "="*70)
    print("Generating detailed LUT report...")
    print("="*70)

    report_filename = "mnist_lut_report.md"
    with open(report_filename, 'w') as f:
        f.write("# MNIST LUT Layer Report\n\n")
        f.write(f"**Configuration:**\n")
        f.write(f"- Input size: {lut_layer.input_size}\n")
        f.write(f"- Output size: {lut_layer.output_size}\n")
        f.write(f"- Inputs per LUT (n): {lut_layer.n}\n")
        f.write(f"- Truth table size: 2^{lut_layer.n} = {2**lut_layer.n}\n")
        f.write(f"- Total parameters: {lut_layer.output_size * (2**lut_layer.n):,}\n\n")

        if hasattr(lut_layer, 'mapping'):
            mapping = lut_layer.mapping
            if hasattr(mapping, 'weights'):
                discrete_mapping = mapping.weights.argmax(dim=0).cpu()
                f.write(f"**Mapping type:** LearnableMapping\n\n")
            elif torch.is_tensor(mapping):
                discrete_mapping = mapping.cpu()
                f.write(f"**Mapping type:** Fixed\n\n")
            else:
                discrete_mapping = mapping.cpu() if hasattr(mapping, 'cpu') else mapping
                f.write(f"**Mapping type:** {type(mapping).__name__}\n\n")

            # Binarize truth tables: > 0 → 1, else → 0
            luts_continuous = lut_layer.luts.data.cpu()
            luts_binary = (luts_continuous > 0).int()

            f.write("---\n\n")
            f.write("## Individual LUT Details\n\n")
            f.write("*Truth tables binarized: value > 0 → 1, else → 0*\n\n")

            # Write details for each LUT
            for i in range(lut_layer.output_size):
                f.write(f"### LUT {i}\n\n")

                # Input mapping
                inputs = discrete_mapping[i].numpy() if hasattr(discrete_mapping[i], 'numpy') else discrete_mapping[i]
                # Handle both arrays and scalars
                inputs_list = np.atleast_1d(inputs).tolist()
                f.write(f"**Input mapping:** `{inputs_list}`\n\n")

                # Continuous truth table stats
                tt_continuous = luts_continuous[i].numpy()
                f.write(f"**Continuous truth table stats:**\n")
                f.write(f"- Mean: {tt_continuous.mean():.4f}\n")
                f.write(f"- Std: {tt_continuous.std():.4f}\n")
                f.write(f"- Min: {tt_continuous.min():.4f}\n")
                f.write(f"- Max: {tt_continuous.max():.4f}\n\n")

                # Binary truth table
                tt_binary = luts_binary[i].numpy()
                f.write(f"**Binary truth table (>0 → 1):**\n")
                f.write(f"```\n{tt_binary}\n```\n\n")

                # Show truth table in table format for smaller LUTs
                if 2**lut_layer.n <= 64:
                    f.write(f"**Truth table (table format):**\n\n")
                    f.write("| Index | Binary Input | Output |\n")
                    f.write("|-------|--------------|--------|\n")
                    for idx in range(2**lut_layer.n):
                        binary_input = format(idx, f'0{lut_layer.n}b')
                        output = tt_binary[idx]
                        f.write(f"| {idx:3d} | {binary_input} | {output} |\n")
                    f.write("\n")

                # Statistics
                ones_count = tt_binary.sum()
                zeros_count = len(tt_binary) - ones_count
                f.write(f"**Binary statistics:**\n")
                f.write(f"- Ones: {ones_count} / {len(tt_binary)} ({100*ones_count/len(tt_binary):.1f}%)\n")
                f.write(f"- Zeros: {zeros_count} / {len(tt_binary)} ({100*zeros_count/len(tt_binary):.1f}%)\n\n")

                f.write("---\n\n")

        f.write("\n## Summary Statistics\n\n")
        f.write(f"- Total LUTs: {lut_layer.output_size}\n")
        if hasattr(lut_layer, 'mapping'):
            overall_ones = luts_binary.sum().item()
            overall_total = luts_binary.numel()
            f.write(f"- Overall binary distribution: {overall_ones} ones / {overall_total} total ({100*overall_ones/overall_total:.1f}% ones)\n")

    print(f"Report saved to: {report_filename}")
    print(f"  Contains {lut_layer.output_size} LUTs with binarized truth tables")

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
