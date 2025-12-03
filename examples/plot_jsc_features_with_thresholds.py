"""
Plot JSC Dataset Features with Thresholds Overlay
This script visualizes features with distributive and uniform thresholds overlaid.
"""
import torch
import openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.insert(0, '..')
from torch_dwn.binarization import DistributiveThermometer, Thermometer


def plot_features_with_thresholds(data, num_features=16, rows=2, cols=8,
                                   num_bits=200, filename="jsc_features_thresholds.png",
                                   show_distributive=True, show_uniform=True):
    """
    Plot features in an MxN grid layout with thresholds overlaid

    Args:
        data: torch tensor of shape (samples, features)
        num_features: number of features to plot
        rows: number of rows in the grid
        cols: number of columns in the grid
        num_bits: number of thermometer bits (thresholds)
        filename: output filename
        show_distributive: show distributive thresholds
        show_uniform: show uniform thresholds
    """
    num_features = min(num_features, data.shape[1])

    # Initialize thermometer encoders
    input_min = data.min().item()
    input_max = data.max().item()

    print(f"\nInitializing thermometer encoders...")
    print(f"  - Input range: [{input_min:.4f}, {input_max:.4f}]")
    print(f"  - Number of bits: {num_bits}")

    if show_distributive:
        dist_therm = DistributiveThermometer(num_bits=num_bits, feature_wise=True)
        dist_therm.fit(data)
        dist_thresholds = dist_therm.thresholds.cpu().numpy()
        print(f"  - Distributive thresholds shape: {dist_thresholds.shape}")

    if show_uniform:
        uniform_therm = Thermometer(num_bits=num_bits, feature_wise=True)
        uniform_therm.fit(data)
        uniform_thresholds = uniform_therm.thresholds.cpu().numpy()
        print(f"  - Uniform thresholds shape: {uniform_thresholds.shape}")

    # Create figure with specified grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()
    
    linewidth = 4

    # Plot each feature
    for i in range(num_features):
        feature_data = data[:, i].cpu().numpy()

        # Create histogram
        axes[i].hist(feature_data, bins=50, color='lightblue', edgecolor='black',
                    alpha=0.6, label='Data', density=True)

        # Overlay thresholds
        y_max = axes[i].get_ylim()[1]

        if show_distributive:
            for thresh in dist_thresholds[i]:
                axes[i].axvline(x=thresh, color='green', alpha=0.15, linewidth=linewidth)
            # Add one visible line for legend
            axes[i].axvline(x=dist_thresholds[i][0], color='green', alpha=0.6,
                          linewidth=1.5, label=f'Distributive ({num_bits})')

        if show_uniform:
            for thresh in uniform_thresholds[i]:
                axes[i].axvline(x=thresh, color='red', alpha=0.15, linewidth=linewidth)
            # Add one visible line for legend
            axes[i].axvline(x=uniform_thresholds[i][0], color='red', alpha=0.6,
                          linewidth=1.5, label=f'Uniform ({num_bits})')

        #axes[i].set_title(f'Feature {i}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=8)
        axes[i].set_ylabel('Density', fontsize=8)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=7, loc='upper right')

        # Add statistics text
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        min_val = feature_data.min()
        max_val = feature_data.max()

        #stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nmin={min_val:.2f}\nmax={max_val:.2f}'
        #axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
        #            fontsize=7, verticalalignment='top', horizontalalignment='left',
        #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Hide any unused subplots
    for i in range(num_features, rows * cols):
        axes[i].axis('off')

    # Overall title
    threshold_types = []
    if show_distributive:
        threshold_types.append("Distributive")
    if show_uniform:
        threshold_types.append("Uniform")
    threshold_str = " & ".join(threshold_types)

    fig.suptitle(f'JSC Dataset - Features with {threshold_str} Thresholds ({rows}x{cols} Grid)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n✓ Saved plot: {filename}')


def plot_threshold_comparison(data, feature_idx=0, num_bits=200, filename="threshold_comparison.png"):
    """
    Detailed comparison plot for a single feature showing threshold distributions

    Args:
        data: torch tensor of shape (samples, features)
        feature_idx: which feature to plot
        num_bits: number of thermometer bits
        filename: output filename
    """
    input_min = data.min().item()
    input_max = data.max().item()

    # Initialize thermometer encoders
    dist_therm = DistributiveThermometer(num_bits=num_bits, feature_wise=True)
    dist_therm.fit(data)

    uniform_therm = Thermometer(num_bits=num_bits, feature_wise=True)
    uniform_therm.fit(data)

    dist_thresholds = dist_therm.thresholds[feature_idx].cpu().numpy()
    uniform_thresholds = uniform_therm.thresholds[feature_idx].cpu().numpy()

    feature_data = data[:, feature_idx].cpu().numpy()

    # Create detailed comparison figure - only histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    linewidth = 4  # Set threshold line width

    # Left: Histogram with distributive thresholds
    ax1.hist(feature_data, bins=100, color='lightgreen', edgecolor='black', alpha=0.6, density=True)
    for thresh in dist_thresholds:
        ax1.axvline(x=thresh, color='green', alpha=0.4, linewidth=linewidth)
    #ax1.set_title(f'Feature {feature_idx} - Distributive', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=20)
    ax1.set_ylabel('Density', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid(True, alpha=0.3)

    # Right: Histogram with uniform thresholds
    ax2.hist(feature_data, bins=100, color='lightcoral', edgecolor='black', alpha=0.6, density=True)
    for thresh in uniform_thresholds:
        ax2.axvline(x=thresh, color='red', alpha=0.4, linewidth=linewidth)
    #ax2.set_title(f'Feature {feature_idx} - Uniform', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=20)
    ax2.set_ylabel('Density', fontsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved detailed comparison: {filename}')


def load_and_preprocess_jsc(sigma=3.5, normalize=True):
    """Load JSC dataset and optionally preprocess it"""
    print("\nLoading JSC dataset from OpenML (ID: 42468)...")
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )

    # Convert to numpy arrays
    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

    print(f"Dataset loaded:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Feature names: {list(attribute_names)}")

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    if not normalize:
        return x_train, x_test

    # Preprocessing: clipping and normalization
    print(f"\nPreprocessing:")
    print(f"  - Clipping to ±{sigma} standard deviations...")
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    clip_min = mean - sigma * std
    clip_max = mean + sigma * std

    x_train = torch.max(torch.min(x_train, clip_max), clip_min)
    x_test = torch.max(torch.min(x_test, clip_max), clip_min)

    # Normalize to [-1, 1)
    print(f"  - Normalizing to [-1, 1)...")
    x_min = x_train.min(dim=0, keepdim=True)[0]
    x_max = x_train.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)

    x_train = 2 * (x_train - x_min) / x_range - 1
    x_test = 2 * (x_test - x_min) / x_range - 1

    print(f"    Final: min={x_train.min().item():.4f}, max={x_train.max().item():.4f}")

    return x_train, x_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot JSC Features with Thresholds')
    parser.add_argument('--features', type=int, default=16,
                        help='Number of features to plot (default: 16)')
    parser.add_argument('--rows', type=int, default=2,
                        help='Number of rows in grid (default: 2)')
    parser.add_argument('--cols', type=int, default=8,
                        help='Number of columns in grid (default: 8)')
    parser.add_argument('--bits', type=int, default=200,
                        help='Number of thermometer bits/thresholds (default: 200)')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply preprocessing (clipping and normalization)')
    parser.add_argument('--sigma', type=float, default=3.5,
                        help='Sigma for clipping outliers (default: 3.5)')
    parser.add_argument('--distributive-only', action='store_true',
                        help='Show only distributive thresholds')
    parser.add_argument('--uniform-only', action='store_true',
                        help='Show only uniform thresholds')
    parser.add_argument('--detailed-feature', type=int, default=None,
                        help='Generate detailed comparison plot for specific feature index')
    parser.add_argument('--output', type=str, default='jsc_features_thresholds.png',
                        help='Output filename (default: jsc_features_thresholds.png)')
    args = parser.parse_args()

    # Determine which thresholds to show
    show_dist = not args.uniform_only
    show_uni = not args.distributive_only

    print("="*70)
    print("JSC Dataset Feature Visualization with Thresholds")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Grid layout: {args.rows}x{args.cols}")
    print(f"  - Number of features: {args.features}")
    print(f"  - Thermometer bits: {args.bits}")
    print(f"  - Show distributive: {show_dist}")
    print(f"  - Show uniform: {show_uni}")
    print(f"  - Preprocessing: {'Yes' if args.normalize else 'No (raw data)'}")
    if args.normalize:
        print(f"  - Sigma for clipping: {args.sigma}")

    # Load data
    x_train, x_test = load_and_preprocess_jsc(sigma=args.sigma, normalize=args.normalize)

    # Plot grid with thresholds
    print("\nGenerating feature plots with thresholds...")
    plot_features_with_thresholds(
        x_train,
        num_features=args.features,
        rows=args.rows,
        cols=args.cols,
        num_bits=args.bits,
        filename=args.output,
        show_distributive=show_dist,
        show_uniform=show_uni
    )

    # Generate detailed comparison for specific feature if requested
    if args.detailed_feature is not None:
        detail_filename = args.output.replace('.png', f'_feature{args.detailed_feature}_detailed.png')
        print(f"\nGenerating detailed comparison for feature {args.detailed_feature}...")
        plot_threshold_comparison(
            x_train,
            feature_idx=args.detailed_feature,
            num_bits=args.bits,
            filename=detail_filename
        )

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
