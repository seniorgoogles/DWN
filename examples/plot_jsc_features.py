"""
Plot JSC Dataset Features in MxN Grid Format
This script loads the JSC dataset and plots feature distributions in a configurable grid layout.
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


def plot_features_grid(data, num_features=16, rows=2, cols=8, filename="jsc_features_grid.png",
                       title_prefix="Feature", normalized=False, num_bits=None,
                       show_distributive=False, show_uniform=False):
    """
    Plot features in an MxN grid layout

    Args:
        data: torch tensor of shape (samples, features)
        num_features: number of features to plot (default: 16, all features)
        rows: number of rows in the grid
        cols: number of columns in the grid
        filename: output filename
        title_prefix: prefix for subplot titles
        normalized: whether the data has been normalized
        num_bits: number of thermometer bits/thresholds (if None, no thresholds plotted)
        show_distributive: show distributive thresholds
        show_uniform: show uniform thresholds
    """
    num_features = min(num_features, data.shape[1])

    # Initialize thresholds if requested
    dist_thresholds = None
    uniform_thresholds = None

    if num_bits is not None and (show_distributive or show_uniform):
        print(f"\nInitializing thermometer encoders with {num_bits} bits...")

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

    # Plot each feature
    for i in range(num_features):
        feature_data = data[:, i].cpu().numpy()

        # Create histogram with density if showing thresholds
        use_density = num_bits is not None and (show_distributive or show_uniform)
        axes[i].hist(feature_data, bins=50, color='lightblue', edgecolor='black',
                    alpha=0.6, density=use_density)

        # Overlay thresholds if requested (thicker lines)
        if dist_thresholds is not None:
            for thresh in dist_thresholds[i]:
                axes[i].axvline(x=thresh, color='green', alpha=0.3, linewidth=1.2)
            # Add one visible line for legend
            axes[i].axvline(x=dist_thresholds[i][0], color='green', alpha=0.7,
                          linewidth=2.5, label=f'Distributive ({num_bits})')

        if uniform_thresholds is not None:
            for thresh in uniform_thresholds[i]:
                axes[i].axvline(x=thresh, color='red', alpha=0.3, linewidth=1.2)
            # Add one visible line for legend
            axes[i].axvline(x=uniform_thresholds[i][0], color='red', alpha=0.7,
                          linewidth=2.5, label=f'Uniform ({num_bits})')

        axes[i].set_title(f'{title_prefix} {i}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=8)
        axes[i].set_ylabel('Density' if use_density else 'Frequency', fontsize=8)
        axes[i].grid(True, alpha=0.3)

        if dist_thresholds is not None or uniform_thresholds is not None:
            axes[i].legend(fontsize=7, loc='upper right')

        # Add statistics text
        mean_val = feature_data.mean()
        std_val = feature_data.std()
        min_val = feature_data.min()
        max_val = feature_data.max()

        stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nmin={min_val:.2f}\nmax={max_val:.2f}'
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                    fontsize=7, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Hide any unused subplots
    for i in range(num_features, rows * cols):
        axes[i].axis('off')

    # Overall title
    threshold_info = ""
    if num_bits is not None and (show_distributive or show_uniform):
        threshold_types = []
        if show_distributive:
            threshold_types.append("Distributive")
        if show_uniform:
            threshold_types.append("Uniform")
        threshold_info = f" with {' & '.join(threshold_types)} Thresholds"

    fig.suptitle(f'JSC Dataset - {"Normalized" if normalized else "Raw"} Features{threshold_info} ({rows}x{cols} Grid)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved plot: {filename}')


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
    num_classes = labels.max() + 1

    print(f"Dataset loaded:")
    print(f"  - Features shape: {features.shape}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Class names: {label_names}")
    print(f"  - Feature names: {list(attribute_names)}")

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels,
        train_size=0.8,
        random_state=42
    )

    print(f"\nTrain/test split (80/20):")
    print(f"  - Training samples: {len(x_train)}")
    print(f"  - Test samples: {len(x_test)}")

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

    print(f"    Before: min={x_train.min().item():.4f}, max={x_train.max().item():.4f}")
    x_train = torch.max(torch.min(x_train, clip_max), clip_min)
    x_test = torch.max(torch.min(x_test, clip_max), clip_min)
    print(f"    After:  min={x_train.min().item():.4f}, max={x_train.max().item():.4f}")

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
    parser = argparse.ArgumentParser(description='Plot JSC Dataset Features in Grid Format')
    parser.add_argument('--features', type=int, default=16,
                        help='Number of features to plot (default: 16, all features)')
    parser.add_argument('--rows', type=int, default=2,
                        help='Number of rows in grid (default: 2)')
    parser.add_argument('--cols', type=int, default=8,
                        help='Number of columns in grid (default: 8)')
    parser.add_argument('--normalize', action='store_true',
                        help='Apply preprocessing (clipping and normalization to [-1, 1))')
    parser.add_argument('--sigma', type=float, default=3.5,
                        help='Sigma for clipping outliers (default: 3.5)')
    parser.add_argument('--output', type=str, default='jsc_features_grid.png',
                        help='Output filename (default: jsc_features_grid.png)')
    parser.add_argument('--bits', type=int, default=None,
                        help='Number of thermometer bits/thresholds to overlay (default: None, no thresholds)')
    parser.add_argument('--distributive', action='store_true',
                        help='Show distributive thresholds (requires --bits)')
    parser.add_argument('--uniform', action='store_true',
                        help='Show uniform thresholds (requires --bits)')
    args = parser.parse_args()

    # Determine which thresholds to show
    show_dist = args.distributive
    show_uni = args.uniform

    # If bits specified but no threshold type, show both
    if args.bits is not None and not show_dist and not show_uni:
        show_dist = True
        show_uni = True

    print("="*70)
    print("JSC Dataset Feature Visualization")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Grid layout: {args.rows}x{args.cols}")
    print(f"  - Number of features: {args.features}")
    print(f"  - Preprocessing: {'Yes' if args.normalize else 'No (raw data)'}")
    if args.normalize:
        print(f"  - Sigma for clipping: {args.sigma}")
    if args.bits is not None:
        print(f"  - Thermometer bits: {args.bits}")
        print(f"  - Show distributive: {show_dist}")
        print(f"  - Show uniform: {show_uni}")
    print(f"  - Output file: {args.output}")

    # Load data
    x_train, x_test = load_and_preprocess_jsc(sigma=args.sigma, normalize=args.normalize)

    # Plot training set features
    print("\nGenerating feature plots...")
    plot_features_grid(
        x_train,
        num_features=args.features,
        rows=args.rows,
        cols=args.cols,
        filename=args.output,
        title_prefix="Feature",
        normalized=args.normalize,
        num_bits=args.bits,
        show_distributive=show_dist,
        show_uniform=show_uni
    )

    # Also plot test set if desired
    test_filename = args.output.replace('.png', '_test.png')
    plot_features_grid(
        x_test,
        num_features=args.features,
        rows=args.rows,
        cols=args.cols,
        filename=test_filename,
        title_prefix="Feature",
        normalized=args.normalize,
        num_bits=args.bits,
        show_distributive=show_dist,
        show_uniform=show_uni
    )

    print("\n" + "="*70)
    print("Visualization complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - Training set: {args.output}")
    print(f"  - Test set: {test_filename}")
