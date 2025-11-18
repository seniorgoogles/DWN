"""
JSC Dataset Classification using DWN (Deep Weight-sharing Network)
OpenML Dataset ID: 42468
This example demonstrates using LUTLayers for JSC classification.
"""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
from torch_dwn import EncoderLayer
import openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse


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

        plt.savefig(f'feature{feature_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved plot: feature{feature_idx}.png')


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


def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size,
                       device, threshold_warmup_epochs=0, threshold_update_frequency=1):
    """Train model and evaluate after each epoch"""
    n_samples = x_train.shape[0]
    best_test_acc = 0.0
    best_epoch = 0
    batch_counter = 0

    print("\n" + "="*70)
    print(f"Training for {epochs} epochs (batch_size={batch_size})")
    print(f"Threshold warmup: {threshold_warmup_epochs} epochs")
    print(f"Threshold update frequency: every {threshold_update_frequency} batches")
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
    print("="*70)
    print("JSC Dataset Classification with DWN LUTLayers")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load JSC dataset from OpenML
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
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    plot_dataset(x_train, filename="before_jsc_train_dataset.png")
    plot_dataset(x_test, filename="before_jsc_test_dataset.png")

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

    plot_dataset(x_train, filename="after_jsc_train_dataset.png")
    plot_dataset(x_test, filename="after_jsc_test_dataset.png")

    print(f"  - Min value: {x_train.min().item():.4f}")
    print(f"  - Max value: {x_train.max().item():.4f}")

    print("\nUsing EncoderLayer with DistributiveThermometer (200 bits)...")
    print(f"Original features shape: {x_train.shape}")
    print(f"  - Each feature will be converted to 200 binary values")
    print(f"  - Thresholds initialized from DistributiveThermometer")

    # Build model with LUTLayers
    print("\nBuilding DWN model with EncoderLayer...")
    # hidden_size = 16030 to match VC dimension of jsc_default.py (1,025,920 parameters)
    # Calculation: LUTLayer params = hidden_size × 2^n = 16030 × 64 = 1,025,920
    learning_rate = 1e-3
    hidden_size = 10
    epochs = 25
    num_features = x_train.size(1)
    thermometer_bits = 200

    # Threshold training control
    threshold_warmup_epochs = 0  # Don't update thresholds for first N epochs
    threshold_update_frequency = 1  # Update thresholds every N epochs (1 = every epoch)
    importance_enable_epoch = 12  # Enable importance weighting after this epoch

    model = nn.Sequential(
        EncoderLayer(num_features, thermometer_bits, input_dataset=x_train),
        nn.Flatten(start_dim=1),
        dwn.LUTLayer(num_features * thermometer_bits, hidden_size, n=6, mapping='learnable'),
        dwn.GroupSum(k=num_classes, tau=1/0.3)
    )

    model = model.cuda()

    # Enable threshold learning
    model[0].thresholds.requires_grad = True

    # Single learning rate for all parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

    print(f"\nOptimizer configuration:")
    print(f"  - Learning rate: {learning_rate} (uniform for all parameters)")
    print(f"  - Importance weighting: enabled at epoch {importance_enable_epoch}")

    # Save initial thresholds for comparison (sorted)
    print("\nSaving initial thresholds for plotting...")
    initial_thresholds = torch.sort(model[0].thresholds.data, dim=1)[0].clone()

    # PHASE 1: Train without importance (let LUTs learn connections)
    print("\n" + "="*70)
    print(f"PHASE 1: Training without importance ({importance_enable_epoch} epochs)")
    print("="*70)
    best_test_acc, best_epoch = train_and_evaluate(
        model, optimizer, scheduler, x_train, y_train, x_test, y_test,
        epochs=importance_enable_epoch, batch_size=128, device=device,
        threshold_warmup_epochs=threshold_warmup_epochs,
        threshold_update_frequency=threshold_update_frequency
    )

    # Extract importance from trained LUT mapping
    print("\n" + "="*70)
    print("Extracting importance from trained LUT mapping...")
    print("="*70)
    model[0].set_importance_from_lut(model[2], method='weighted', min_weight=0.1)

    importance = model[0].importance_weights
    print(f"\nImportance statistics:")
    print(f"  - Mean: {importance.mean():.4f}")
    print(f"  - Zero importance bits: {(importance == 0).sum().item()} / {importance.numel()}")
    print(f"  - Coverage: {100 * (importance > 0).float().mean():.1f}% of bits connected")

    # PHASE 2: Continue training WITH importance
    print("\n" + "="*70)
    print(f"PHASE 2: Training WITH importance ({epochs - importance_enable_epoch} more epochs)")
    print("  - Important bits: 100% gradient")
    print("  - Unused bits: 10% gradient (can still learn!)")
    print("="*70)

    remaining_epochs = epochs - importance_enable_epoch
    best_test_acc_phase2, best_epoch_phase2 = train_and_evaluate(
        model, optimizer, scheduler, x_train, y_train, x_test, y_test,
        epochs=remaining_epochs, batch_size=128, device=device,
        threshold_warmup_epochs=0,  # Already warmed up
        threshold_update_frequency=threshold_update_frequency
    )

    # Update best results
    best_test_acc = max(best_test_acc, best_test_acc_phase2)

    # Plot threshold comparisons for features 0-15 (sorted)
    print("\nGenerating threshold comparison plots for features 0-15...")
    final_thresholds = torch.sort(model[0].thresholds.data, dim=1)[0]
    plot_thresholds_comparison(initial_thresholds, final_thresholds, x_train, features_to_plot=range(16))
    print("All plots saved!")
