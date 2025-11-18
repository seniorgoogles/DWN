"""
Example: Minimum Threshold Spacing During Training

Demonstrates how to prevent threshold collapse by enforcing minimum spacing
after gradient updates in backpropagation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')

from torch_dwn.encoder_layer import EncoderLayer


def visualize_spacing(thresholds, title, filename):
    """Helper to visualize threshold spacing"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot thresholds
    num_features = thresholds.shape[0]
    for f in range(min(3, num_features)):  # Show first 3 features
        sorted_thresh = torch.sort(thresholds[f])[0].cpu().numpy()
        x_points = range(len(sorted_thresh))
        ax1.plot(x_points, sorted_thresh, 'o-', label=f'Feature {f}',
                linewidth=2, markersize=8, alpha=0.7)

    ax1.set_xlabel('Threshold Index', fontsize=12)
    ax1.set_ylabel('Threshold Value', fontsize=12)
    ax1.set_title('Threshold Values', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot spacing between consecutive thresholds
    for f in range(min(3, num_features)):
        sorted_thresh = torch.sort(thresholds[f])[0].cpu().numpy()
        spacings = sorted_thresh[1:] - sorted_thresh[:-1]
        x_points = range(len(spacings))
        ax2.plot(x_points, spacings, 'o-', label=f'Feature {f}',
                linewidth=2, markersize=8, alpha=0.7)

    ax2.set_xlabel('Threshold Pair Index', fontsize=12)
    ax2.set_ylabel('Spacing (t[i+1] - t[i])', fontsize=12)
    ax2.set_title('Spacing Between Consecutive Thresholds', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero spacing')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {filename}")
    plt.close()


def example_1_without_spacing_constraint():
    """
    Example 1: Training WITHOUT minimum spacing - thresholds can collapse
    """
    print("=" * 70)
    print("Example 1: Training WITHOUT Minimum Spacing Constraint")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(1000, 5)
    train_labels = torch.randint(0, 2, (1000, 10)).float()

    # Create encoder WITHOUT spacing constraint
    encoder = EncoderLayer(
        inputs=5,
        output_size=8,
        input_dataset=train_data,
        min_threshold_spacing=None  # No spacing constraint
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(5 * 8, 10)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.05)  # High LR to induce collapse
    criterion = nn.BCEWithLogitsLoss()

    print("\nInitial thresholds (Feature 0):")
    print(torch.sort(encoder.thresholds[0])[0])

    # Train without spacing enforcement
    for epoch in range(20):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Note: NOT calling encoder.smooth_thresholds()

    print("\nFinal thresholds (Feature 0) - may have collapsed:")
    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    print(sorted_thresh)

    # Check for collapsed thresholds
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]
    min_spacing = spacings.min().item()
    print(f"\nMinimum spacing between thresholds: {min_spacing:.6f}")
    print(f"Number of near-zero spacings (<0.01): {(spacings < 0.01).sum().item()}")

    visualize_spacing(encoder.thresholds.data,
                     "WITHOUT Minimum Spacing Constraint",
                     "spacing_without_constraint.png")

    return encoder


def example_2_with_spacing_constraint():
    """
    Example 2: Training WITH minimum spacing - prevents collapse
    """
    print("\n" + "=" * 70)
    print("Example 2: Training WITH Minimum Spacing Constraint")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(1000, 5)
    train_labels = torch.randint(0, 2, (1000, 10)).float()

    # Create encoder WITH spacing constraint
    encoder = EncoderLayer(
        inputs=5,
        output_size=8,
        input_dataset=train_data,
        min_threshold_spacing=0.1,      # Enforce minimum spacing!
        spacing_method='push_apart'      # Method to enforce spacing
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(5 * 8, 10)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.05)  # Same high LR
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nMinimum spacing enforced: {encoder.min_threshold_spacing}")
    print(f"Spacing method: {encoder.spacing_method}")
    print("\nInitial thresholds (Feature 0):")
    print(torch.sort(encoder.thresholds[0])[0])

    # Train WITH spacing enforcement
    for epoch in range(20):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Enforce spacing after gradient update!
            encoder.smooth_thresholds()

    print("\nFinal thresholds (Feature 0) - spacing maintained:")
    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    print(sorted_thresh)

    # Check spacing
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]
    min_spacing = spacings.min().item()
    print(f"\nMinimum spacing between thresholds: {min_spacing:.6f}")
    print(f"All spacings >= minimum: {(spacings >= encoder.min_threshold_spacing - 1e-6).all().item()}")

    visualize_spacing(encoder.thresholds.data,
                     "WITH Minimum Spacing Constraint (push_apart)",
                     "spacing_with_constraint.png")

    return encoder


def example_3_compare_spacing_methods():
    """
    Example 3: Compare different spacing enforcement methods
    """
    print("\n" + "=" * 70)
    print("Example 3: Comparing Spacing Enforcement Methods")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(800, 4)
    train_labels = torch.randint(0, 2, (800, 8)).float()

    methods = ['push_apart', 'sequential', 'redistribute']
    results = {}

    for method in methods:
        print(f"\nTraining with '{method}' spacing method...")

        encoder = EncoderLayer(
            inputs=4,
            output_size=10,
            input_dataset=train_data,
            min_threshold_spacing=0.08,
            spacing_method=method
        )

        class SimpleModel(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
                self.fc = nn.Linear(4 * 10, 8)

            def forward(self, x):
                encoded = self.encoder(x)
                return self.fc(encoded.flatten(1))

        model = SimpleModel(encoder)
        optimizer = optim.Adam(model.parameters(), lr=0.03)
        criterion = nn.BCEWithLogitsLoss()

        # Train
        losses = []
        for epoch in range(15):
            epoch_loss = 0
            num_batches = 0

            for i in range(0, len(train_data), 32):
                batch_x = train_data[i:i+32]
                batch_y = train_labels[i:i+32]

                output = model(batch_x)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                encoder.smooth_thresholds()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)

        # Analyze results
        sorted_thresh = torch.sort(encoder.thresholds[0])[0]
        spacings = sorted_thresh[1:] - sorted_thresh[:-1]

        results[method] = {
            'losses': losses,
            'thresholds': sorted_thresh.cpu().numpy(),
            'spacings': spacings.cpu().numpy(),
            'min_spacing': spacings.min().item(),
            'mean_spacing': spacings.mean().item(),
            'std_spacing': spacings.std().item()
        }

        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Min spacing: {results[method]['min_spacing']:.6f}")
        print(f"  Mean spacing: {results[method]['mean_spacing']:.6f}")

    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training curves
    for method, data in results.items():
        axes[0, 0].plot(data['losses'], label=method, linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss by Spacing Method', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Final thresholds
    x_points = range(10)
    for method, data in results.items():
        axes[0, 1].plot(x_points, data['thresholds'], 'o-', label=method,
                       linewidth=2, markersize=6, alpha=0.7)
    axes[0, 1].set_xlabel('Threshold Index', fontsize=11)
    axes[0, 1].set_ylabel('Threshold Value', fontsize=11)
    axes[0, 1].set_title('Final Thresholds (Feature 0)', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Spacing distribution
    x_points = range(9)  # 9 spacings for 10 thresholds
    for method, data in results.items():
        axes[1, 0].plot(x_points, data['spacings'], 'o-', label=method,
                       linewidth=2, markersize=6, alpha=0.7)
    axes[1, 0].axhline(y=0.08, color='r', linestyle='--', alpha=0.5,
                       label='Min required (0.08)')
    axes[1, 0].set_xlabel('Threshold Pair Index', fontsize=11)
    axes[1, 0].set_ylabel('Spacing', fontsize=11)
    axes[1, 0].set_title('Spacing Between Consecutive Thresholds', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Statistics comparison
    methods_list = list(results.keys())
    min_spacings = [results[m]['min_spacing'] for m in methods_list]
    mean_spacings = [results[m]['mean_spacing'] for m in methods_list]

    x = range(len(methods_list))
    width = 0.35
    axes[1, 1].bar([i - width/2 for i in x], min_spacings, width, label='Min spacing', alpha=0.8)
    axes[1, 1].bar([i + width/2 for i in x], mean_spacings, width, label='Mean spacing', alpha=0.8)
    axes[1, 1].axhline(y=0.08, color='r', linestyle='--', alpha=0.5,
                       label='Required min (0.08)')
    axes[1, 1].set_xlabel('Method', fontsize=11)
    axes[1, 1].set_ylabel('Spacing Value', fontsize=11)
    axes[1, 1].set_title('Spacing Statistics by Method', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods_list)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('spacing_methods_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison saved: spacing_methods_comparison.png")
    plt.close()


def example_4_combined_spacing_and_smoothing():
    """
    Example 4: Combine minimum spacing with spline smoothing
    """
    print("\n" + "=" * 70)
    print("Example 4: Combined Spacing Enforcement + Spline Smoothing")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(500, 3)
    train_labels = torch.randint(0, 2, (500, 6)).float()

    # Create encoder with BOTH spacing AND smoothing
    encoder = EncoderLayer(
        inputs=3,
        output_size=12,
        input_dataset=train_data,
        # Spacing constraint
        min_threshold_spacing=0.05,
        spacing_method='redistribute',
        # Spline smoothing
        enable_spline_smoothing=True,
        smoothing_method='monotonic',
        smooth_every_n_steps=15
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(3 * 12, 6)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nConfiguration:")
    print(f"  Minimum spacing: {encoder.min_threshold_spacing}")
    print(f"  Spacing method: {encoder.spacing_method}")
    print(f"  Spline smoothing: {encoder.enable_spline_smoothing}")
    print(f"  Smoothing method: {encoder.smoothing_method}")
    print(f"  Smooth every: {encoder.smooth_every_n_steps} steps")

    # Track evolution
    snapshots = []
    snapshot_steps = [0, 50, 150, 300]
    step = 0

    print("\nTraining with combined constraints...")
    for epoch in range(15):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            if step in snapshot_steps:
                snapshots.append({
                    'step': step,
                    'thresholds': torch.sort(encoder.thresholds[0])[0].clone().cpu().numpy()
                })

            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # This applies BOTH smoothing (periodic) AND spacing (always)!
            encoder.smooth_thresholds()

            step += 1

    # Final snapshot
    snapshots.append({
        'step': step,
        'thresholds': torch.sort(encoder.thresholds[0])[0].clone().cpu().numpy()
    })

    # Visualize evolution
    plt.figure(figsize=(12, 6))

    for snapshot in snapshots:
        step_num = snapshot['step']
        thresholds = snapshot['thresholds']
        x_points = range(len(thresholds))
        plt.plot(x_points, thresholds, 'o-', label=f'Step {step_num}',
                linewidth=2, markersize=6, alpha=0.7)

    plt.xlabel('Threshold Index', fontsize=12)
    plt.ylabel('Threshold Value', fontsize=12)
    plt.title('Threshold Evolution: Spacing + Smoothing', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('combined_spacing_smoothing.png', dpi=150)
    print(f"\nEvolution saved: combined_spacing_smoothing.png")
    plt.close()

    # Final analysis
    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]
    print(f"\nFinal analysis:")
    print(f"  Min spacing: {spacings.min().item():.6f}")
    print(f"  Mean spacing: {spacings.mean().item():.6f}")
    print(f"  All spacings >= minimum: {(spacings >= encoder.min_threshold_spacing - 1e-6).all().item()}")


if __name__ == "__main__":
    example_1_without_spacing_constraint()
    example_2_with_spacing_constraint()
    example_3_compare_spacing_methods()
    example_4_combined_spacing_and_smoothing()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Minimum spacing prevents threshold collapse during training")
    print("2. 'push_apart' method works from center outward")
    print("3. 'sequential' method is simple but may compress one end")
    print("4. 'redistribute' method maintains overall range")
    print("5. Can combine spacing + smoothing for best results")
    print("=" * 70)
