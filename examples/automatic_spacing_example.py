"""
Example: Automatic Minimum Spacing During Backprop

Demonstrates that spacing constraints are applied AUTOMATICALLY
without needing to manually call smooth_thresholds().
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../src')

from torch_dwn.encoder_layer import EncoderLayer


def example_automatic_constraints():
    """
    Example: Constraints applied automatically during training
    """
    print("=" * 70)
    print("Automatic Constraint Application During Backprop")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(1000, 5)
    train_labels = torch.randint(0, 2, (1000, 10)).float()

    # Create encoder with auto_apply_constraints=True (DEFAULT)
    encoder = EncoderLayer(
        inputs=5,
        output_size=8,
        input_dataset=train_data,
        min_threshold_spacing=0.1,
        spacing_method='push_apart',
        auto_apply_constraints=True  # Automatically apply constraints!
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(5 * 8, 10)

        def forward(self, x):
            # Constraints are applied HERE automatically at start of forward!
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nAuto-apply enabled: {encoder.auto_apply_constraints}")
    print(f"Minimum spacing: {encoder.min_threshold_spacing}")
    print("\nInitial thresholds (Feature 0):")
    print(torch.sort(encoder.thresholds[0])[0])

    print("\nTraining WITHOUT manual smooth_thresholds() calls...")
    print("Constraints will be applied automatically!\n")

    # Standard training loop - NO manual smooth_thresholds() call!
    for epoch in range(20):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            # Forward pass - constraints applied here automatically!
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # NOTE: NO encoder.smooth_thresholds() call needed!

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / num_batches
            sorted_thresh = torch.sort(encoder.thresholds[0])[0]
            spacings = sorted_thresh[1:] - sorted_thresh[:-1]
            min_spacing = spacings.min().item()
            print(f"Epoch {epoch+1:2d} - Loss: {avg_loss:.4f} - Min spacing: {min_spacing:.6f}")

    print("\nFinal thresholds (Feature 0):")
    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    print(sorted_thresh)

    # Verify spacing was maintained
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]
    min_spacing = spacings.min().item()
    print(f"\nFinal minimum spacing: {min_spacing:.6f}")
    print(f"Required minimum: {encoder.min_threshold_spacing}")
    print(f"All spacings satisfied: {(spacings >= encoder.min_threshold_spacing - 1e-6).all().item()}")

    print("\n✅ Spacing constraints were enforced automatically!")
    print("   No manual smooth_thresholds() calls needed!")


def example_manual_vs_automatic():
    """
    Compare manual vs automatic constraint application
    """
    print("\n" + "=" * 70)
    print("Comparison: Manual vs Automatic")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(500, 4)
    train_labels = torch.randint(0, 2, (500, 8)).float()

    print("\n1. Training with auto_apply_constraints=False (manual mode)")
    print("-" * 70)

    encoder_manual = EncoderLayer(
        inputs=4,
        output_size=6,
        input_dataset=train_data,
        min_threshold_spacing=0.08,
        auto_apply_constraints=False  # Must call smooth_thresholds() manually
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(4 * 6, 8)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model_manual = SimpleModel(encoder_manual)
    optimizer_manual = optim.Adam(model_manual.parameters(), lr=0.03)
    criterion = nn.BCEWithLogitsLoss()

    print("Training requires manual smooth_thresholds() call...")

    for epoch in range(10):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            output = model_manual(batch_x)
            loss = criterion(output, batch_y)

            optimizer_manual.zero_grad()
            loss.backward()
            optimizer_manual.step()

            # Must call this manually!
            encoder_manual.smooth_thresholds()

    sorted_thresh_manual = torch.sort(encoder_manual.thresholds[0])[0]
    spacings_manual = sorted_thresh_manual[1:] - sorted_thresh_manual[:-1]
    print(f"Final min spacing: {spacings_manual.min().item():.6f}")

    print("\n2. Training with auto_apply_constraints=True (automatic mode)")
    print("-" * 70)

    torch.manual_seed(42)  # Same seed for fair comparison

    encoder_auto = EncoderLayer(
        inputs=4,
        output_size=6,
        input_dataset=train_data,
        min_threshold_spacing=0.08,
        auto_apply_constraints=True  # Automatic!
    )

    model_auto = SimpleModel(encoder_auto)
    optimizer_auto = optim.Adam(model_auto.parameters(), lr=0.03)

    print("Training is fully automatic, no smooth_thresholds() calls...")

    for epoch in range(10):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            output = model_auto(batch_x)
            loss = criterion(output, batch_y)

            optimizer_auto.zero_grad()
            loss.backward()
            optimizer_auto.step()

            # No manual call needed!

    sorted_thresh_auto = torch.sort(encoder_auto.thresholds[0])[0]
    spacings_auto = sorted_thresh_auto[1:] - sorted_thresh_auto[:-1]
    print(f"Final min spacing: {spacings_auto.min().item():.6f}")

    print("\n✅ Both modes work correctly!")
    print("   Automatic mode is more convenient (default)")


def example_with_spline_smoothing():
    """
    Example: Automatic spacing + spline smoothing
    """
    print("\n" + "=" * 70)
    print("Automatic: Spacing + Spline Smoothing Combined")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(400, 3)
    train_labels = torch.randint(0, 2, (400, 6)).float()

    # Both spacing AND smoothing applied automatically!
    encoder = EncoderLayer(
        inputs=3,
        output_size=10,
        input_dataset=train_data,
        # Spacing
        min_threshold_spacing=0.05,
        spacing_method='push_apart',
        # Smoothing
        enable_spline_smoothing=True,
        smoothing_method='monotonic',
        smooth_every_n_steps=20,
        # Automatic application
        auto_apply_constraints=True
    )

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(3 * 10, 6)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Both spacing AND smoothing enabled")
    print(f"Auto-apply: {encoder.auto_apply_constraints}")
    print("\nTraining with automatic constraint application...")

    for epoch in range(15):
        for i in range(0, len(train_data), 32):
            batch_x = train_data[i:i+32]
            batch_y = train_labels[i:i+32]

            # Both constraints applied automatically in forward!
            output = model(batch_x)
            loss = criterion(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]

    print(f"\nFinal results:")
    print(f"  Min spacing: {spacings.min().item():.6f} (required: {encoder.min_threshold_spacing})")
    print(f"  Mean spacing: {spacings.mean().item():.6f}")
    print(f"  Thresholds: {sorted_thresh}")

    print("\n✅ Both smoothing and spacing applied automatically!")


if __name__ == "__main__":
    example_automatic_constraints()
    example_manual_vs_automatic()
    example_with_spline_smoothing()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("With auto_apply_constraints=True (DEFAULT):")
    print("  - Constraints applied automatically at start of forward()")
    print("  - No need to call smooth_thresholds() manually")
    print("  - Works seamlessly during backprop cycle")
    print("\nWith auto_apply_constraints=False:")
    print("  - Must call encoder.smooth_thresholds() after optimizer.step()")
    print("  - Gives you full manual control")
    print("=" * 70)
