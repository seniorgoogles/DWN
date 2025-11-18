"""
Test: Verify spacing constraints are applied during backpropagation cycle
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('../src')

from torch_dwn.encoder_layer import EncoderLayer


def test_backprop_integration():
    """
    Verify that constraints are applied automatically during backprop
    """
    print("=" * 70)
    print("Testing Backprop Integration")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(100, 3)
    train_labels = torch.randint(0, 2, (100, 6)).float()

    # Create encoder with automatic constraint application
    encoder = EncoderLayer(
        inputs=3,
        output_size=6,
        input_dataset=train_data,
        min_threshold_spacing=0.15,
        spacing_method='push_apart',
        auto_apply_constraints=True  # Enabled by default
    )

    # Verify hook is registered
    print(f"Auto-apply enabled: {encoder.auto_apply_constraints}")
    print(f"Backward hooks registered: {len(encoder._backward_hooks) > 0}")

    class SimpleModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.fc = nn.Linear(3 * 6, 6)

        def forward(self, x):
            encoded = self.encoder(x)
            return self.fc(encoded.flatten(1))

    model = SimpleModel(encoder)
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # High LR to test
    criterion = nn.BCEWithLogitsLoss()

    print("\nInitial thresholds (Feature 0):")
    initial_thresh = torch.sort(encoder.thresholds[0])[0].clone()
    print(initial_thresh)

    print("\nRunning training iterations and tracking constraint application...")
    print("-" * 70)

    for i in range(5):
        batch_x = train_data[:32]
        batch_y = train_labels[:32]

        print(f"\nIteration {i+1}:")

        # Check flag before forward
        has_flag_before = hasattr(encoder, '_needs_constraint_application')
        flag_value_before = getattr(encoder, '_needs_constraint_application', None)
        print(f"  Before forward: flag exists={has_flag_before}, value={flag_value_before}")

        # Forward pass
        output = model(batch_x)
        loss = criterion(output, batch_y)
        print(f"  After forward: Loss={loss.item():.4f}")

        # Backward pass - this triggers the backward hook!
        optimizer.zero_grad()
        loss.backward()

        # Check flag after backward
        flag_after_backward = getattr(encoder, '_needs_constraint_application', None)
        print(f"  After backward: flag={flag_after_backward} (hook should set this to True)")

        # Optimizer step
        optimizer.step()
        print(f"  After optimizer.step(): Parameters updated")

        # Check spacing after optimizer.step() (before next forward)
        sorted_thresh = torch.sort(encoder.thresholds[0])[0]
        spacings = sorted_thresh[1:] - sorted_thresh[:-1]
        min_spacing = spacings.min().item()
        print(f"  Min spacing: {min_spacing:.6f} (required: {encoder.min_threshold_spacing})")

        # Verify spacing is NOT yet enforced (happens in next forward)
        if min_spacing < encoder.min_threshold_spacing - 1e-6:
            print(f"  ⚠️  Spacing violation detected (will be fixed in next forward)")
        else:
            print(f"  ✅ Spacing already satisfied")

    print("\n" + "-" * 70)
    print("\nFinal thresholds (Feature 0):")
    final_thresh = torch.sort(encoder.thresholds[0])[0]
    print(final_thresh)

    # Verify final spacing
    spacings = final_thresh[1:] - final_thresh[:-1]
    min_spacing = spacings.min().item()
    all_satisfied = (spacings >= encoder.min_threshold_spacing - 1e-6).all().item()

    print(f"\nFinal verification:")
    print(f"  Minimum spacing: {min_spacing:.6f}")
    print(f"  Required spacing: {encoder.min_threshold_spacing}")
    print(f"  All constraints satisfied: {all_satisfied}")

    if all_satisfied:
        print("\n✅ SUCCESS: Constraints enforced during backprop cycle!")
    else:
        print("\n❌ FAILED: Some constraints not satisfied")

    return all_satisfied


def test_hook_triggering():
    """
    Detailed test of hook triggering mechanism
    """
    print("\n" + "=" * 70)
    print("Testing Backward Hook Mechanism")
    print("=" * 70)

    torch.manual_seed(42)
    train_data = torch.randn(50, 2)

    encoder = EncoderLayer(
        inputs=2,
        output_size=4,
        input_dataset=train_data,
        min_threshold_spacing=0.2,
        auto_apply_constraints=True
    )

    # Manually corrupt thresholds to create spacing violations
    print("\nManually creating spacing violations...")
    with torch.no_grad():
        encoder.thresholds[0] = torch.tensor([0.0, 0.05, 0.06, 0.07])  # Violations!

    print(f"Corrupted thresholds: {encoder.thresholds[0]}")
    spacings = encoder.thresholds[0][1:] - encoder.thresholds[0][:-1]
    print(f"Spacings: {spacings}")
    print(f"Min spacing: {spacings.min().item():.6f} (required: {encoder.min_threshold_spacing})")

    # Do a forward-backward cycle
    print("\nRunning forward-backward cycle...")

    x = train_data[:16]

    # Forward
    encoder.train()  # Ensure training mode
    output = encoder(x)

    print(f"After forward, flag = {getattr(encoder, '_needs_constraint_application', 'NOT SET')}")

    # Create a dummy loss and backward
    loss = output.sum()
    loss.backward()

    print(f"After backward, flag = {getattr(encoder, '_needs_constraint_application', 'NOT SET')}")
    print("  (Backward hook should have set flag to True)")

    # Next forward should apply constraints
    print("\nRunning next forward (should apply constraints)...")
    output2 = encoder(train_data[:16])

    print(f"After second forward, flag = {getattr(encoder, '_needs_constraint_application', 'NOT SET')}")

    # Check if spacing is now fixed
    sorted_thresh = torch.sort(encoder.thresholds[0])[0]
    spacings = sorted_thresh[1:] - sorted_thresh[:-1]
    min_spacing = spacings.min().item()

    print(f"\nCorrected thresholds: {sorted_thresh}")
    print(f"Corrected spacings: {spacings}")
    print(f"Min spacing: {min_spacing:.6f} (required: {encoder.min_threshold_spacing})")

    if (spacings >= encoder.min_threshold_spacing - 1e-6).all():
        print("\n✅ SUCCESS: Backward hook triggered constraint application!")
        return True
    else:
        print("\n❌ FAILED: Constraints not applied")
        return False


if __name__ == "__main__":
    test1_passed = test_backprop_integration()
    test2_passed = test_hook_triggering()

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Backprop Integration Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Hook Triggering Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Constraints are applied during backprop.")
    else:
        print("\n⚠️  Some tests failed.")
    print("=" * 70)
