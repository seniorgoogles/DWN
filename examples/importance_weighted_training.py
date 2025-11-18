"""
Example: Training with Importance-Weighted Finite Difference Estimator

This demonstrates how to use bit importance from LUTLayer mapping to focus
threshold learning on the bits that actually matter.
"""
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import sys
sys.path.insert(0, 'src')

from torch_dwn.encoder_layer import EncoderLayer
from torch_dwn import LUTLayer, GroupSum

print("="*70)
print("Importance-Weighted Threshold Training")
print("="*70)

# ============================================================================
# 1. Setup: Create model and dummy data
# ============================================================================

num_features = 16
thermometer_bits = 50
total_bits = num_features * thermometer_bits
hidden_size = 64
num_classes = 10
batch_size = 32

# Create dummy classification data
x_train = torch.randn(200, num_features) * 0.5
y_train = torch.randint(0, num_classes, (200,))
x_test = torch.randn(50, num_features) * 0.5
y_test = torch.randint(0, num_classes, (50,))

print(f"\nDataset:")
print(f"  - Training: {x_train.shape[0]} samples")
print(f"  - Test: {x_test.shape[0]} samples")
print(f"  - Features: {num_features}")
print(f"  - Classes: {num_classes}")

# ============================================================================
# 2. Build model: EncoderLayer + LUTLayer + GroupSum
# ============================================================================

encoder = EncoderLayer(
    inputs=num_features,
    output_size=thermometer_bits,
    input_dataset=x_train,
    estimator_type='finite_difference'
)

lut_layer = LUTLayer(
    input_size=total_bits,
    output_size=hidden_size,
    n=6,
    mapping='random'
)

model = nn.Sequential(
    encoder,
    nn.Flatten(start_dim=1),
    lut_layer,
    GroupSum(k=num_classes, tau=1/0.3)
)

if torch.cuda.is_available():
    model = model.cuda()
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_test = x_test.cuda()
    y_test = y_test.cuda()

print(f"\nModel architecture:")
print(f"  EncoderLayer: {num_features} features → {total_bits} bits")
print(f"  LUTLayer: {total_bits} inputs → {hidden_size} LUTs (6-input)")
print(f"  GroupSum: {hidden_size} → {num_classes} classes")

# ============================================================================
# 3. Phase 1: Train WITHOUT importance weighting
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: Training WITHOUT importance weighting (10 epochs)")
print("="*70)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for i in range(0, len(x_train), batch_size):
        optimizer.zero_grad()
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_x)
        loss = cross_entropy(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean()

    print(f"Epoch {epoch+1:2d}/10 | Loss: {total_loss:.4f} | Test Acc: {test_acc:.4f}")

print("\n✓ Phase 1 complete")

# ============================================================================
# 4. Extract importance from trained LUTLayer
# ============================================================================

print("\n" + "="*70)
print("Extracting bit importance from LUTLayer mapping...")
print("="*70)

# Set importance weights in encoder based on LUT connections
encoder.set_importance_from_lut(lut_layer, method='weighted')

# Check which thresholds are important
importance = encoder.importance_weights
print(f"\nImportance statistics:")
print(f"  - Mean: {importance.mean():.4f}")
print(f"  - Max: {importance.max():.4f}")
print(f"  - Min: {importance.min():.4f}")
print(f"  - Zero importance: {(importance == 0).sum().item()} / {importance.numel()}")

# Find most important thresholds
importance_flat = importance.view(-1)
top_k = 10
top_indices = importance_flat.topk(top_k).indices
print(f"\nTop {top_k} most important thresholds:")
for i, idx in enumerate(top_indices):
    feature = idx.item() // thermometer_bits
    threshold = idx.item() % thermometer_bits
    weight = importance_flat[idx].item()
    print(f"  {i+1}. Feature {feature:2d}, Threshold {threshold:3d}: {weight:.4f}")

# ============================================================================
# 5. Phase 2: Continue training WITH importance weighting
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: Training WITH importance weighting (10 more epochs)")
print("="*70)
print("Now threshold gradients are weighted by bit importance!")
print("Important thresholds will update faster than unimportant ones.\n")

# Reset optimizer for fair comparison
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    total_loss = 0
    for i in range(0, len(x_train), batch_size):
        optimizer.zero_grad()
        batch_x = x_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        outputs = model(batch_x)
        loss = cross_entropy(outputs, batch_y)
        loss.backward()

        # Importance weighting happens automatically in backward pass!
        # No manual gradient modification needed

        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean()

    print(f"Epoch {epoch+1:2d}/10 | Loss: {total_loss:.4f} | Test Acc: {test_acc:.4f}")

print("\n✓ Phase 2 complete")

# ============================================================================
# 6. Summary
# ============================================================================

print("\n" + "="*70)
print("Summary: How Importance Weighting Works")
print("="*70)
print("""
1. Train model with EncoderLayer + LUTLayer
2. Extract bit importance from LUT mapping: encoder.set_importance_from_lut(lut_layer)
3. Continue training - gradients automatically weighted by importance!

Key insight:
- Thresholds producing unused bits get 0 gradient → no wasted updates
- Thresholds producing important bits get full gradient → faster convergence
- Gradient updates focused where they matter most

The ImportanceWeightedFiniteDifferenceEstimator handles this automatically!
""")

print("="*70)
print("Done!")
print("="*70)
