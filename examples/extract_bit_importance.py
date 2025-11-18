"""
Example: Extract bit importance from LUTLayer mapping and truth tables

This shows how to:
1. Get importance of each thermometer bit based on LUT connections
2. Analyze which features/thresholds are most important
3. Use this information for pruning, analysis, or gradient weighting
"""
import torch
import sys
sys.path.insert(0, 'src')

from torch_dwn.encoder_layer import EncoderLayer, compute_bit_importance_from_lut
from torch_dwn import LUTLayer
import matplotlib.pyplot as plt

# ============================================================================
# 1. Create a simple model (EncoderLayer + LUTLayer)
# ============================================================================

num_features = 16
thermometer_bits = 200  # bits per feature
total_bits = num_features * thermometer_bits  # 3200 total bits
hidden_size = 100

print("="*70)
print("Extracting Bit Importance from LUTLayer Mapping & Truth Tables")
print("="*70)

# Create dummy training data
x_train = torch.randn(100, num_features) * 0.5

# Build encoder + LUT
encoder = EncoderLayer(
    inputs=num_features,
    output_size=thermometer_bits,
    input_dataset=x_train,
    estimator_type='finite_difference'
)

lut_layer = LUTLayer(
    input_size=total_bits,
    output_size=hidden_size,
    n=6,  # 6-input LUTs
    mapping='random'
)

print(f"\nModel structure:")
print(f"  - EncoderLayer: {num_features} features × {thermometer_bits} bits = {total_bits} total bits")
print(f"  - LUTLayer: {total_bits} inputs → {hidden_size} LUTs (6-input)")
print(f"  - Mapping shape: {lut_layer.mapping.shape}")
print(f"  - LUTs shape: {lut_layer.luts.shape}")

# ============================================================================
# 2. Extract bit importance using different methods
# ============================================================================

print("\n" + "="*70)
print("Method 1: Count - How many times each bit appears in connections")
print("="*70)

importance_count = compute_bit_importance_from_lut(lut_layer, method='count')
print(f"Importance shape: {importance_count.shape}")  # [3200]
print(f"Importance range: [{importance_count.min():.2f}, {importance_count.max():.2f}]")
print(f"Bits with 0 connections: {(importance_count == 0).sum().item()}")
print(f"Mean connections per bit: {importance_count.mean().item():.2f}")

# Find most/least connected bits
top_bits = importance_count.topk(10)
print(f"\nTop 10 most connected bits:")
for i, (idx, count) in enumerate(zip(top_bits.indices, top_bits.values)):
    feature = idx.item() // thermometer_bits
    threshold = idx.item() % thermometer_bits
    print(f"  {i+1}. Bit {idx.item():4d} (feature {feature:2d}, threshold {threshold:3d}): {count.item():.0f} connections")

print("\n" + "="*70)
print("Method 2: Weighted - Weight by LUT truth table magnitudes")
print("="*70)

importance_weighted = compute_bit_importance_from_lut(lut_layer, method='weighted')
print(f"Importance range: [{importance_weighted.min():.4f}, {importance_weighted.max():.4f}]")

top_bits_weighted = importance_weighted.topk(10)
print(f"\nTop 10 most important bits (by LUT weight):")
for i, (idx, weight) in enumerate(zip(top_bits_weighted.indices, top_bits_weighted.values)):
    feature = idx.item() // thermometer_bits
    threshold = idx.item() % thermometer_bits
    print(f"  {i+1}. Bit {idx.item():4d} (feature {feature:2d}, threshold {threshold:3d}): {weight.item():.4f}")

# ============================================================================
# 3. Reshape to analyze per-feature importance
# ============================================================================

print("\n" + "="*70)
print("Analyzing importance per feature")
print("="*70)

# Reshape from [total_bits] to [num_features, thermometer_bits]
importance_per_feature = importance_count.view(num_features, thermometer_bits)
print(f"Importance per feature shape: {importance_per_feature.shape}")

# Average importance per feature
feature_importance = importance_per_feature.mean(dim=1)
print(f"\nAverage importance per feature:")
for i, imp in enumerate(feature_importance):
    print(f"  Feature {i:2d}: {imp.item():.2f} avg connections")

# Find which thresholds are most important within each feature
print(f"\nMost important threshold for each feature:")
for i in range(num_features):
    max_thresh_idx = importance_per_feature[i].argmax()
    max_thresh_val = importance_per_feature[i].max()
    print(f"  Feature {i:2d}: Threshold {max_thresh_idx:3d} ({max_thresh_val:.0f} connections)")

# ============================================================================
# 4. Visualize importance
# ============================================================================

print("\n" + "="*70)
print("Visualizing importance...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram of connection counts
axes[0, 0].hist(importance_count.cpu().numpy(), bins=50, edgecolor='black')
axes[0, 0].set_xlabel('Number of connections')
axes[0, 0].set_ylabel('Number of bits')
axes[0, 0].set_title('Distribution of Bit Connections')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Importance per feature (heatmap)
im = axes[0, 1].imshow(importance_per_feature.cpu().numpy(), aspect='auto', cmap='viridis')
axes[0, 1].set_xlabel('Threshold index')
axes[0, 1].set_ylabel('Feature index')
axes[0, 1].set_title('Bit Importance per Feature')
plt.colorbar(im, ax=axes[0, 1], label='Connections')

# Plot 3: Average importance per feature (bar plot)
axes[1, 0].bar(range(num_features), feature_importance.cpu().numpy(), edgecolor='black')
axes[1, 0].set_xlabel('Feature index')
axes[1, 0].set_ylabel('Average connections')
axes[1, 0].set_title('Average Importance per Feature')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Comparison of methods
axes[1, 1].scatter(importance_count.detach().cpu().numpy(), importance_weighted.detach().cpu().numpy(), alpha=0.3, s=10)
axes[1, 1].set_xlabel('Count method')
axes[1, 1].set_ylabel('Weighted method')
axes[1, 1].set_title('Comparison: Count vs Weighted')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bit_importance_analysis.png', dpi=150)
print("Saved visualization to: bit_importance_analysis.png")

# ============================================================================
# 5. Example: Using importance for gradient weighting (optional)
# ============================================================================

print("\n" + "="*70)
print("Example usage: Gradient weighting during training")
print("="*70)

print("""
# During training, you can use importance to weight threshold gradients:

# After backward pass:
if encoder.thresholds.grad is not None:
    importance_weights = importance_per_feature.to(encoder.thresholds.device)
    encoder.thresholds.grad *= importance_weights

# This focuses gradient updates on thresholds that produce important bits!
""")

print("\n" + "="*70)
print("Done! You now know how to extract bit importance from LUTLayer.")
print("="*70)
