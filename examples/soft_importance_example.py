"""
Example: Soft Importance Weighting

Shows how min_weight parameter affects threshold learning:
  - min_weight=0.0: Hard weighting (unused bits get 0 gradient)
  - min_weight=0.1: Soft weighting (unused bits get 10% gradient) ← RECOMMENDED
  - min_weight=0.5: Medium weighting (unused bits get 50% gradient)
  - min_weight=1.0: No weighting (all bits equal)
"""
import torch
import torch.nn as nn

# Your existing training code...
# model = nn.Sequential(encoder, nn.Flatten(), lut_layer, GroupSum(...))

# After initial training (e.g., epoch 15):

# ============================================================================
# Option 1: Soft Weighting (RECOMMENDED)
# ============================================================================
# Unused bits still learn at 10% rate, important bits at 100%
model[0].set_importance_from_lut(model[2], method='weighted', min_weight=0.1)

print("""
Soft weighting enabled (min_weight=0.1):
  - Important bits (weight=1.0): 100% gradient
  - Unused bits (weight=0.0): 10% gradient
  - All thresholds can still learn!
""")

# ============================================================================
# Option 2: Hard Weighting
# ============================================================================
# Unused bits get ZERO gradient (your current issue)
# model[0].set_importance_from_lut(model[2], method='weighted', min_weight=0.0)

# ============================================================================
# Option 3: Very Soft Weighting
# ============================================================================
# Unused bits learn at 50% rate (almost equal to important bits)
# model[0].set_importance_from_lut(model[2], method='weighted', min_weight=0.5)

# ============================================================================
# Option 4: No Weighting
# ============================================================================
# All bits treated equally (importance has no effect)
# model[0].set_importance_from_lut(model[2], method='weighted', min_weight=1.0)

# Continue training...
# for epoch in range(15, 30):
#     ...

print("\nComparison of min_weight settings:")
print("="*70)
print("min_weight | Unused Bit Gradient | Effect on Learning")
print("-"*70)
print("   0.0     |        0%          | Thresholds frozen (current issue!)")
print("   0.1     |       10%          | Slow learning for unused bits ✓")
print("   0.3     |       30%          | Moderate learning")
print("   0.5     |       50%          | Similar to important bits")
print("   1.0     |      100%          | No importance effect")
print("="*70)

print("\n💡 Recommendation: Use min_weight=0.1 to 0.3")
print("   - Allows all thresholds to adapt")
print("   - Focuses more gradient on important bits")
print("   - Balances efficiency and flexibility")
