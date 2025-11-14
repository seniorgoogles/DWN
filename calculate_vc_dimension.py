"""
Calculate VC dimension (number of parameters) for jsc.py and jsc_default.py
"""

print("="*70)
print("VC Dimension Calculation (Number of Parameters)")
print("="*70)

# Input size (16 features × 200 bits)
input_size = 16 * 200  # 3200

# Number of output classes
num_classes = 5

print(f"\nInput size: {input_size}")
print(f"Number of classes: {num_classes}")

print("\n" + "="*70)
print("jsc_default.py (Linear layers)")
print("="*70)

# jsc_default.py uses Linear layers
hidden_size_default = 320

# Layer 1: Linear(3200 → 320)
params_layer1 = input_size * hidden_size_default + hidden_size_default
print(f"Layer 1: Linear({input_size} → {hidden_size_default})")
print(f"  Weights: {input_size} × {hidden_size_default} = {input_size * hidden_size_default:,}")
print(f"  Biases:  {hidden_size_default}")
print(f"  Total:   {params_layer1:,}")

# Layer 2: Linear(320 → 5)
params_layer2 = hidden_size_default * num_classes + num_classes
print(f"\nLayer 2: Linear({hidden_size_default} → {num_classes})")
print(f"  Weights: {hidden_size_default} × {num_classes} = {hidden_size_default * num_classes:,}")
print(f"  Biases:  {num_classes}")
print(f"  Total:   {params_layer2:,}")

total_params_default = params_layer1 + params_layer2
print(f"\nTotal parameters (VC dimension): {total_params_default:,}")

print("\n" + "="*70)
print("jsc.py (LUTLayer)")
print("="*70)

# jsc.py uses LUTLayer
hidden_size_lut = 10  # Current value
n = 6  # LUT input size

# LUTLayer parameters = output_size × 2^n
params_lut = hidden_size_lut * (2 ** n)
print(f"LUTLayer({input_size} → {hidden_size_lut}, n={n})")
print(f"  Number of LUTs: {hidden_size_lut}")
print(f"  Entries per LUT: 2^{n} = {2**n}")
print(f"  Total parameters: {hidden_size_lut} × {2**n} = {params_lut:,}")

print(f"\nGroupSum: 0 parameters")

total_params_lut = params_lut
print(f"\nTotal parameters (VC dimension): {total_params_lut:,}")

print("\n" + "="*70)
print("Comparison")
print("="*70)

ratio = total_params_default / total_params_lut
print(f"jsc_default.py: {total_params_default:,} parameters")
print(f"jsc.py:         {total_params_lut:,} parameters")
print(f"Ratio:          {ratio:.1f}x")

print("\n" + "="*70)
print("To Match VC Dimensions")
print("="*70)

# Calculate required hidden_size for jsc.py to match jsc_default.py
required_hidden_size = total_params_default // (2 ** n)
print(f"\nFor jsc.py to match jsc_default.py's VC dimension:")
print(f"  Required hidden_size = {total_params_default:,} / {2**n} = {required_hidden_size:,}")
print(f"  (Currently using hidden_size = {hidden_size_lut})")

# Verify
verify_params = required_hidden_size * (2 ** n)
print(f"\nVerification:")
print(f"  LUTLayer params = {required_hidden_size:,} × {2**n} = {verify_params:,}")
print(f"  Target params   = {total_params_default:,}")
print(f"  Difference      = {abs(verify_params - total_params_default):,}")

print("\n" + "="*70)
print("Recommendation")
print("="*70)
print(f"Change line 69 in examples/jsc.py to:")
print(f"  hidden_size = {required_hidden_size}")
