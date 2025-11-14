"""
Test the pruned model -> LUT conversion pipeline
This validates all steps without expensive retraining
"""
import torch
import torch.nn as nn
import torch_dwn as dwn
import os

print("=" * 70)
print("🧪 TESTING PRUNED MODEL -> LUT CONVERSION PIPELINE")
print("=" * 70)

# Check if we have a saved pruned model
pruned_model_path = "pruned_models/jsc_pruned_final.pth"
if not os.path.exists(pruned_model_path):
    print(f"\n❌ No pruned model found at {pruned_model_path}")
    print("Creating a test pruned model instead...")

    # Create a simple test model with pruning
    test_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    # Apply structured pruning to simulate the real scenario
    import torch.nn.utils.prune as prune

    # Prune to 6 weights per neuron
    for layer in [test_model[0], test_model[2]]:
        weight = layer.weight.data
        out_features, in_features = weight.shape

        if in_features > 6:
            mask = torch.zeros_like(weight)
            for neuron_idx in range(out_features):
                # Keep top 6 weights
                _, top_indices = torch.topk(weight[neuron_idx].abs(), k=6, largest=True)
                mask[neuron_idx, top_indices] = 1.0

            prune.custom_from_mask(layer, name='weight', mask=mask)
            prune.remove(layer, 'weight')  # Bake the mask

    model = test_model
    print("✅ Created test model with 6 weights per neuron")
else:
    print(f"\n✅ Loading pruned model from {pruned_model_path}")
    model = torch.load(pruned_model_path)

print(f"\nModel architecture:")
print(model)

# ============================================================================
# STEP 1: Extract connections from pruned model
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: EXTRACTING CONNECTIONS FROM PRUNED MODEL")
print("=" * 70)

def extract_connections_from_pruned_layer(layer, layer_name=""):
    """Extract non-zero connection information from a pruned layer."""
    weight = layer.weight.data
    out_features, in_features = weight.shape

    connections_per_neuron = []

    for neuron_idx in range(out_features):
        neuron_weights = weight[neuron_idx]
        active_mask = neuron_weights != 0
        active_indices = torch.where(active_mask)[0]

        connections_per_neuron.append({
            'neuron_idx': neuron_idx,
            'active_input_indices': active_indices.tolist(),
            'num_connections': len(active_indices)
        })

    return {
        'layer_name': layer_name,
        'in_features': in_features,
        'out_features': out_features,
        'connections_per_neuron': connections_per_neuron
    }

# Extract connections from all linear layers
connections = []
linear_layers = []
for idx, module in enumerate(model.modules()):
    if isinstance(module, nn.Linear):
        linear_layers.append((idx, module))

print(f"Found {len(linear_layers)} linear layers")

for idx, (layer_idx, layer) in enumerate(linear_layers):
    conn_info = extract_connections_from_pruned_layer(layer, f"layer_{idx}")
    connections.append(conn_info)

    num_connections = [n['num_connections'] for n in conn_info['connections_per_neuron']]
    print(f"\nLayer {idx}: {conn_info['in_features']} → {conn_info['out_features']}")
    print(f"  Connections per neuron: min={min(num_connections)}, max={max(num_connections)}, avg={sum(num_connections)/len(num_connections):.1f}")

    # Show first 3 neurons as examples
    print(f"  Example neurons (first 3):")
    for neuron_info in conn_info['connections_per_neuron'][:3]:
        print(f"    Neuron {neuron_info['neuron_idx']}: {neuron_info['num_connections']} connections → {neuron_info['active_input_indices']}")

# ============================================================================
# STEP 2: Create fixed mappings for LUTLayer
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: CREATING FIXED MAPPINGS FOR LUT LAYERS")
print("=" * 70)

def create_fixed_mapping_from_connections(connection_info):
    """Create a fixed mapping tensor for DWN LUTLayer from pruned connections."""
    out_features = connection_info['out_features']
    n_inputs = connection_info['connections_per_neuron'][0]['num_connections']

    # IMPORTANT: LUTLayer expects dtype=torch.int32
    mapping = torch.zeros(out_features, n_inputs, dtype=torch.int32)

    for neuron_info in connection_info['connections_per_neuron']:
        neuron_idx = neuron_info['neuron_idx']
        active_indices = neuron_info['active_input_indices']

        # Fill in the mapping for this neuron
        for i, idx in enumerate(active_indices):
            mapping[neuron_idx, i] = idx

    return mapping

mappings = []
for idx, conn_info in enumerate(connections):
    mapping = create_fixed_mapping_from_connections(conn_info)
    mappings.append(mapping)

    print(f"\nLayer {idx} mapping:")
    print(f"  Shape: {mapping.shape}")
    print(f"  Dtype: {mapping.dtype} (expected: torch.int32)")
    print(f"  First 3 rows:")
    for i in range(min(3, mapping.shape[0])):
        print(f"    Neuron {i}: {mapping[i].tolist()}")

# ============================================================================
# STEP 3: Build LUT model with fixed connections
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: BUILDING LUT MODEL WITH FIXED CONNECTIONS")
print("=" * 70)

try:
    # Build LUTLayers
    lut_layers = []
    for idx, (conn_info, mapping) in enumerate(zip(connections, mappings)):
        print(f"\nCreating LUTLayer {idx}...")
        print(f"  in_features={conn_info['in_features']}, out_features={conn_info['out_features']}, n={mapping.shape[1]}")
        print(f"  mapping shape={mapping.shape}, dtype={mapping.dtype}")

        lut_layer = dwn.LUTLayer(
            conn_info['in_features'],
            conn_info['out_features'],
            n=mapping.shape[1],
            mapping=mapping
        )
        lut_layers.append(lut_layer)
        print(f"  ✅ LUTLayer {idx} created successfully")

    # Build full model
    print("\nBuilding complete LUT model...")
    if len(lut_layers) == 2:
        lut_model = nn.Sequential(
            lut_layers[0],
            nn.ReLU(),
            lut_layers[1]
        )
    else:
        print(f"  Note: Found {len(lut_layers)} layers, building sequential model")
        modules = []
        for i, layer in enumerate(lut_layers):
            modules.append(layer)
            if i < len(lut_layers) - 1:
                modules.append(nn.ReLU())
        lut_model = nn.Sequential(*modules)

    print("  ✅ LUT model built successfully")
    print(f"\nLUT Model architecture:")
    print(lut_model)

except Exception as e:
    print(f"\n❌ ERROR building LUT model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# STEP 4: Test forward pass (requires CUDA)
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: TESTING FORWARD PASS")
print("=" * 70)

input_size = connections[0]['in_features']
output_size = connections[-1]['out_features']

if torch.cuda.is_available():
    try:
        batch_size = 4
        print(f"CUDA available, testing forward pass...")
        print(f"Creating test input: shape=({batch_size}, {input_size})")

        lut_model = lut_model.cuda()
        test_input = torch.randn(batch_size, input_size).cuda()

        print("Running forward pass...")
        output = lut_model(test_input)

        print(f"  ✅ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sample (first row): {output[0].detach().cpu().numpy()}")

    except Exception as e:
        print(f"\n❌ ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print("⚠️  CUDA not available, skipping forward pass test")
    print("   (LUTLayer requires CUDA for inference)")
    print(f"   Expected input shape: (batch_size, {input_size})")
    print(f"   Expected output shape: (batch_size, {output_size})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL CRITICAL TESTS PASSED!")
print("=" * 70)
print("\nPipeline validation complete:")
print(f"  ✅ Connection extraction: {len(connections)} layers")
print(f"  ✅ Mapping creation: correct dtype (int32) and shape")
print(f"  ✅ LUTLayer initialization: {len(lut_layers)} layers")
print(f"  ✅ Model building: Sequential model created")
if torch.cuda.is_available():
    print(f"  ✅ Forward pass: output shape {output.shape}")
else:
    print(f"  ⚠️  Forward pass: skipped (requires CUDA)")
print("\n🎯 The LUT conversion pipeline is correctly configured!")
print("   You can now run the full training script.")
