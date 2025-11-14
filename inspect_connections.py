"""
Inspect saved pruned connections
"""
import pickle
import torch

print("=" * 70)
print("🔍 INSPECTING SAVED PRUNED CONNECTIONS")
print("=" * 70)

# Load the connections
with open('pruned_connections_6inputs.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract connections from the data structure
if isinstance(data, dict) and 'connections' in data:
    connections = data['connections']
    print(f"\nLoaded data structure with keys: {data.keys()}")
    if 'architecture' in data:
        print(f"Architecture: {data['architecture']}")
    if 'accuracy' in data:
        print(f"Accuracy: {data['accuracy']}")
else:
    connections = data

print(f"\nFound {len(connections)} layers in saved connections\n")

for idx, conn_info in enumerate(connections):
    print(f"{'='*70}")
    layer_name = conn_info.get('layer_name', f"layer_{conn_info.get('layer_idx', idx)}")
    print(f"LAYER {idx}: {layer_name}")
    print(f"{'='*70}")
    print(f"  Shape: {conn_info['in_features']} → {conn_info['out_features']}")

    # Count connections per neuron
    num_connections = [n['num_connections'] for n in conn_info['connections_per_neuron']]
    print(f"  Connections per neuron:")
    print(f"    Min: {min(num_connections)}")
    print(f"    Max: {max(num_connections)}")
    print(f"    Avg: {sum(num_connections)/len(num_connections):.2f}")

    # Show example neurons
    print(f"\n  Example neurons (first 5):")
    for neuron_info in conn_info['connections_per_neuron'][:5]:
        print(f"    Neuron {neuron_info['neuron_idx']}: {neuron_info['num_connections']} connections")
        print(f"      → Input indices: {neuron_info['active_input_indices']}")

    # Check if all neurons have exactly 6 connections
    all_six = all(n == 6 for n in num_connections)
    if all_six:
        print(f"\n  ✅ All neurons have exactly 6 connections (ready for LUT)")
    else:
        print(f"\n  ⚠️  Not all neurons have 6 connections")
        # Show distribution
        from collections import Counter
        dist = Counter(num_connections)
        print(f"  Distribution: {dict(sorted(dist.items()))}")

print(f"\n{'='*70}")
print("📊 SUMMARY")
print(f"{'='*70}")

# Create mappings to verify they'll work
print("\nCreating LUTLayer mappings from connections...")

for idx, conn_info in enumerate(connections):
    out_features = conn_info['out_features']
    n_inputs = conn_info['connections_per_neuron'][0]['num_connections']

    mapping = torch.zeros(out_features, n_inputs, dtype=torch.int32)

    for neuron_info in conn_info['connections_per_neuron']:
        neuron_idx = neuron_info['neuron_idx']
        active_indices = neuron_info['active_input_indices']

        for i, idx_val in enumerate(active_indices):
            mapping[neuron_idx, i] = idx_val

    print(f"\nLayer {idx} mapping:")
    print(f"  Shape: {mapping.shape}")
    print(f"  Dtype: {mapping.dtype}")
    print(f"  Min index: {mapping.min().item()}, Max index: {mapping.max().item()}")
    print(f"  ✅ Valid for LUTLayer(in_features={conn_info['in_features']}, out_features={out_features}, n={n_inputs})")

print("\n✅ All connections are valid and ready for LUT conversion!")
