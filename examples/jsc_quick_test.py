"""
Quick Test: Prune to 6 inputs and train DWN LUTLayer
This is a simplified version for testing the workflow.
"""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
import openml
import numpy as np
from sklearn.model_selection import train_test_split

print("="*70)
print("QUICK TEST: Pruning to 6 inputs + DWN LUTLayer")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load JSC dataset
print("\nLoading JSC dataset...")
dataset = openml.datasets.get_dataset(42468)
df_features, df_labels, _, _ = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

features = df_features.values.astype(np.float32)
label_names = list(df_labels.unique())
labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

x_train, x_test, y_train, y_test = train_test_split(
    features, labels, train_size=0.8, random_state=42
)

# Convert to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Binarize
print("Binarizing input...")
thermometer = dwn.DistributiveThermometer(200).fit(x_train)
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)
print(f"Binarized shape: {x_train.shape}")

# Build small model
hidden_size = 10
model = nn.Sequential(
    nn.Linear(x_train.size(1), hidden_size),
    nn.Linear(hidden_size, 5)
).cuda()

print(f"Model: {x_train.size(1)} → {hidden_size} → 5")

# Quick initial training (5 epochs)
print("\n" + "="*70)
print("STEP 1: Initial Training (5 epochs)")
print("="*70)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(5):
    model.train()
    perm = torch.randperm(x_train.shape[0])
    for i in range(0, x_train.shape[0], 128):
        optimizer.zero_grad()
        idx = perm[i:i+128]
        out = model(x_train[idx].cuda())
        loss = cross_entropy(out, y_train[idx].cuda())
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        pred = model(x_test.cuda()).argmax(dim=1).cpu()
        acc = (pred == y_test).float().mean()
    print(f"Epoch {epoch+1}/5: Test Acc = {acc:.4f}")

print(f"\n✓ Initial model trained")

# Prune to exactly 6 weights per neuron
print("\n" + "="*70)
print("STEP 2: Pruning to 6 weights per neuron")
print("="*70)

import torch.nn.utils.prune as prune

for layer_idx, module in enumerate(model.modules()):
    if isinstance(module, nn.Linear):
        print(f"\nLayer {layer_idx}: {module.weight.shape}")

        # For each neuron, keep only top 6 weights
        weight = module.weight.data
        mask = torch.zeros_like(weight)

        for neuron_idx in range(weight.shape[0]):
            _, top_indices = torch.topk(weight[neuron_idx].abs(), k=min(6, weight.shape[1]), largest=True)
            mask[neuron_idx, top_indices] = 1.0

        # Apply pruning mask
        prune.custom_from_mask(module, name='weight', mask=mask)

        sparsity = (mask == 0).sum().item() / mask.numel()
        print(f"  Sparsity: {sparsity*100:.1f}%")
        print(f"  Example neuron 0: {(mask[0] > 0).sum().item()} active weights")

# Quick retrain (5 epochs)
print("\n" + "="*70)
print("STEP 3: Retrain pruned model (5 epochs)")
print("="*70)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(5):
    model.train()
    perm = torch.randperm(x_train.shape[0])
    for i in range(0, x_train.shape[0], 128):
        optimizer.zero_grad()
        idx = perm[i:i+128]
        out = model(x_train[idx].cuda())
        loss = cross_entropy(out, y_train[idx].cuda())
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x_test.cuda()).argmax(dim=1).cpu()
        acc = (pred == y_test).float().mean()
    print(f"Epoch {epoch+1}/5: Test Acc = {acc:.4f}")

print(f"\n✓ Pruned model retrained")

# Export connections
print("\n" + "="*70)
print("STEP 4: Export pruned connections")
print("="*70)

connections = []
for layer_idx, module in enumerate(model.modules()):
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask.cpu().numpy()
        else:
            mask = (module.weight.detach().cpu().numpy() != 0).astype(float)

        layer_info = {
            'layer_idx': layer_idx,
            'in_features': module.weight.shape[1],
            'out_features': module.weight.shape[0],
            'connections_per_neuron': []
        }

        for neuron_idx in range(module.weight.shape[0]):
            active_indices = np.where(mask[neuron_idx] > 0)[0].tolist()
            layer_info['connections_per_neuron'].append({
                'neuron_idx': neuron_idx,
                'active_input_indices': active_indices,
                'num_connections': len(active_indices)
            })

        connections.append(layer_info)
        print(f"\nLayer {layer_idx}:")
        print(f"  Neuron 0: {len(layer_info['connections_per_neuron'][0]['active_input_indices'])} connections")
        print(f"  Indices: {layer_info['connections_per_neuron'][0]['active_input_indices']}")

print(f"\n✓ Connections exported")

# Build DWN LUTLayer model
print("\n" + "="*70)
print("STEP 5: Build DWN LUTLayer with fixed connections")
print("="*70)

def create_fixed_mapping(connection_info):
    out_features = connection_info['out_features']
    n_inputs = connection_info['connections_per_neuron'][0]['num_connections']

    mapping = torch.zeros(out_features, n_inputs, dtype=torch.int32)

    for neuron_info in connection_info['connections_per_neuron']:
        neuron_idx = neuron_info['neuron_idx']
        active_indices = neuron_info['active_input_indices']

        for i, idx in enumerate(active_indices[:n_inputs]):  # Take first n_inputs
            mapping[neuron_idx, i] = idx

    return mapping

mapping_layer1 = create_fixed_mapping(connections[0])
mapping_layer2 = create_fixed_mapping(connections[1])

print(f"Mapping Layer 1 shape: {mapping_layer1.shape}")
print(f"Mapping Layer 2 shape: {mapping_layer2.shape}")
print(f"Example: Neuron 0 connects to inputs {mapping_layer1[0].tolist()}")

# Create DWN LUTLayers with fixed mappings
lut_layer1 = dwn.LUTLayer(
    connections[0]['in_features'],  # in_features (positional)
    connections[0]['out_features'], # out_features (positional)
    n=mapping_layer1.shape[1],      # n (keyword)
    mapping=mapping_layer1           # Pass tensor directly
)

lut_layer2 = dwn.LUTLayer(
    connections[1]['in_features'],
    connections[1]['out_features'],
    n=mapping_layer2.shape[1],
    mapping=mapping_layer2           # Pass tensor directly
)

dwn_lut_model = nn.Sequential(lut_layer1, lut_layer2).cuda()

print(f"✓ DWN LUT model created")

# Train LUT truth tables (10 epochs)
print("\n" + "="*70)
print("STEP 6: Train LUT truth tables (10 epochs)")
print("="*70)

lut_optimizer = torch.optim.Adam(dwn_lut_model.parameters(), lr=1e-2)

for epoch in range(10):
    dwn_lut_model.train()
    perm = torch.randperm(x_train.shape[0])
    for i in range(0, x_train.shape[0], 128):
        lut_optimizer.zero_grad()
        idx = perm[i:i+128]
        out = dwn_lut_model(x_train[idx].cuda())
        loss = cross_entropy(out, y_train[idx].cuda())
        loss.backward()
        lut_optimizer.step()

    dwn_lut_model.eval()
    with torch.no_grad():
        pred = dwn_lut_model(x_test.cuda()).argmax(dim=1).cpu()
        acc = (pred == y_test).float().mean()
    print(f"Epoch {epoch+1}/10: Test Acc = {acc:.4f}")

# Final evaluation
dwn_lut_model.eval()
with torch.no_grad():
    pred = dwn_lut_model(x_test.cuda()).argmax(dim=1).cpu()
    final_lut_acc = (pred == y_test).float().mean()

print("\n" + "="*70)
print("✅ QUICK TEST COMPLETE!")
print("="*70)
print(f"Final DWN LUT Model Accuracy: {final_lut_acc:.4f}")
print(f"✓ Connections fixed to {mapping_layer1.shape[1]} inputs per neuron")
print(f"✓ Truth tables trained")
print(f"✓ Ready for simulation!")
print("="*70)
