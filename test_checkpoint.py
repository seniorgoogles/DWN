"""
Test checkpoint save/restore logic with pruned models
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

print("=" * 60)
print("INITIAL MODEL STATE")
print("=" * 60)
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: has weight_orig = {hasattr(module, 'weight_orig')}")

# Apply pruning to the first layer (simulating first pruning step)
print("\n" + "=" * 60)
print("APPLYING FIRST PRUNING")
print("=" * 60)
prune.random_unstructured(model[0], name='weight', amount=0.3)
prune.random_unstructured(model[2], name='weight', amount=0.3)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: has weight_orig = {hasattr(module, 'weight_orig')}")

# SAVE CHECKPOINT (using the new logic)
print("\n" + "=" * 60)
print("SAVING CHECKPOINT")
print("=" * 60)

# Remove any existing pruning to save checkpoint in standard format
for module in model.modules():
    if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
        print(f"Removing pruning from {module}")
        prune.remove(module, 'weight')

checkpoint_state = copy.deepcopy(model.state_dict())
print(f"Checkpoint keys: {list(checkpoint_state.keys())}")

# Apply more pruning (simulating next pruning step)
print("\n" + "=" * 60)
print("APPLYING SECOND PRUNING")
print("=" * 60)
prune.random_unstructured(model[0], name='weight', amount=0.5)
prune.random_unstructured(model[2], name='weight', amount=0.5)

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: has weight_orig = {hasattr(module, 'weight_orig')}")

# RESTORE CHECKPOINT (using the new logic)
print("\n" + "=" * 60)
print("RESTORING CHECKPOINT")
print("=" * 60)

# Remove pruning parametrizations before loading checkpoint
for module in model.modules():
    if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
        print(f"Removing pruning from {module}")
        prune.remove(module, 'weight')

# Try to load checkpoint
try:
    model.load_state_dict(checkpoint_state)
    print("✅ SUCCESS: Checkpoint loaded successfully!")
except RuntimeError as e:
    print(f"❌ FAILED: {e}")
    exit(1)

# Verify final state
print("\n" + "=" * 60)
print("FINAL MODEL STATE")
print("=" * 60)
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name}: has weight_orig = {hasattr(module, 'weight_orig')}")

print("\n✅ All tests passed!")
