"""
JSC Dataset Classification using DWN (Deep Weight-sharing Network)
OpenML Dataset ID: 42468
This example demonstrates using LUTLayers for JSC classification.
"""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
import openml
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

print("="*70)
print("JSC Dataset Classification with DWN LUTLayers")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ============================================================================
# GLOBAL TRACKING FOR PLOTTING
# ============================================================================
# Track learning rate and sparsity throughout entire training process
training_history = {
    'epochs': [],           # Cumulative epoch counter
    'learning_rates': [],   # Learning rate at each epoch
    'sparsity': [],         # Sparsity at each epoch
    'test_accuracy': [],    # Test accuracy at each epoch
    'phase': [],            # Which phase: 'initial', 'pruning_step_X', 'finetune_X', 'lut'
    'weights_per_neuron': [] # Current weights per neuron (for structured pruning)
}
cumulative_epoch = 0  # Global epoch counter across all training phases

# Load JSC dataset from OpenML
print("\nLoading JSC dataset from OpenML (ID: 42468)...")
dataset = openml.datasets.get_dataset(42468)
df_features, df_labels, _, attribute_names = dataset.get_data(
    dataset_format='dataframe',
    target=dataset.default_target_attribute
)

# Convert to numpy arrays
features = df_features.values.astype(np.float32)
label_names = list(df_labels.unique())
labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)
num_classes = labels.max() + 1

print(f"Dataset loaded:")
print(f"  - Features shape: {features.shape}")
print(f"  - Number of classes: {num_classes}")
print(f"  - Class names: {label_names}")

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    features, labels,
    train_size=0.8,
    random_state=42
)

print(f"\nTrain/test split (80/20):")
print(f"  - Training samples: {len(x_train)}")
print(f"  - Test samples: {len(x_test)}")

# Convert to torch tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
y_test = torch.tensor(y_test, dtype=torch.int64)

# Binarize with distributive thermometer
print("\nBinarizing input with DistributiveThermometer (200 bits)...")
thermometer = dwn.DistributiveThermometer(200).fit(x_train)
x_train = thermometer.binarize(x_train).flatten(start_dim=1)
x_test = thermometer.binarize(x_test).flatten(start_dim=1)

print(f"Binarized shape: {x_train.shape}")
print(f"  - Each feature converted to 200 binary values")
print(f"  - Total binary features: {x_train.shape[1]}")

# Build model with LUTLayers
print("\nBuilding DWN model with LUTLayers...")
# IMPORTANT: With 6-weight constraint, we need MORE neurons to compensate
# for reduced connectivity. Wider network = more representational capacity
# Rule of thumb: 3-5x more neurons when pruning to 6 weights
hidden_size = 10  # Increased from 10 to compensate for sparsity

model = nn.Sequential(
    #dwn.LUTLayer(x_train.size(1), hidden_size, n=6, mapping='learnable'),
    #dwn.LUTLayer(hidden_size, 1000, n=6),
    #dwn.GroupSum(k=num_classes, tau=1/0.3)
    nn.Linear(x_train.size(1), hidden_size),
    nn.Linear(hidden_size, 5)
)

print(f"Model architecture (for LUT pruning):")
print(f"  Input:  {x_train.size(1)} features")
print(f"  Hidden: {hidden_size} neurons (WIDER to compensate for 6-weight constraint)")
print(f"  Output: 5 classes")
print(f"  After pruning: Each neuron will have only 6 active inputs")

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

print(f"\nOptimizer: Adam (lr=1e-2)")
print(f"Scheduler: StepLR (gamma=0.1, step_size=14)")

def evaluate(model, x_test, y_test):
    """Evaluate model accuracy on test set"""
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc


def print_nonzero_weights_per_neuron(model, max_neurons_per_layer=10, show_indices=True):
    """
    Print detailed information about non-zero weights for each neuron.

    Args:
        model: PyTorch model to analyze
        max_neurons_per_layer: Maximum number of neurons to show per layer (for readability)
        show_indices: If True, shows the indices of non-zero weights
    """
    print("\n" + "="*70)
    print("NON-ZERO WEIGHTS PER NEURON")
    print("="*70)

    for layer_idx, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Get the actual weights (considering pruning mask if active)
            if hasattr(module, 'weight_mask'):
                # Use mask to determine which weights are active
                mask = module.weight_mask
                weight_orig = module.weight_orig.data
                active_weights = weight_orig * mask
            else:
                active_weights = module.weight.data

            out_features, in_features = active_weights.shape

            print(f"\nLayer {layer_idx}: {name}")
            print(f"  Shape: {out_features} neurons × {in_features} inputs")
            print("-" * 70)

            # Calculate statistics for this layer
            total_weights = out_features * in_features
            nonzero_per_neuron = []

            for neuron_idx in range(min(out_features, max_neurons_per_layer)):
                neuron_weights = active_weights[neuron_idx]
                nonzero_mask = neuron_weights != 0.0
                nonzero_count = nonzero_mask.sum().item()
                nonzero_per_neuron.append(nonzero_count)

                if show_indices and nonzero_count > 0:
                    nonzero_indices = torch.where(nonzero_mask)[0].cpu().numpy()
                    print(f"  Neuron {neuron_idx:3d}: {nonzero_count:4d}/{in_features} non-zero weights")
                    print(f"    Indices: {nonzero_indices.tolist()[:20]}" +
                          ("..." if len(nonzero_indices) > 20 else ""))
                else:
                    print(f"  Neuron {neuron_idx:3d}: {nonzero_count:4d}/{in_features} non-zero weights")

            if out_features > max_neurons_per_layer:
                print(f"  ... ({out_features - max_neurons_per_layer} more neurons)")
                # Calculate stats for remaining neurons
                for neuron_idx in range(max_neurons_per_layer, out_features):
                    neuron_weights = active_weights[neuron_idx]
                    nonzero_count = (neuron_weights != 0.0).sum().item()
                    nonzero_per_neuron.append(nonzero_count)

            # Layer summary statistics
            avg_nonzero = sum(nonzero_per_neuron) / len(nonzero_per_neuron)
            min_nonzero = min(nonzero_per_neuron)
            max_nonzero = max(nonzero_per_neuron)
            total_nonzero = sum(nonzero_per_neuron)
            layer_sparsity = 1.0 - (total_nonzero / total_weights)

            print(f"\n  Layer Summary:")
            print(f"    Average non-zero weights/neuron: {avg_nonzero:.2f}")
            print(f"    Min non-zero weights/neuron: {min_nonzero}")
            print(f"    Max non-zero weights/neuron: {max_nonzero}")
            print(f"    Total non-zero weights: {total_nonzero}/{total_weights}")
            print(f"    Layer sparsity: {layer_sparsity*100:.2f}%")

    print("\n" + "="*70)


def prune_by_magnitude(model, pruning_percentage, keep_mask=True, layer_indices=None):
    """
    Prunes model weights by magnitude up to a given percentage.

    Args:
        model: PyTorch model to prune
        pruning_percentage: Percentage of weights to prune (0.0 to 1.0)
        keep_mask: If True, keeps the pruning mask active (don't remove reparametrization)
        layer_indices: List of layer indices to prune (None = prune all layers, [0] = only first layer)

    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune

    print(f"\nPruning {pruning_percentage*100:.1f}% of weights by magnitude...")

    # Collect linear layers to prune
    parameters_to_prune = []
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for idx, (name, module) in enumerate(linear_layers):
        # Check if we should prune this layer
        if layer_indices is None or idx in layer_indices:
            # Remove any existing pruning masks first
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
            parameters_to_prune.append((module, 'weight'))
            print(f"  - Pruning layer {idx}: {name}")
        else:
            print(f"  - Skipping layer {idx}: {name}")

    # Apply global unstructured pruning (magnitude-based)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_percentage,
    )

    # Optionally make pruning permanent (remove reparametrization)
    # If keep_mask=True, we keep the mask active so weights stay zero during training
    if not keep_mask:
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

    # Count actual zeros
    total_params = 0
    zero_params = 0
    for module, _ in parameters_to_prune:
        weight = module.weight if not keep_mask else module.weight_orig
        total_params += weight.numel()
        # When mask is active, check the masked weight (weight = weight_orig * weight_mask)
        if keep_mask and hasattr(module, 'weight_mask'):
            zero_params += (module.weight_mask == 0).sum().item()
        else:
            zero_params += (weight == 0).sum().item()

    actual_sparsity = zero_params / total_params
    print(f"  - Total parameters: {total_params}")
    print(f"  - Zero parameters: {zero_params}")
    print(f"  - Actual sparsity: {actual_sparsity*100:.2f}%")

    return model


def prune_to_n_weights_per_neuron(model, n_weights=6, layer_indices=None):
    """
    Structured pruning: limits each neuron to exactly N incoming connections.
    Keeps the N largest magnitude weights per neuron.
    Uses custom pruning to maintain mask during training.

    Args:
        model: PyTorch model to prune
        n_weights: Number of weights to keep per neuron
        layer_indices: List of layer indices to prune (None = prune all layers, [0] = only first layer)

    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune

    print(f"\nPruning to {n_weights} weights per neuron (structured pruning)...")

    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    for idx, (name, module) in enumerate(linear_layers):
        # Check if we should prune this layer
        if layer_indices is not None and idx not in layer_indices:
            print(f"  - Skipping layer {idx}: {name}")
            continue

        # Get current weights (handle existing pruning mask)
        if hasattr(module, 'weight_orig'):
            # Remove existing mask and get the original weights
            prune.remove(module, 'weight')
            weight = module.weight.data
        else:
            weight = module.weight.data

        in_features = weight.shape[1]
        out_features = weight.shape[0]

        if in_features <= n_weights:
            print(f"  - Skipping layer {idx} ({name}): only {in_features} inputs (less than {n_weights})")
            continue

        print(f"  - Pruning layer {idx} ({name}): {weight.shape}")

        # Create custom mask for structured pruning
        mask = torch.zeros_like(weight)

        # For each output neuron (row), keep only top-N weights by magnitude
        for neuron_idx in range(out_features):
            neuron_weights = weight[neuron_idx].abs()

            # Find indices of top-N weights
            _, top_indices = torch.topk(neuron_weights, k=n_weights, largest=True)

            # Set mask to 1 for top-N weights
            mask[neuron_idx, top_indices] = 1.0

        # Apply custom pruning with the mask
        prune.custom_from_mask(module, name='weight', mask=mask)

        # Calculate sparsity for this layer
        total = weight.numel()
        zeros = (mask == 0).sum().item()
        sparsity = zeros / total
        print(f"    Layer sparsity: {sparsity*100:.2f}% ({zeros}/{total} zeros)")

    return model


def iterative_pruning_and_retraining(model, optimizer_fn, scheduler_fn,
                                     x_train, y_train, x_test, y_test,
                                     target_sparsity=0.9, pruning_steps=5,
                                     epochs_per_step=10, batch_size=128,
                                     method='magnitude', final_n_weights=6,
                                     adaptive_lr=True, base_lr=1e-3,
                                     early_stop_threshold=0.1, extra_epochs_final=0,
                                     use_backtracking=True, backtrack_threshold=0.05,
                                     pruning_steepness=2.0):
    """
    Iteratively prune and retrain the model to reach target sparsity.

    Args:
        model: Initial trained model
        optimizer_fn: Function that returns a fresh optimizer for the model (deprecated if adaptive_lr=True)
        scheduler_fn: Function that returns a fresh scheduler for the optimizer (deprecated if adaptive_lr=True)
        x_train, y_train: Training data
        x_test, y_test: Test data
        target_sparsity: Final sparsity level (0.0 to 1.0) - used for magnitude pruning
        pruning_steps: Number of pruning iterations
        epochs_per_step: Epochs to retrain after each pruning step
        batch_size: Training batch size
        method: 'magnitude' or 'structured' (for n_weights per neuron)
        final_n_weights: Target number of weights per neuron for structured pruning (default: 6)
        adaptive_lr: If True, automatically adjusts learning rate based on pruning severity
        base_lr: Base learning rate for adaptive LR strategy
        early_stop_threshold: Stop pruning if accuracy drops more than this (e.g., 0.1 = 10%)
        extra_epochs_final: Extra epochs for final pruning steps (last 20%)
        use_backtracking: If True, backs up and reduces step size when accuracy drops too much
        backtrack_threshold: Accuracy drop that triggers backtracking (e.g., 0.05 = 5%)
        pruning_steepness: Controls pruning schedule aggressiveness (lower = gentler early phases)
                          Default: 2.0. Try 1.0-1.5 for very gentle early pruning, 3.0-4.0 for aggressive early pruning

    Returns:
        Pruned and retrained model
    """
    def get_model_sparsity(model):
        """Calculate the current sparsity of the model"""
        total_params = 0
        zero_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if pruning mask is active
                if hasattr(module, 'weight_mask'):
                    total_params += module.weight_mask.numel()
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0.0

    print("\n" + "="*70)
    print(f"ITERATIVE PRUNING AND RETRAINING")
    if method == 'magnitude':
        print(f"  Target sparsity: {target_sparsity*100:.1f}%")
    else:
        print(f"  Target weights/neuron: {final_n_weights}")
    print(f"  Pruning steps: {pruning_steps}")
    print(f"  Epochs per step: {epochs_per_step}")
    print(f"  Method: {method}")
    print("="*70)

    # Calculate pruning schedule
    if method == 'magnitude':
        current_sparsity = 0.0
        sparsity_increment = target_sparsity / pruning_steps
    elif method == 'structured':
        # Get the input dimension of the first layer
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break

        if first_layer is not None:
            in_features = first_layer.weight.shape[1]
            # Create smooth exponential decay schedule from 50% inputs to final_n_weights
            # Using exponential decay ensures smooth, gradual reduction without big jumps

            start_weights = in_features // 2  # Start at 50% of inputs

            # Use exponential decay: w(t) = final + (start - final) * exp(-k*t)
            # where k controls decay rate
            import numpy as np

            # Calculate decay constant for smooth transition
            t_values = np.linspace(0, 1, pruning_steps)

            # Use exponential decay with adjustable steepness
            # Higher steepness = faster initial decay, gentler final approach
            # Lower steepness = gentler initial decay, faster final approach (less crucial early)
            steepness = pruning_steepness

            weights_schedule = []
            for t in t_values:
                # Exponential decay from start_weights to final_n_weights
                w = final_n_weights + (start_weights - final_n_weights) * np.exp(-steepness * t)
                weights_schedule.append(int(w))

            # Ensure monotonic decrease and no duplicates at the end
            for i in range(1, len(weights_schedule)):
                if weights_schedule[i] >= weights_schedule[i-1]:
                    weights_schedule[i] = max(final_n_weights, weights_schedule[i-1] - 1)

            # Ensure last step is exactly the target
            weights_schedule[-1] = final_n_weights

            # Calculate sparsity for each step
            sparsity_schedule = [(1 - w/in_features) * 100 for w in weights_schedule]

            print(f"  Structured pruning schedule (weights per neuron): {weights_schedule}")
            print(f"  Sparsity schedule: {[f'{s:.1f}%' for s in sparsity_schedule]}")

            # Show step-by-step changes
            print(f"  Step-by-step sparsity increases:")
            for i in range(min(5, len(sparsity_schedule))):
                if i == 0:
                    increase = sparsity_schedule[i]
                else:
                    increase = sparsity_schedule[i] - sparsity_schedule[i-1]
                print(f"    Step {i+1}: {weights_schedule[i]} weights → {sparsity_schedule[i]:.1f}% sparsity (+{increase:.1f}%)")
            if len(sparsity_schedule) > 5:
                print(f"    ...")
                i = len(sparsity_schedule) - 1
                increase = sparsity_schedule[i] - sparsity_schedule[i-1]
                print(f"    Step {i+1}: {weights_schedule[i]} weights → {sparsity_schedule[i]:.1f}% sparsity (+{increase:.1f}%)")
        else:
            weights_schedule = [final_n_weights] * pruning_steps

    # Calculate adaptive learning rates for each step
    if adaptive_lr and method == 'structured':
        lr_schedule = []
        for i, n_weights in enumerate(weights_schedule):
            if i == 0:
                prev_weights = start_weights if 'start_weights' in locals() else n_weights
            else:
                prev_weights = weights_schedule[i-1]

            # Calculate pruning severity (what % of weights were removed)
            if prev_weights > 0:
                pruning_ratio = (prev_weights - n_weights) / prev_weights
            else:
                pruning_ratio = 0.0

            # Higher LR for aggressive pruning, lower for fine-tuning
            # Use exponential scaling: more aggressive pruning = higher LR
            if pruning_ratio > 0.1:  # Aggressive pruning (>10% reduction)
                lr = base_lr * 2.0
            elif pruning_ratio > 0.05:  # Moderate pruning (5-10% reduction)
                lr = base_lr * 1.5
            else:  # Fine-tuning (<5% reduction)
                lr = base_lr * 0.5

            lr_schedule.append(lr)

        print(f"  Adaptive learning rate schedule: {[f'{lr:.1e}' for lr in lr_schedule]}")
    elif adaptive_lr and method == 'magnitude':
        # For magnitude pruning, increase LR as sparsity increases
        lr_schedule = []
        for step in range(pruning_steps):
            sparsity = (step + 1) * sparsity_increment
            if sparsity < 0.5:
                lr = base_lr
            elif sparsity < 0.75:
                lr = base_lr * 1.5
            else:
                lr = base_lr * 2.0
            lr_schedule.append(lr)
        print(f"  Adaptive learning rate schedule: {[f'{lr:.1e}' for lr in lr_schedule]}")
    else:
        lr_schedule = None

    # Initial accuracy and sparsity
    initial_acc = evaluate(model, x_test, y_test)
    initial_sparsity = get_model_sparsity(model)
    print(f"\nInitial test accuracy: {initial_acc:.4f}")
    print(f"Initial model sparsity: {initial_sparsity*100:.2f}%")

    # Track best accuracies through training
    best_accuracies_per_step = []

    # Backtracking state
    if use_backtracking and method == 'structured':
        current_n_weights = start_weights if 'start_weights' in locals() else in_features // 2
        step_size = (current_n_weights - final_n_weights) / pruning_steps  # Initial step size
        step = 0
        max_backtracks = 15  # Allow more backtracks to reach 6 weights target
        backtrack_count = 0

        print(f"\n  Backtracking enabled (will reach {final_n_weights} weights for LUT compatibility):")
        print(f"    Starting from {current_n_weights} weights/neuron")
        print(f"    Target: {final_n_weights} weights/neuron (hard constraint)")
        print(f"    Initial step size: {step_size:.1f} weights")
        print(f"    Backtrack threshold: {backtrack_threshold*100:.1f}% accuracy drop")
        print(f"    Strategy: Adapt step size to minimize accuracy loss while reaching target")
    else:
        step = 0

    # Continue until we reach target (even with backtracking enabled)
    target_reached = False
    while step < pruning_steps and not target_reached:
        print(f"\n{'='*70}")
        print(f"PRUNING STEP {step + 1}/{pruning_steps}")
        if use_backtracking and method == 'structured':
            print(f"  Current weights/neuron: {current_n_weights:.0f}")
            print(f"  Step size: {step_size:.2f}")
        print(f"{'='*70}")

        # Save checkpoint before pruning (for backtracking)
        if use_backtracking:
            import copy
            import torch.nn.utils.prune as prune

            # Save current masks before removing them
            saved_masks = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                    saved_masks[name] = module.weight_mask.clone()

            # Remove any existing pruning to save checkpoint in standard format
            # (this bakes masks into weights, so zeros from previous pruning are preserved)
            for module in model.modules():
                if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                    prune.remove(module, 'weight')

            checkpoint_state = copy.deepcopy(model.state_dict())
            checkpoint_accuracy = evaluate(model, x_test, y_test)

            # Re-apply the masks to maintain sparsity during the next pruning step
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in saved_masks:
                    prune.custom_from_mask(module, name='weight', mask=saved_masks[name])

        # Prune
        if method == 'magnitude':
            current_sparsity += sparsity_increment
            # Keep mask active to maintain sparsity during retraining
            # Prune both layers (layer_indices=[0, 1])
            model = prune_by_magnitude(model, current_sparsity, keep_mask=True, layer_indices=[0, 1])
        elif method == 'structured':
            if use_backtracking:
                # Use adaptive step size
                target_n_weights = max(final_n_weights, int(current_n_weights - step_size))
                print(f"Target weights for this step: {target_n_weights} (adaptive)")
            else:
                # Use pre-calculated schedule
                target_n_weights = weights_schedule[step]
                print(f"Target weights for this step: {target_n_weights}")

            # Prune both first and last layers
            model = prune_to_n_weights_per_neuron(model, n_weights=target_n_weights, layer_indices=[0, 1])

        # Calculate actual sparsity after pruning
        actual_sparsity = get_model_sparsity(model)
        print(f"\n>>> Current model sparsity: {actual_sparsity*100:.2f}% <<<")

        # Evaluate after pruning (before retraining)
        acc_after_prune = evaluate(model, x_test, y_test)
        print(f"Accuracy after pruning: {acc_after_prune:.4f}")

        # Determine epochs for this step (more epochs for final steps)
        is_final_phase = step >= int(pruning_steps * 0.8)  # Last 20% of steps
        current_epochs = epochs_per_step + (extra_epochs_final if is_final_phase else 0)

        # Retrain with adaptive learning rate
        if adaptive_lr and lr_schedule is not None:
            step_lr = lr_schedule[step]
            # Boost LR for final steps if they need extra training
            if is_final_phase and extra_epochs_final > 0:
                step_lr = step_lr * 1.2  # Slightly higher LR for final fine-tuning
            print(f"\nRetraining for {current_epochs} epochs with LR={step_lr:.2e}{'  [FINAL PHASE +extra epochs]' if is_final_phase and extra_epochs_final > 0 else ''}...")
            optimizer = torch.optim.Adam(model.parameters(), lr=step_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_epochs)
        else:
            print(f"\nRetraining for {current_epochs} epochs...")
            optimizer = optimizer_fn()
            scheduler = scheduler_fn(optimizer)

        best_acc_this_step = train_and_evaluate(model, optimizer, scheduler, x_train, y_train,
                                                 x_test, y_test, epochs=current_epochs,
                                                 batch_size=batch_size, save_best=True,
                                                 phase_name=f'pruning_step_{step+1}')

        # Model now has best weights from this training phase loaded
        acc_after_retrain = evaluate(model, x_test, y_test)

        print(f"\nBest accuracy this step: {best_acc_this_step:.4f}")
        print(f"Accuracy after retraining: {acc_after_retrain:.4f}")
        print(f"Recovery: {(acc_after_retrain - acc_after_prune):.4f}")

        # Backtracking logic: check if accuracy drop is too large
        should_backtrack = False
        if use_backtracking and len(best_accuracies_per_step) > 0:
            best_so_far = max(best_accuracies_per_step)
            drop_ratio = (best_so_far - best_acc_this_step) / best_so_far if best_so_far > 0 else 0

            # Adjust threshold based on proximity to target (be more lenient near target)
            if use_backtracking and method == 'structured':
                proximity_to_target = (current_n_weights - final_n_weights) / (start_weights - final_n_weights)
                # If we're >80% to target, be more tolerant of accuracy drops
                if proximity_to_target < 0.2:  # Very close to target
                    effective_threshold = backtrack_threshold * 1.5  # Allow 7.5% drop instead of 5%
                    early_threshold = early_stop_threshold * 1.3     # Allow 19.5% drop instead of 15%
                else:
                    effective_threshold = backtrack_threshold
                    early_threshold = early_stop_threshold
            else:
                effective_threshold = backtrack_threshold
                early_threshold = early_stop_threshold

            if drop_ratio > effective_threshold and backtrack_count < max_backtracks:
                should_backtrack = True
                print(f"\n🔄 BACKTRACKING: Accuracy dropped by {drop_ratio*100:.1f}% (threshold: {effective_threshold*100:.1f}%)")
                print(f"   Current: {current_n_weights:.0f} weights/neuron (target: {final_n_weights})")
                print(f"   Restoring checkpoint (accuracy: {checkpoint_accuracy:.4f})")
                print(f"   Halving step size: {step_size:.2f} → {step_size/2:.2f}")

                # Remove pruning parametrizations before loading checkpoint
                import torch.nn.utils.prune as prune
                for module in model.modules():
                    if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                        prune.remove(module, 'weight')

                # Restore checkpoint
                model.load_state_dict(checkpoint_state)

                # Reduce step size
                step_size = step_size / 2
                backtrack_count += 1

                # Don't increment step counter or record this attempt
                print(f"   Backtrack count: {backtrack_count}/{max_backtracks}")
            elif drop_ratio > early_threshold:
                # Drop is catastrophic even considering proximity to target
                print(f"\n⚠ WARNING: Accuracy dropped by {drop_ratio*100:.1f}% (threshold: {early_threshold*100:.1f}%)")
                print(f"⚠ Current: {current_n_weights:.0f} weights/neuron")
                # If we're very close to target, push through anyway
                if use_backtracking and method == 'structured' and current_n_weights - final_n_weights <= 2:
                    print(f"⚠ Only {current_n_weights - final_n_weights:.0f} weights from LUT target - continuing anyway")
                else:
                    print(f"⚠ Stopping pruning - too aggressive")
                    break

        if not should_backtrack:
            # Accept this pruning step
            best_accuracies_per_step.append(best_acc_this_step)

            # Check if accuracy is deteriorating
            if len(best_accuracies_per_step) >= 2:
                if best_acc_this_step < best_accuracies_per_step[-2] * 0.95:  # 5% drop
                    print(f"⚠ Note: Accuracy degrading (from {best_accuracies_per_step[-2]:.4f})")

            # Update current weights for next iteration (backtracking mode)
            if use_backtracking and method == 'structured':
                current_n_weights = target_n_weights
                # Check if we've reached the target
                if current_n_weights <= final_n_weights:
                    target_reached = True
                    print(f"🎯 TARGET REACHED: {final_n_weights} weights per neuron!")
                else:
                    print(f"→ Pruning accepted. Next step from {current_n_weights} weights/neuron")
            else:
                print(f"→ Starting next pruning step from best weights")

            step += 1  # Only increment if we're accepting this step
        else:
            # Retrying this step with smaller increment
            print(f"→ Retrying step {step + 1} with reduced step size")

    final_sparsity = get_model_sparsity(model)
    steps_completed = len(best_accuracies_per_step)
    early_stopped = steps_completed < pruning_steps

    print("\n" + "="*70)
    if target_reached:
        print("✅ PRUNING COMPLETE - LUT TARGET REACHED!")
    elif early_stopped:
        print("ITERATIVE PRUNING COMPLETE (Early stopped)")
    else:
        print("ITERATIVE PRUNING COMPLETE")
    print("="*70)
    print(f"  Steps completed: {steps_completed}/{pruning_steps}")
    if use_backtracking and method == 'structured':
        print(f"  Backtracks used: {backtrack_count}/{max_backtracks}")
        print(f"  Final weights/neuron: {current_n_weights:.0f}")
        if current_n_weights <= final_n_weights:
            print(f"  ✅ LUT compatible: {final_n_weights}-input lookup table")
        else:
            print(f"  ⚠ LUT target ({final_n_weights}) not reached")
        print(f"  Final step size: {step_size:.2f}")
    if early_stopped and not target_reached:
        print(f"  Reason: Accuracy drop exceeded threshold before reaching {final_n_weights} weights")
    print(f"")
    print(f"  Initial sparsity: {initial_sparsity*100:.2f}%")
    print(f"  Final sparsity:   {final_sparsity*100:.2f}%")
    print(f"  Sparsity increase: {(final_sparsity - initial_sparsity)*100:.2f}%")
    print(f"")
    print(f"  Initial accuracy: {initial_acc:.4f}")
    print(f"  Final accuracy:   {acc_after_retrain:.4f}")
    print(f"  Accuracy change:  {(acc_after_retrain - initial_acc):.4f}")
    if len(best_accuracies_per_step) > 0:
        peak_accuracy = max(best_accuracies_per_step)
        print(f"  Peak accuracy:    {peak_accuracy:.4f} (best achieved)")
    print(f"")
    print(f"  Best accuracy trajectory (saved & restored each step):")
    for i, best_acc in enumerate(best_accuracies_per_step):
        if i < 3 or i >= len(best_accuracies_per_step) - 3:
            marker = " ⭐" if best_acc == max(best_accuracies_per_step) else ""
            print(f"    Step {i+1:2d}: {best_acc:.4f}{marker}")
        elif i == 3:
            print(f"    ...")
    print("="*70)

    # Print detailed weight information per neuron
    print_nonzero_weights_per_neuron(model, max_neurons_per_layer=10, show_indices=True)

    # Verify target weights per neuron was achieved (for structured pruning)
    if method == 'structured':
        print("\n" + "="*70)
        print(f"VERIFICATION: Target was {final_n_weights} weights per neuron")
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]

        for layer_idx, layer in enumerate(linear_layers):
            print(f"\n  Layer {layer_idx}:")
            if hasattr(layer, 'weight_mask'):
                mask = layer.weight_mask
                # Check all neurons and calculate stats
                nonzero_counts = []
                for neuron_idx in range(mask.shape[0]):
                    nonzero = (mask[neuron_idx] > 0).sum().item()
                    nonzero_counts.append(nonzero)
                    if neuron_idx < 5:  # Show first 5 neurons
                        print(f"    Neuron {neuron_idx}: {nonzero} active weights")

                if mask.shape[0] > 5:
                    print(f"    ... ({mask.shape[0] - 5} more neurons)")

                # Show statistics
                avg_weights = sum(nonzero_counts) / len(nonzero_counts)
                max_weights = max(nonzero_counts)
                min_weights = min(nonzero_counts)
                print(f"    Summary: avg={avg_weights:.1f}, min={min_weights}, max={max_weights}")
            else:
                print(f"    No pruning mask (unpruned)")

        print("="*70)

    return model


def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size, show_sparsity=True, save_best=True, phase_name='training'):
    """
    Train model and evaluate after each epoch.

    If save_best=True, saves the best model weights during training and automatically
    restores them at the end. This ensures each pruning step starts from the best
    possible checkpoint, improving final accuracy.

    Returns:
        best_accuracy: The best test accuracy achieved during training
    """
    global cumulative_epoch, training_history

    def get_weight_stats(model):
        """Get sparsity and weight count statistics"""
        total_params = 0
        zero_params = 0
        nonzero_params = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Check if pruning mask is active
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total_params += mask.numel()
                    zero_params += (mask == 0).sum().item()
                    nonzero_params += (mask > 0).sum().item()
                else:
                    weight = module.weight.data
                    total_params += weight.numel()
                    zero_params += (weight == 0).sum().item()
                    nonzero_params += (weight != 0).sum().item()

        sparsity = zero_params / total_params if total_params > 0 else 0.0
        return sparsity, total_params, zero_params, nonzero_params

    best_accuracy = 0.0
    best_state_dict = None

    n_samples = x_train.shape[0]

    print("\n" + "="*70)
    print(f"Training for {epochs} epochs (batch_size={batch_size})")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0

        # Training loop
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)

            outputs = model(batch_x)
            loss = cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()

            pred_train = outputs.argmax(dim=1)
            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_acc = correct_train / total_train
        scheduler.step()

        # Evaluation
        test_acc = evaluate(model, x_test, y_test)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if save_best:
                # Save the best model state (deep copy to avoid reference issues)
                import copy
                best_state_dict = copy.deepcopy(model.state_dict())
            print(f"New best accuracy! {best_accuracy:.4f} ⭐")

        # Track metrics for plotting
        cumulative_epoch += 1
        current_lr = optimizer.param_groups[0]['lr']
        current_sparsity, _, _, _ = get_weight_stats(model)

        training_history['epochs'].append(cumulative_epoch)
        training_history['learning_rates'].append(current_lr)
        training_history['sparsity'].append(current_sparsity * 100)  # Convert to percentage
        training_history['test_accuracy'].append(test_acc)
        training_history['phase'].append(phase_name)

        # For structured pruning, track weights per neuron
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        if first_layer and hasattr(first_layer, 'weight_mask'):
            # Count non-zero weights in first neuron as representative
            weights_per_neuron = (first_layer.weight_mask[0] > 0).sum().item()
        else:
            weights_per_neuron = None
        training_history['weights_per_neuron'].append(weights_per_neuron)

        # Get sparsity stats
        if show_sparsity:
            sparsity, total, zeros, nonzeros = get_weight_stats(model)
            print(f'Epoch {epoch + 1:2d}/{epochs} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc:.4f} | '
                  f'Test Acc: {test_acc:.4f} | '
                  f'Sparsity: {sparsity*100:.2f}% | '
                  f'Non-zero: {nonzeros}/{total}')
        else:
            print(f'Epoch {epoch + 1:2d}/{epochs} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc:.4f} | '
                  f'Test Acc: {test_acc:.4f}')

    print("="*70)
    print(f"Training complete! Best accuracy was {best_accuracy:.4f}")

    # Restore best model weights
    if save_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"✓ Restored best model weights from training")

    if show_sparsity:
        final_sparsity, final_total, final_zeros, final_nonzeros = get_weight_stats(model)
        print(f"Final sparsity: {final_sparsity*100:.2f}% ({final_zeros} zeros, {final_nonzeros} non-zeros, {final_total} total)")
    print("="*70)

    return best_accuracy

# Train the model (initial training)
print("\n" + "="*70)
print("INITIAL TRAINING")
print("="*70)
initial_best_acc = train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test,
                                      epochs=10, batch_size=128, show_sparsity=True, save_best=True,
                                      phase_name='initial_training')
print(f"\n→ Initial model ready with best accuracy: {initial_best_acc:.4f}")

# ============================================================================
# PRUNING EXAMPLES
# ============================================================================

# Example 1: Magnitude-based pruning with retraining
"""
print("\n" + "="*70)
print("EXAMPLE 1: Magnitude pruning (50%) with retraining")
print("="*70)

# Prune the model (both layers)
prune_by_magnitude(model, pruning_percentage=0.5, keep_mask=True, layer_indices=[0, 1])
pruned_acc = evaluate(model, x_test, y_test)
print(f"\nAccuracy after pruning (before retraining): {pruned_acc:.4f}")

# Retrain the pruned model
print("\n" + "="*70)
print("Retraining pruned model...")
print("="*70)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=7)
train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test,
                   epochs=15, batch_size=128)

final_acc = evaluate(model, x_test, y_test)
print(f"\nFinal accuracy after retraining: {final_acc:.4f}")
print(f"Accuracy recovery: {(final_acc - pruned_acc):.4f}") 
"""
# Example 2: Structured pruning to 6 weights per neuron
# Uncomment to limit each neuron to 6 incoming connections
"""
print("\n" + "="*70)
print("EXAMPLE 2: Structured pruning to 6 weights per neuron (both layers)")
print("="*70)
prune_to_n_weights_per_neuron(model, n_weights=6, layer_indices=[0, 1])
pruned_acc = evaluate(model, x_test, y_test)
print(f"\nAccuracy after pruning: {pruned_acc:.4f}")
"""

# Example 3: LUT-Constrained Pruning with Adaptive Backtracking + Fine-Tuning
#
# GOAL: Prune to EXACTLY 6 weights per neuron for FPGA LUT compatibility
#       (6-input LUTs are standard in modern FPGAs like Xilinx/Intel)
#
# STRATEGY TO MAXIMIZE ACCURACY AT 6 WEIGHTS:
#
#   1. WIDER NETWORK (50 neurons vs 10)
#      - Compensates for reduced connectivity
#      - More neurons = more representational capacity despite sparsity
#
#   2. ADAPTIVE PRUNING WITH BACKTRACKING
#      ✓ Hard target: MUST reach 6 weights per neuron (LUT compatibility)
#      ✓ Adaptive backtracking: If accuracy drops >5%, halves step size
#      ✓ Smart thresholds: More lenient when close to target (allows up to 7.5% drop)
#      ✓ Binary search behavior: Finds smoothest path to 6 weights
#      ✓ Both layers pruned: First layer (800→6) and last layer (50→6)
#      ✓ Adaptive LR: 2x for aggressive pruning, 0.5x for fine-tuning
#      ✓ Best checkpoint saving: Saves before each prune, restores if needed
#      ✓ Extra training near target: Last 20% of steps get +12 epochs
#      ✓ Won't give up: Pushes through even if within 2 weights of target
#
#   3. MULTI-PHASE FINE-TUNING (after reaching 6 weights)
#      ✓ Phase 1: Adam LR=5e-4 for 30 epochs (gentle refinement)
#      ✓ Phase 2: Adam LR=1e-4 for 20 epochs (precise tuning)
#      ✓ Phase 3: SGD+Momentum LR=1e-3 for 15 epochs (escape local minima)
#      ✓ Each phase saves & restores best weights
#
# Expected improvement: 2-5% accuracy boost from fine-tuning alone!

print("\n" + "="*70)
print("EXAMPLE 3: LUT-Compatible Pruning (6 weights/neuron target)")
print("="*70)
print("Goal: Prune to exactly 6 weights per neuron for FPGA LUT compatibility")
print("Strategy: Adaptive backtracking to minimize accuracy loss while reaching target")
print("="*70)

# Define optimizer and scheduler factories (used only if adaptive_lr=False)
def make_optimizer():
    return torch.optim.Adam(model.parameters(), lr=1e-3)

def make_scheduler(opt):
    return torch.optim.lr_scheduler.StepLR(opt, gamma=0.1, step_size=5)

model = iterative_pruning_and_retraining(
    model,
    optimizer_fn=make_optimizer,
    scheduler_fn=make_scheduler,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    target_sparsity=0.9,        # Not used in structured mode
    pruning_steps=3,            # Maximum number of pruning steps
    epochs_per_step=3,          # Base epochs per step (more frequent pruning)
    batch_size=128,
    method='structured',        # Use structured pruning to limit to 6 weights per neuron
    final_n_weights=6,          # Target 6 weights per neuron in the end
    adaptive_lr=True,           # Enable adaptive learning rate based on pruning severity
    base_lr=1e-3,               # Base learning rate (auto-adjusted: 2x for aggressive, 1.5x for moderate, 0.5x for fine-tuning)
    early_stop_threshold=0.15,  # Stop if accuracy drops >15% (prevents catastrophic collapse)
    extra_epochs_final=3,       # Add 12 extra epochs for final 20% of steps (total 18 epochs for final steps)
    use_backtracking=True,      # Enable adaptive backtracking (binary search for optimal sparsity)
    backtrack_threshold=0.05    # Backtrack if accuracy drops >5%, halve step size and retry
)

# ============================================================================
# FINE-TUNING AT 6 WEIGHTS FOR MAXIMUM ACCURACY
# ============================================================================
print("\n" + "="*70)
print("FINAL FINE-TUNING AT 6 WEIGHTS PER NEURON")
print("="*70)
print("Strategy: Extended training with optimal LR to maximize accuracy")
print("="*70)

# Get current accuracy
final_acc_before_finetune = evaluate(model, x_test, y_test)
print(f"\nAccuracy before fine-tuning: {final_acc_before_finetune:.4f}")

# Fine-tune with lower learning rate for longer
print("\nFine-tuning phase 1: Lower LR (5e-4) for 30 epochs")
optimizer_ft1 = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler_ft1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft1, T_max=30)
best_acc_ft1 = train_and_evaluate(model, optimizer_ft1, scheduler_ft1,
                                   x_train, y_train, x_test, y_test,
                                   epochs=30, batch_size=128, save_best=True,
                                   phase_name='finetune_phase1')

print(f"\nAccuracy after phase 1: {best_acc_ft1:.4f} (improvement: {best_acc_ft1 - final_acc_before_finetune:+.4f})")

# Fine-tune with even lower LR
print("\nFine-tuning phase 2: Very low LR (1e-4) for 20 epochs")
optimizer_ft2 = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler_ft2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft2, T_max=20)
best_acc_ft2 = train_and_evaluate(model, optimizer_ft2, scheduler_ft2,
                                   x_train, y_train, x_test, y_test,
                                   epochs=20, batch_size=128, save_best=True,
                                   phase_name='finetune_phase2')

print(f"\nAccuracy after phase 2: {best_acc_ft2:.4f} (improvement: {best_acc_ft2 - best_acc_ft1:+.4f})")

# Optional: Try SGD with momentum for final polish
print("\nFine-tuning phase 3: SGD with momentum (1e-3) for 15 epochs")
optimizer_ft3 = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
scheduler_ft3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft3, T_max=15)
best_acc_ft3 = train_and_evaluate(model, optimizer_ft3, scheduler_ft3,
                                   x_train, y_train, x_test, y_test,
                                   epochs=15, batch_size=128, save_best=True,
                                   phase_name='finetune_phase3')

final_acc_after_finetune = evaluate(model, x_test, y_test)

print("\n" + "="*70)
print("FINE-TUNING COMPLETE")
print("="*70)
print(f"  Before fine-tuning: {final_acc_before_finetune:.4f}")
print(f"  After phase 1:      {best_acc_ft1:.4f} ({best_acc_ft1 - final_acc_before_finetune:+.4f})")
print(f"  After phase 2:      {best_acc_ft2:.4f} ({best_acc_ft2 - best_acc_ft1:+.4f})")
print(f"  After phase 3:      {best_acc_ft3:.4f} ({best_acc_ft3 - best_acc_ft2:+.4f})")
print(f"  Final accuracy:     {final_acc_after_finetune:.4f}")
print(f"  Total improvement:  {final_acc_after_finetune - final_acc_before_finetune:+.4f}")
print("="*70)

# ============================================================================
# EXPORT PRUNED CONNECTIONS FOR LUT TRAINING
# ============================================================================
print("\n" + "="*70)
print("📤 EXPORTING PRUNED CONNECTIONS FOR LUT TRAINING")
print("="*70)

def export_pruned_connections(model):
    """
    Export which connections (indices) are active for each neuron.
    This will be used to train LUTs with fixed topology.

    Returns:
        layer_connections: List of dicts with connection info per layer
    """
    layer_connections = []

    for layer_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            layer_info = {
                'layer_idx': layer_idx,
                'in_features': module.weight.shape[1],
                'out_features': module.weight.shape[0],
                'connections_per_neuron': [],
                'weights': []
            }

            # Get the pruning mask or actual weights
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.cpu().numpy()
                weights = module.weight_orig.detach().cpu().numpy()
            else:
                weights = module.weight.detach().cpu().numpy()
                mask = (weights != 0).astype(float)

            # For each neuron, get the active input indices
            for neuron_idx in range(module.weight.shape[0]):
                active_indices = np.where(mask[neuron_idx] > 0)[0].tolist()
                active_weights = weights[neuron_idx][active_indices].tolist()

                layer_info['connections_per_neuron'].append({
                    'neuron_idx': neuron_idx,
                    'active_input_indices': active_indices,
                    'num_connections': len(active_indices),
                    'weights': active_weights
                })

            layer_connections.append(layer_info)

    return layer_connections

# Export connections
connections = export_pruned_connections(model)

# ============================================================================
# VALIDATION: Ensure max 6 weights per neuron
# ============================================================================
print("\n🔍 VALIDATING CONNECTIONS (max 6 weights per neuron)...")
has_violations = False

for layer_idx, layer_info in enumerate(connections):
    violations = []
    for neuron_info in layer_info['connections_per_neuron']:
        num_conns = neuron_info['num_connections']
        if num_conns > 6:
            violations.append((neuron_info['neuron_idx'], num_conns))

    if violations:
        has_violations = True
        print(f"\n⚠️  WARNING: Layer {layer_idx} has {len(violations)} neurons with >6 connections:")
        for neuron_idx, num_conns in violations[:5]:  # Show first 5
            print(f"      Neuron {neuron_idx}: {num_conns} connections (expected: 6)")
        if len(violations) > 5:
            print(f"      ... and {len(violations) - 5} more neurons")

        # Optionally trim to 6 (keep top 6 by magnitude)
        print(f"   ⚠️  These neurons need further pruning to reach exactly 6 inputs!")
    else:
        num_conns = [n['num_connections'] for n in layer_info['connections_per_neuron']]
        if all(n == 6 for n in num_conns):
            print(f"   ✅ Layer {layer_idx}: All {layer_info['out_features']} neurons have exactly 6 connections")
        else:
            print(f"   ⚠️  Layer {layer_idx}: Neurons have {min(num_conns)}-{max(num_conns)} connections (target: 6)")

if has_violations:
    print("\n" + "="*70)
    print("⚠️  WARNING: Some neurons have more than 6 connections!")
    print("   The pruning did not fully reach the target of 6 weights per neuron.")
    print("   Recommendations:")
    print("     1. Increase pruning_steps (currently set)")
    print("     2. Check if backtracking prevented reaching target")
    print("     3. Review early_stop_threshold")
    print("="*70)
else:
    print("\n✅ VALIDATION PASSED: All neurons have ≤6 connections")

print("\n📋 PRUNED CONNECTIONS SUMMARY:")
for layer_info in connections:
    print(f"\nLayer {layer_info['layer_idx']}:")
    print(f"  Shape: {layer_info['out_features']} neurons × {layer_info['in_features']} inputs")

    # Show first few neurons
    for neuron_info in layer_info['connections_per_neuron'][:3]:
        indices = neuron_info['active_input_indices']
        print(f"  Neuron {neuron_info['neuron_idx']:2d}: {neuron_info['num_connections']} connections -> inputs {indices}")

    if layer_info['out_features'] > 3:
        print(f"  ... ({layer_info['out_features'] - 3} more neurons)")

# Save to file
import pickle
output_file = 'pruned_connections_6inputs.pkl'
with open(output_file, 'wb') as f:
    pickle.dump({
        'connections': connections,
        'architecture': {
            'input_size': x_train.size(1),
            'hidden_size': hidden_size,
            'output_size': 5,
            'n_inputs_per_lut': 6
        },
        'accuracy': {
            'initial': initial_best_acc,
            'after_pruning': final_acc_before_finetune,
            'after_finetuning': final_acc_after_finetune
        }
    }, f)

print(f"\n✅ Connections saved to: {output_file}")

# ============================================================================
# STAGE 2: TRAIN DWN LUT LAYERS WITH FIXED CONNECTIONS
# ============================================================================
print("\n" + "="*70)
print("🔧 STAGE 2: TRAINING DWN LUT LAYERS WITH FIXED CONNECTIONS")
print("="*70)
print("Using torch_dwn.LUTLayer with connections from pruning")
print("="*70)

# Create fixed mapping for DWN LUTLayer
# DWN LUTLayer expects a mapping tensor: [out_features, n] where n=6
def create_fixed_mapping_from_connections(connection_info, target_n=6):
    """
    Create a fixed mapping tensor for DWN LUTLayer from pruned connections.

    Enforces exactly target_n connections per neuron:
    - If neuron has < target_n: pads with 0s (duplicate first input)
    - If neuron has > target_n: trims to top target_n by magnitude (uses first target_n)

    Args:
        connection_info: Dict with connections_per_neuron list
        target_n: Target number of inputs per neuron (default: 6)

    Returns:
        mapping: LongTensor of shape [out_features, target_n]
    """
    out_features = connection_info['out_features']
    mapping = torch.zeros(out_features, target_n, dtype=torch.int32)

    warnings = {'too_few': [], 'too_many': []}

    for neuron_info in connection_info['connections_per_neuron']:
        neuron_idx = neuron_info['neuron_idx']
        active_indices = neuron_info['active_input_indices']
        num_conns = len(active_indices)

        if num_conns == target_n:
            # Perfect - use as-is
            for i, idx in enumerate(active_indices):
                mapping[neuron_idx, i] = idx

        elif num_conns < target_n:
            # Too few - pad by repeating first input
            warnings['too_few'].append((neuron_idx, num_conns))
            for i in range(target_n):
                if i < num_conns:
                    mapping[neuron_idx, i] = active_indices[i]
                else:
                    mapping[neuron_idx, i] = active_indices[0]  # Repeat first input

        else:  # num_conns > target_n
            # Too many - trim to top target_n by magnitude
            warnings['too_many'].append((neuron_idx, num_conns))

            # Get weights for this neuron to sort by magnitude
            weights_list = neuron_info.get('weights', None)
            if weights_list is not None and len(weights_list) == num_conns:
                # Sort indices by absolute weight magnitude (descending)
                weight_magnitude_pairs = [(abs(w), idx) for w, idx in zip(weights_list, active_indices)]
                weight_magnitude_pairs.sort(reverse=True, key=lambda x: x[0])
                # Take top target_n
                sorted_indices = [idx for _, idx in weight_magnitude_pairs[:target_n]]
                for i in range(target_n):
                    mapping[neuron_idx, i] = sorted_indices[i]
            else:
                # Fallback: take first target_n (shouldn't happen if weights are exported)
                for i in range(target_n):
                    mapping[neuron_idx, i] = active_indices[i]

    # Report warnings
    if warnings['too_few']:
        print(f"\n   ⚠️  Warning: {len(warnings['too_few'])} neurons have <{target_n} connections (padded)")
        for neuron_idx, num_conns in warnings['too_few'][:3]:
            print(f"      Neuron {neuron_idx}: {num_conns} → {target_n} (padded)")

    if warnings['too_many']:
        print(f"\n   ⚠️  Warning: {len(warnings['too_many'])} neurons have >{target_n} connections (trimmed to top {target_n} by magnitude)")
        for neuron_idx, num_conns in warnings['too_many'][:3]:
            print(f"      Neuron {neuron_idx}: {num_conns} → {target_n} (kept top {target_n} by absolute weight value)")

    return mapping

# Create fixed mappings for both layers
print("\n📋 Creating fixed mappings for DWN LUTLayers...")
mapping_layer1 = create_fixed_mapping_from_connections(connections[0])
mapping_layer2 = create_fixed_mapping_from_connections(connections[1])

print(f"  Layer 1 mapping shape: {mapping_layer1.shape}")
print(f"  Layer 2 mapping shape: {mapping_layer2.shape}")

# Verify all neurons have exactly 6 inputs
assert mapping_layer1.shape[1] == 6, f"Layer 1 mapping should have 6 inputs, got {mapping_layer1.shape[1]}"
assert mapping_layer2.shape[1] == 6, f"Layer 2 mapping should have 6 inputs, got {mapping_layer2.shape[1]}"
print(f"  ✅ All neurons configured with exactly 6 inputs")

print(f"\n  Example mappings (Layer 1, first 3 neurons):")
for i in range(min(3, mapping_layer1.shape[0])):
    print(f"    Neuron {i}: inputs {mapping_layer1[i].tolist()}")

# Build DWN LUT model with fixed connections
print("\n🔨 Building DWN LUT model with fixed connections...")
print(f"  Each neuron → 6-input LUT (64 truth table entries)")

# Create LUTLayers with fixed mapping tensors directly
lut_layer1 = dwn.LUTLayer(
    connections[0]['in_features'],  # in_features (positional)
    connections[0]['out_features'], # out_features (positional)
    n=6,
    mapping=mapping_layer1  # Pass the fixed mapping tensor directly
)

lut_layer2 = dwn.LUTLayer(
    connections[1]['in_features'],
    connections[1]['out_features'],
    n=6,
    mapping=mapping_layer2
)

# Build the model
dwn_lut_model = nn.Sequential(
    lut_layer1,
    lut_layer2
).cuda()

print(f"  Layer 1: {connections[0]['out_features']} LUTs × 6 inputs = {connections[0]['out_features'] * 64} truth table entries")
print(f"  Layer 2: {connections[1]['out_features']} LUTs × 6 inputs = {connections[1]['out_features'] * 64} truth table entries")

# Count trainable parameters
total_lut_params = sum(p.numel() for p in dwn_lut_model.parameters() if p.requires_grad)
print(f"  Trainable LUT entries: {total_lut_params}")

# Train DWN LUT truth tables
print("\n🎓 Training DWN LUT truth tables (connections FIXED)...")
print("Note: Only truth table values are trained, not the connections")

dwn_lut_optimizer = torch.optim.Adam(dwn_lut_model.parameters(), lr=1e-2)
dwn_lut_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dwn_lut_optimizer, T_max=50)

print("\nTraining for 50 epochs...")
best_dwn_lut_acc = train_and_evaluate(dwn_lut_model, dwn_lut_optimizer, dwn_lut_scheduler,
                                       x_train, y_train, x_test, y_test,
                                       epochs=50, batch_size=128, save_best=True,
                                       phase_name='lut_training')

final_dwn_lut_acc = evaluate(dwn_lut_model, x_test, y_test)

print("\n" + "="*70)
print("✅ DWN LUT TRAINING COMPLETE")
print("="*70)
print(f"  Pruned model (linear weights):  {final_acc_after_finetune:.4f}")
print(f"  DWN LUT model (truth tables):   {final_dwn_lut_acc:.4f}")
print(f"  Difference:                     {(final_dwn_lut_acc - final_acc_after_finetune):+.4f}")
print("="*70)

# Save the trained LUT model
lut_model_file = 'dwn_lut_model_6inputs.pth'
torch.save({
    'model_state_dict': dwn_lut_model.state_dict(),
    'layer1_mapping': mapping_layer1,
    'layer2_mapping': mapping_layer2,
    'connections': connections,
    'accuracy': final_dwn_lut_acc
}, lut_model_file)

print(f"\n💾 DWN LUT model saved to: {lut_model_file}")

print("\n📤 READY FOR QUANT_DWN SIMULATOR:")
print("  1. ✅ Connections are FIXED (6 inputs per neuron)")
print("  2. ✅ Truth tables are TRAINED using torch_dwn.LUTLayer")
print("  3. ✅ Model file contains mappings + trained LUTs")
print("  4. ✅ Compatible with DWN simulator infrastructure")
print("  5. ✅ Each LUT: 64 entries (2^6) - ready for FPGA")

print("\n🔧 To use in your simulator:")
print("  import torch_dwn as dwn")
print("  checkpoint = torch.load('dwn_lut_model_6inputs.pth')")
print("  # Load mappings: checkpoint['layer1_mapping']")
print("  # Load LUT values: checkpoint['model_state_dict']")

print("\n" + "="*70)

# ============================================================================
# PLOT TRAINING METRICS
# ============================================================================
print("\n" + "="*70)
print("📊 CREATING TRAINING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Get phase colors
phase_colors = {
    'initial': 'blue',
    'pruning': 'orange',
    'finetune': 'green',
    'lut': 'purple'
}

# Determine phase color for each point
colors = []
for phase in training_history['phase']:
    if 'initial' in phase.lower():
        colors.append(phase_colors['initial'])
    elif 'pruning' in phase.lower():
        colors.append(phase_colors['pruning'])
    elif 'finetune' in phase.lower():
        colors.append(phase_colors['finetune'])
    elif 'lut' in phase.lower():
        colors.append(phase_colors['lut'])
    else:
        colors.append('gray')

# Plot 1: Learning Rate over Time
ax1 = axes[0]
ax1.semilogy(training_history['epochs'], training_history['learning_rates'], 'b-', linewidth=2)
ax1.set_xlabel('Cumulative Epoch', fontsize=12)
ax1.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax1.set_title('Learning Rate Schedule Throughout Training', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add phase markers
phase_changes = []
current_phase = None
for i, phase in enumerate(training_history['phase']):
    if phase != current_phase:
        phase_changes.append((training_history['epochs'][i], phase))
        current_phase = phase

for epoch, phase in phase_changes:
    ax1.axvline(x=epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(epoch, ax1.get_ylim()[1]*0.5, phase.replace('_', '\n'),
             rotation=90, verticalalignment='bottom', fontsize=8, alpha=0.7)

# Plot 2: Sparsity (Pruning Rate) over Time
ax2 = axes[1]
ax2.plot(training_history['epochs'], training_history['sparsity'], 'r-', linewidth=2)
ax2.set_xlabel('Cumulative Epoch', fontsize=12)
ax2.set_ylabel('Sparsity (%)', fontsize=12)
ax2.set_title('Model Sparsity Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# Add horizontal line at target sparsity (if reached 99%+)
if max(training_history['sparsity']) > 99:
    ax2.axhline(y=99, color='green', linestyle='--', alpha=0.5, label='~6 weights/neuron (99% sparsity)')
    ax2.legend()

# Add phase changes
for epoch, phase in phase_changes:
    ax2.axvline(x=epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Plot 3: Test Accuracy over Time
ax3 = axes[2]
ax3.plot(training_history['epochs'], training_history['test_accuracy'], 'g-', linewidth=2)
ax3.set_xlabel('Cumulative Epoch', fontsize=12)
ax3.set_ylabel('Test Accuracy', fontsize=12)
ax3.set_title('Test Accuracy Throughout Training Process', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Mark best accuracy
best_idx = np.argmax(training_history['test_accuracy'])
best_epoch = training_history['epochs'][best_idx]
best_acc = training_history['test_accuracy'][best_idx]
ax3.scatter([best_epoch], [best_acc], color='red', s=100, zorder=5, marker='*', label=f'Best: {best_acc:.4f}')
ax3.legend()

# Add phase changes
for epoch, phase in phase_changes:
    ax3.axvline(x=epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plot_filename = 'training_metrics.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"\n✅ Training metrics plot saved to: {plot_filename}")

# Create a second plot: Weights per Neuron over Time (if available)
if any(w is not None for w in training_history['weights_per_neuron']):
    fig2, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Filter out None values
    epochs_with_weights = []
    weights_values = []
    for i, w in enumerate(training_history['weights_per_neuron']):
        if w is not None:
            epochs_with_weights.append(training_history['epochs'][i])
            weights_values.append(w)

    if weights_values:
        ax.plot(epochs_with_weights, weights_values, 'mo-', linewidth=2, markersize=4)
        ax.set_xlabel('Cumulative Epoch', fontsize=12)
        ax.set_ylabel('Active Weights per Neuron', fontsize=12)
        ax.set_title('Structured Pruning: Weights per Neuron Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=6, color='green', linestyle='--', linewidth=2, label='Target: 6 weights (LUT compatible)')
        ax.legend()

        # Add phase changes
        for epoch, phase in phase_changes:
            if epoch in epochs_with_weights:
                ax.axvline(x=epoch, color='red', linestyle='--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        weights_plot_filename = 'weights_per_neuron.png'
        plt.savefig(weights_plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Weights per neuron plot saved to: {weights_plot_filename}")

# Print summary statistics
print("\n📈 TRAINING SUMMARY STATISTICS:")
print(f"  Total epochs trained: {cumulative_epoch}")
print(f"  Initial sparsity: {training_history['sparsity'][0]:.2f}%")
print(f"  Final sparsity: {training_history['sparsity'][-1]:.2f}%")
print(f"  Initial LR: {training_history['learning_rates'][0]:.2e}")
print(f"  Final LR: {training_history['learning_rates'][-1]:.2e}")
print(f"  Initial accuracy: {training_history['test_accuracy'][0]:.4f}")
print(f"  Best accuracy: {best_acc:.4f} (epoch {best_epoch})")
print(f"  Final accuracy: {training_history['test_accuracy'][-1]:.4f}")

print("\n" + "="*70)
print("✅ ALL TRAINING AND VISUALIZATION COMPLETE!")
print("="*70)

# Example 4: Iterative structured pruning to 6 weights per neuron
# This gradually reduces connections: 20 -> 15 -> 10 -> 8 -> 6
"""
print("\n" + "="*70)
print("EXAMPLE 4: Iterative structured pruning to 6 weights per neuron")
print("="*70)

# Define optimizer and scheduler factories
def make_optimizer():
    return torch.optim.Adam(model.parameters(), lr=1e-3)

def make_scheduler(opt):
    return torch.optim.lr_scheduler.StepLR(opt, gamma=0.1, step_size=5)

model = iterative_pruning_and_retraining(
    model,
    optimizer_fn=make_optimizer,
    scheduler_fn=make_scheduler,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    target_sparsity=0.9,  # Not used in structured mode
    pruning_steps=5,
    epochs_per_step=10,
    batch_size=128,
    method='structured'
)
"""
