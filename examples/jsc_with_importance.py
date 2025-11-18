"""
JSC Dataset Classification with Importance-Weighted Training
Two-phase approach:
  Phase 1: Train normally to learn initial LUT mappings
  Phase 2: Use importance weighting to focus on important thresholds
"""
import torch
from torch import nn
from torch.nn.functional import cross_entropy
import torch_dwn as dwn
from torch_dwn import EncoderLayer
import openml
import numpy as np
from sklearn.model_selection import train_test_split

def evaluate(model, x_test, y_test, device):
    """Evaluate model accuracy on test set"""
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_epoch(model, optimizer, x_train, y_train, batch_size, device):
    """Train for one epoch"""
    model.train()
    n_samples = x_train.shape[0]
    permutation = torch.randperm(n_samples)
    total_loss = 0

    for i in range(0, n_samples, batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)

        outputs = model(batch_x)
        loss = cross_entropy(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

if __name__ == "__main__":
    print("="*70)
    print("JSC Classification with Importance-Weighted Training")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ========================================================================
    # Load and preprocess data
    # ========================================================================
    print("\nLoading JSC dataset from OpenML (ID: 42468)...")
    dataset = openml.datasets.get_dataset(42468)
    df_features, df_labels, _, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )

    features = df_features.values.astype(np.float32)
    label_names = list(df_labels.unique())
    labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)
    num_classes = labels.max() + 1

    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features, {num_classes} classes")

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, train_size=0.8, random_state=42
    )

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # Preprocess: clip and normalize
    sigma = 3.5
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True)
    clip_min = mean - sigma * std
    clip_max = mean + sigma * std

    x_train = torch.max(torch.min(x_train, clip_max), clip_min)
    x_test = torch.max(torch.min(x_test, clip_max), clip_min)

    # Normalize to [-1, 1)
    x_min = x_train.min(dim=0, keepdim=True)[0]
    x_max = x_train.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)

    x_train = 2 * (x_train - x_min) / x_range - 1
    x_test = 2 * (x_test - x_min) / x_range - 1

    print(f"Preprocessed: range [{x_train.min().item():.4f}, {x_train.max().item():.4f}]")

    # ========================================================================
    # Build model
    # ========================================================================
    num_features = x_train.size(1)
    thermometer_bits = 200
    hidden_size = 100
    batch_size = 128

    print(f"\nBuilding model:")
    print(f"  EncoderLayer: {num_features} features × {thermometer_bits} bits = {num_features * thermometer_bits} total")
    print(f"  LUTLayer: {hidden_size} neurons (6-input LUTs)")

    model = nn.Sequential(
        EncoderLayer(num_features, thermometer_bits, input_dataset=x_train,
                     estimator_type='finite_difference'),
        nn.Flatten(start_dim=1),
        dwn.LUTLayer(num_features * thermometer_bits, hidden_size, n=6, mapping='random'),
        dwn.GroupSum(k=num_classes, tau=1/0.3)
    ).cuda(device)

    # Get reference to encoder and lut_layer for later
    encoder = model[0]
    lut_layer = model[2]

    # Enable threshold learning
    encoder.thresholds.requires_grad = True

    # ========================================================================
    # PHASE 1: Initial training WITHOUT importance weighting
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Training without importance weighting (15 epochs)")
    print("="*70)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)

    best_acc = 0.0
    for epoch in range(15):
        total_loss = train_epoch(model, optimizer, x_train, y_train, batch_size, device)
        scheduler.step()

        test_acc = evaluate(model, x_test, y_test, device)
        best_acc = max(best_acc, test_acc)

        print(f'Epoch {epoch+1:2d}/15 | Loss: {total_loss:.4f} | '
              f'Test Acc: {test_acc:.4f} | Best: {best_acc:.4f}')

    print(f"\n✓ Phase 1 complete. Best accuracy: {best_acc:.4f}")

    # ========================================================================
    # Extract importance from trained LUT mapping
    # ========================================================================
    print("\n" + "="*70)
    print("Extracting bit importance from LUTLayer mapping...")
    print("="*70)

    encoder.set_importance_from_lut(lut_layer, method='weighted')

    # Plot importance visualization
    from plot_importance import plot_importance_analysis
    plot_importance_analysis(encoder, lut_layer, save_path='jsc_importance_phase1.png')

    # Analyze importance
    importance = encoder.importance_weights
    print(f"\nImportance statistics:")
    print(f"  - Mean: {importance.mean():.4f}")
    print(f"  - Std: {importance.std():.4f}")
    print(f"  - Zero importance bits: {(importance == 0).sum().item()} / {importance.numel()}")
    print(f"  - Coverage: {100 * (importance > 0).float().mean():.1f}% of bits are connected")

    # ========================================================================
    # PHASE 2: Continue training WITH importance weighting
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Training WITH importance weighting (15 more epochs)")
    print("="*70)
    print("Threshold gradients now weighted by bit importance!")
    print("Important thresholds will learn faster, unused thresholds won't update.\n")

    # Reset optimizer and scheduler for phase 2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10)

    phase2_best_acc = test_acc  # Start from end of phase 1
    for epoch in range(15):
        total_loss = train_epoch(model, optimizer, x_train, y_train, batch_size, device)
        scheduler.step()

        test_acc = evaluate(model, x_test, y_test, device)
        phase2_best_acc = max(phase2_best_acc, test_acc)

        print(f'Epoch {epoch+1:2d}/15 | Loss: {total_loss:.4f} | '
              f'Test Acc: {test_acc:.4f} | Best: {phase2_best_acc:.4f}')

    # ========================================================================
    # Final summary
    # ========================================================================
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Phase 1 (no importance): Best = {best_acc:.4f}")
    print(f"Phase 2 (with importance): Best = {phase2_best_acc:.4f}")
    print(f"Improvement: {(phase2_best_acc - best_acc)*100:.2f}%")
    print("="*70)
