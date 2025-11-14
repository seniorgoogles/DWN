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

print("="*70)
print("JSC Dataset Classification with DWN LUTLayers")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

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
# hidden_size = 16030 to match VC dimension of jsc_default.py (1,025,920 parameters)
# Calculation: LUTLayer params = hidden_size × 2^n = 16030 × 64 = 1,025,920
hidden_size = 10

model = nn.Sequential(
    dwn.LUTLayer(x_train.size(1), hidden_size, n=10, mapping='learnable'),
    dwn.GroupSum(k=num_classes, tau=1/0.3)
)

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

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size):
    """Train model and evaluate after each epoch"""
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

        print(f'Epoch {epoch + 1:2d}/{epochs} | '
              f'Loss: {loss.item():.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Test Acc: {test_acc:.4f}')

    print("="*70)
    print("Training complete!")
    print("="*70)

# Train the model
train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test,
                   epochs=30, batch_size=128)
