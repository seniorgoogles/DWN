import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch_dwn as dwn
import torch.nn.functional as F

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((32, 32)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(32)
    ],p=0.3)
])

transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((32, 32)),
])
    
def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = (model(x_test.cuda(device)).cpu()).argmax(dim=1).numpy()
        acc = (pred == y_test.numpy()).sum() / y_test.shape[0]
    return acc

def train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs, batch_size):
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct_train = 0
        total_train = 0
        
        for i in range(0, n_samples, batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x_train[indices].cuda(device), y_train[indices].cuda(device)
            outputs = model(batch_x)
            loss = F.cross_entropy(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            pred_train = outputs.argmax(dim=1)
            correct_train += (pred_train == batch_y).sum().item()
            total_train += batch_y.size(0)
        
        train_acc = correct_train / total_train
        
        scheduler.step()
        
        test_acc = evaluate(model, x_test, y_test)
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        



# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

x_train, y_train = next(iter(train_loader))
x_test, y_test = next(iter(test_loader))

# Binarize with distributive thermometer
#thermometer = dwn.DistributiveThermometer(10).fit(x_train)
#x_train = thermometer.binarize(x_train).flatten(start_dim=1)
#x_test = thermometer.binarize(x_test).flatten(start_dim=1)

print(x_train.shape)

factor = 10
min_size = 2

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(x_train.size(1), min_size*factor),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(min_size*factor, min_size*factor//2),
    nn.ReLU(),
    nn.Linear(min_size*factor//2, 10),
    nn.Softmax(dim=1))

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=14)

train_and_evaluate(model, optimizer, scheduler, x_train, y_train, x_test, y_test, epochs=1000, batch_size=256)
