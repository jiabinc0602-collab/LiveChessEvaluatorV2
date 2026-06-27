from representation import ChessDataset
from resnet import ChessResNet
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChessResNet().to(device)
dataset = ChessDataset("data/output/evaluated_positions.jsonl", use_attack_map=False)

total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataset = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)
val_dataset = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
test_dataset = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

# hyper params
epochs = 10
batch_size = 2048
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.one_cycle.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1000, epochs=epochs)

for k in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (inputs, targets) in enumerate(train_dataset):
        inputs, targets = inputs.to(device), targets.to(device)

        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs.view(-1), targets)
        
        #backpropagation
        loss.backward()

        # update weights
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f"[{k + 1}, {i + 1}] loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    print(f"Finished epoch {k + 1}")
print("Finished Training")

model.eval()
total_val_loss = 0.0
num_batches = 0
with torch.no_grad():
    for inputs, targets in val_dataset:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), targets)
        total_val_loss += loss.item()
        num_batches += 1


print(f"Total validation loss: {total_val_loss / num_batches:.5f}")
