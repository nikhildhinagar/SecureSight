import torch
from tqdm import tqdm

def train(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if scaler:
            # Use generic autocast for cross-platform support (MPS/CUDA)
            device_type = "cuda" if device.type == "cuda" else "cpu"
            if device.type == "mps":
                device_type = "mps"
                
            with torch.autocast(device_type=device_type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    print(f"Training Loss: {total_loss / len(loader)}")

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / len(loader.dataset)
    print(f"Validation Loss: {total_loss / len(loader)}, Accuracy: {accuracy * 100:.2f}%")

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")