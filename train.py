# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Bir epoch eğitim
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for points, labels in pbar:
        points, labels = points.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        
        # Backward
        loss. backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels. size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress bar update
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def test(model, test_loader, criterion, device):
    """
    Test/validation
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for points, labels in test_loader: 
            points, labels = points. to(device), labels.to(device)
            
            outputs = model(points)
            loss = criterion(outputs, labels)
            
            running_loss += loss. item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Model eğitimi
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler. StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Save history
        history['train_loss']. append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
    
    return history, best_acc

if __name__ == "__main__":
    from model import get_model
    from dataset import get_dataloaders
    
    print(" Training Test - Drone 1")
    print("="*60)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"📱 Device: {device}\n")
    
    # Model ve data
    model = get_model().to(device)
    train_loader, test_loader = get_dataloaders(drone_id=1, batch_size=16)
    
    # Train for 3 epochs (test için)
    print("Training başlıyor.. .\n")
    history, best_acc = train_model(
        model, 
        train_loader, 
        test_loader, 
        epochs=3, 
        lr=0.001, 
        device=device
    )
    
    print(f"\n En iyi test accuracy: {best_acc:.2f}%")