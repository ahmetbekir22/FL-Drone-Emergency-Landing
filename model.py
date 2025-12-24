# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetBackbone(nn.Module):
    """
    PointNet feature extractor
    Input: [B, N, 3] - Batch, NumPoints, XYZ
    Output: [B, 1024] - Global feature vector
    """
    def __init__(self):
        super(PointNetBackbone, self).__init__()
        
        # Shared MLPs
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        # x:  [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self. conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]  # [B, 1024]
        
        return x

class PointNetClassifier(nn.Module):
    """
    PointNet for binary classification (Safe/Unsafe landing)
    """
    def __init__(self, num_classes=2):
        super(PointNetClassifier, self).__init__()
        
        self.backbone = PointNetBackbone()
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn. Dropout(p=0.3)
    
    def forward(self, x):
        # Extract global features
        x = self.backbone(x)
        
        # Classification
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def get_model():
    """Create and return the model"""
    return PointNetClassifier(num_classes=2)

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__": 
    # Test the model
    model = get_model()
    print(" Model Mimarisi:")
    print(model)
    print(f"\n Toplam Parametreler: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 1024, 3)  # Batch=4, Points=1024, XYZ=3
    output = model(dummy_input)
    print(f"\n Test başarılı!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")