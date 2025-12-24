# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class DronePointCloudDataset(Dataset):
    """
    Drone için point cloud dataset
    Label 0: Güvenli iniş alanı
    Label 1: Tehlikeli iniş alanı
    """
    def __init__(self, drone_id, train=True, train_split=0.8):
        self.drone_id = drone_id
        self.data_dir = f"data/drone{drone_id}"
        
        # Dosyaları yükle
        safe_files = sorted(Path(self.data_dir).glob("safe_*.npy"))
        unsafe_files = sorted(Path(self.data_dir).glob("unsafe_*.npy"))
        
        # Train/Test split
        safe_split = int(len(safe_files) * train_split)
        unsafe_split = int(len(unsafe_files) * train_split)
        
        if train:
            self.safe_files = safe_files[:safe_split]
            self.unsafe_files = unsafe_files[:unsafe_split]
        else:
            self.safe_files = safe_files[safe_split:]
            self.unsafe_files = unsafe_files[unsafe_split:]
        
        # Tüm dosyalar ve labellar
        self.files = list(self.safe_files) + list(self.unsafe_files)
        self.labels = [0] * len(self.safe_files) + [1] * len(self.unsafe_files)
        
        print(f" Drone {drone_id} {'Train' if train else 'Test'}:  "
              f"{len(self. safe_files)} güvenli + {len(self.unsafe_files)} tehlikeli = {len(self)} toplam")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Point cloud yükle
        points = np.load(self.files[idx])
        label = self.labels[idx]
        
        # Tensor'a çevir
        points = torch. from_numpy(points).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return points, label

def get_dataloaders(drone_id, batch_size=32, train_split=0.8):
    """
    Drone için train ve test dataloader'ları oluştur
    """
    train_dataset = DronePointCloudDataset(drone_id, train=True, train_split=train_split)
    test_dataset = DronePointCloudDataset(drone_id, train=False, train_split=train_split)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MacOS için 0 daha stabil
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__": 
    print(" Dataset Test\n")
    print("="*60)
    
    # Her drone için dataset'i test et
    for drone_id in [1, 2, 3]:
        print(f"\nDrone {drone_id}:")
        train_loader, test_loader = get_dataloaders(drone_id, batch_size=16)
        
        # İlk batch'i al
        points, labels = next(iter(train_loader))
        
        print(f"   Batch shape: {points.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Label distribution: {labels.sum().item()} unsafe, {len(labels)-labels.sum().item()} safe")
    
    print("\n" + "="*60)
    print(" Dataset başarıyla yüklendi!")