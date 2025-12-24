# client.py
import torch
import flwr as fl
import time
import random
import numpy as np
from collections import OrderedDict
from model import get_model
from dataset import get_dataloaders
from train import train_model, test

# Drone network profilleri
DRONE_PROFILES = {
    1: {
        "name": "Sehir Merkezi",
        "priority": "LOW",
        "packet_loss": 0.05,  # 5%
        "latency_range": (0.1, 0.5),  # saniye
        "disconnect_prob": 0.01  # 1%
    },
    2: {
        "name": "Sanayi Bolgesi",
        "priority": "MEDIUM",
        "packet_loss": 0.15,  # 15%
        "latency_range": (0.3, 1.0),
        "disconnect_prob": 0.05
    },
    3: {
        "name": "Orman (KRITIK)",
        "priority": "HIGH",
        "packet_loss": 0.40,  # 40% - Çok kötü! 
        "latency_range":  (1.0, 3.0),
        "disconnect_prob": 0.15  # 15%
    },
    4: {
        "name": "Daglik Alan",
        "priority": "HIGH",
        "packet_loss":  0.35,  # 35%
        "latency_range":  (0.8, 2.5),
        "disconnect_prob":  0.12
    },
    5: {
        "name": "Karma/Test",
        "priority": "LOW",
        "packet_loss": 0.08,  # 8%
        "latency_range": (0.2, 0.7),
        "disconnect_prob":  0.02
    }
}

class NetworkSimulator:
    """Network koşullarını simüle et"""
    
    def __init__(self, drone_id):
        self.profile = DRONE_PROFILES[drone_id]
        self.drone_id = drone_id
    
    def simulate_latency(self):
        """Rastgele gecikme ekle"""
        min_lat, max_lat = self.profile["latency_range"]
        latency = random.uniform(min_lat, max_lat)
        print(f"    Network latency: {latency:.2f}s")
        time.sleep(latency)
    
    def check_packet_loss(self):
        """Paket kaybı kontrolü"""
        if random.random() < self.profile["packet_loss"]:
            print(f"   ⚠️  Paket kaybı!  ({self.profile['packet_loss']*100:.0f}% şansı)")
            return True
        return False
    
    def check_disconnection(self):
        """Bağlantı kesintisi kontrolü"""
        if random.random() < self.profile["disconnect_prob"]:
            print(f"    Bağlantı kesildi!  Yeniden bağlanılıyor...")
            time.sleep(random.uniform(2, 5))  # Reconnect süresi
            return True
        return False
    
    def get_priority_weight(self):
        """Öncelik ağırlığı"""
        priority_weights = {"HIGH": 2.0, "MEDIUM": 1.5, "LOW": 1.0}
        return priority_weights[self.profile["priority"]]

class DroneClient(fl.client.NumPyClient):
    """
    Flower Client - Network challenges ile
    """
    def __init__(self, drone_id, epochs_per_round=7):
        self.drone_id = drone_id
        self. epochs_per_round = epochs_per_round
        self.network = NetworkSimulator(drone_id)
        
        # Device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Model
        self.model = get_model().to(self.device)
        
        # Data
        self.train_loader, self.test_loader = get_dataloaders(
            drone_id=drone_id,
            batch_size=16
        )
        
        profile = DRONE_PROFILES[drone_id]
        print(f" Drone {drone_id} ({profile['name']}) - Priority: {profile['priority']}")
        print(f"   Network: Loss={profile['packet_loss']*100:.0f}%, "
              f"Latency={profile['latency_range'][0]}-{profile['latency_range'][1]}s")
    
    def get_parameters(self, config):
        """Model parametrelerini döndür"""
        # Network latency simülasyonu
        self.network.simulate_latency()
        
        # Paket kaybı kontrolü
        if self.network.check_packet_loss():
            # Paket kayıpsa boş liste döndür (retry gerekecek)
            return []
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Server'dan gelen parametreleri modele yükle"""
        if not parameters:  # Boş gelirse skip
            print(f"   ⚠️  Parametre alınamadı, eski model kullanılıyor")
            return
        
        params_dict = zip(self.model. state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Training round"""
        print(f"\n Drone {self.drone_id} ({DRONE_PROFILES[self. drone_id]['name']}) - Training başlıyor...")
        
        # Bağlantı kesintisi kontrolü
        if self.network. check_disconnection():
            print(f"    Round atlandı (bağlantı sorunu)")
            # Eski parametreleri döndür
            return self.get_parameters(config={}), 0, {"drone_id": self.drone_id, "skipped": True}
        
        # Server'dan gelen parametreleri yükle
        self.set_parameters(parameters)
        
        # Priority'ye göre epoch ayarla
        priority_weight = self.network.get_priority_weight()
        adjusted_epochs = int(self.epochs_per_round * priority_weight)
        
        print(f"    Priority: {DRONE_PROFILES[self.drone_id]['priority']} "
              f"→ {adjusted_epochs} epochs (base: {self.epochs_per_round})")
        
        # Train
        history, best_acc = train_model(
            self.model,
            self.train_loader,
            self.test_loader,
            epochs=adjusted_epochs,
            lr=0.001,
            device=self.device
        )
        
        # Network latency (upload)
        print(f"    Model uploading...")
        self.network. simulate_latency()
        
        # Paket kaybı varsa retry
        retry_count = 0
        while self.network.check_packet_loss() and retry_count < 3:
            print(f"    Retry {retry_count+1}/3...")
            time.sleep(1)
            retry_count += 1
        
        # Güncel parametreleri döndür
        updated_parameters = self.get_parameters(config={})
        
        # İstatistikler
        num_examples = len(self.train_loader.dataset)
        metrics = {
            "drone_id": self.drone_id,
            "priority": DRONE_PROFILES[self.drone_id]['priority'],
            "train_acc": history['train_acc'][-1],
            "test_acc": history['test_acc'][-1],
            "network_quality": 1.0 - DRONE_PROFILES[self.drone_id]['packet_loss']
        }
        
        print(f" Drone {self.drone_id} - Training tamamlandı!  Test Acc: {history['test_acc'][-1]:.2f}%")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters, config):
        """Evaluation round"""
        # Bağlantı kontrolü
        if self.network. check_disconnection():
            return float('inf'), 0, {"accuracy": 0.0}
        
        # Server'dan gelen parametreleri yükle
        self.set_parameters(parameters)
        
        # Test
        import torch. nn as nn
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = test(self.model, self.test_loader, criterion, self.device)
        
        num_examples = len(self.test_loader.dataset)
        
        return test_loss, num_examples, {"accuracy": test_acc}

def start_client(drone_id, server_address="127.0.0.1:8080"):
    """Client'ı başlat"""
    client = DroneClient(drone_id=drone_id, epochs_per_round=7)
    
    # Başlangıç gecikmesi (tüm drone'lar aynı anda başlamasın)
    startup_delay = random.uniform(0, 3)
    print(f" Başlangıç gecikmesi: {startup_delay:.1f}s")
    time.sleep(startup_delay)
    
    fl.client.start_client(
        server_address=server_address,
        client=client. to_client()
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(" Kullanım: python client.py <drone_id>")
        print("   Örnek: python client.py 1")
        print("   Drone ID:   1-5 arası")
        print("\n Drone Profilleri:")
        for did, profile in DRONE_PROFILES. items():
            print(f"   Drone {did}: {profile['name']} ({profile['priority']} priority)")
        sys.exit(1)
    
    drone_id = int(sys.argv[1])
    
    if drone_id not in [1, 2, 3, 4, 5]:
        print(f" Geçersiz drone_id: {drone_id}. 1-5 arası olmalı.")
        sys.exit(1)
    
    print(f" Drone {drone_id} Client başlatılıyor...")
    start_client(drone_id)