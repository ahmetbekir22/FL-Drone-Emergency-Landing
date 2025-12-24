# client.py
import torch
import flwr as fl
from collections import OrderedDict
from model import get_model
from dataset import get_dataloaders
from train import train_model, test

class DroneClient(fl.client.NumPyClient):
    """
    Flower Client - Her drone bir client
    """
    def __init__(self, drone_id, epochs_per_round=5):
        self.drone_id = drone_id
        self.epochs_per_round = epochs_per_round
        
        # Device
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Model
        self.model = get_model().to(self.device)
        
        # Data
        self.train_loader, self.test_loader = get_dataloaders(
            drone_id=drone_id,
            batch_size=16
        )
        
        print(f" Drone {drone_id} Client hazır (Device: {self.device})")
    
    def get_parameters(self, config):
        """Model parametrelerini döndür"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Server'dan gelen parametreleri modele yükle"""
        params_dict = zip(self.model. state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Training round
        Server'dan parametreler al, eğit, geri gönder
        """
        print(f"\n Drone {self.drone_id} - Training başlıyor...")
        
        # Server'dan gelen parametreleri yükle
        self.set_parameters(parameters)
        
        # Train
        history, best_acc = train_model(
            self. model,
            self.train_loader,
            self.test_loader,
            epochs=self. epochs_per_round,
            lr=0.001,
            device=self.device
        )
        
        # Güncel parametreleri döndür
        updated_parameters = self.get_parameters(config={})
        
        # İstatistikleri döndür
        num_examples = len(self.train_loader.dataset)
        metrics = {
            "drone_id": self.drone_id,
            "train_acc": history['train_acc'][-1],
            "test_acc": history['test_acc'][-1]
        }
        
        print(f" Drone {self.drone_id} - Training tamamlandı! Test Acc: {history['test_acc'][-1]:.2f}%")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluation round
        Server'dan parametreler al, test et
        """
        # Server'dan gelen parametreleri yükle
        self.set_parameters(parameters)
        
        # Test
        import torch. nn as nn
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = test(self.model, self.test_loader, criterion, self.device)
        
        num_examples = len(self.test_loader.dataset)
        
        return test_loss, num_examples, {"accuracy": test_acc}

def start_client(drone_id, server_address="127.0.0.1:8080"):
    """
    Client'ı başlat
    """
    client = DroneClient(drone_id=drone_id, epochs_per_round=5)
    fl.client.start_client(
        server_address=server_address,
        client=client. to_client()
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("   Kullanım: python client.py <drone_id>")
        print("   Örnek: python client.py 1")
        sys.exit(1)
    
    drone_id = int(sys.argv[1])
    
    if drone_id not in [1, 2, 3]:
        print(f" Geçersiz drone_id: {drone_id}. 1, 2 veya 3 olmalı.")
        sys.exit(1)
    
    print(f" Drone {drone_id} Client başlatılıyor...")
    start_client(drone_id)