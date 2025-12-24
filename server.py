# server.py
import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average for aggregating metrics
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int) -> Dict: 
    """
    Her round için config
    """
    return {
        "server_round": server_round,
        "local_epochs": 5,
    }

def main():
    """
    Flower Server - Federated Learning koordinatörü
    """
    print(" Flower Federated Learning Server")
    print("="*60)
    print(" 5 Drone için Federated Emergency Landing System")
    print("="*60)
    
    # FedAvg stratejisi
    strategy = fl. server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,  # 5 drone
        min_evaluate_clients=5,
        min_available_clients=5,  # 5 drone hazır olana kadar bekle
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
    )
    
    # Server'ı başlat
    print("\n Server başlatılıyor...")
    print(" Adres: 127.0.0.1:8080")
    print(" 5 drone'un bağlanması bekleniyor.. .\n")
    
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )
    
    print("\n" + "="*60)
    print(" Federated Learning tamamlandı!")
    print("="*60)

if __name__ == "__main__":
    main()