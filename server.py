# server.py
import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import Metrics, Parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
import numpy as np

def priority_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Priority-aware weighted averaging
    """
    total_examples = 0
    weighted_acc = 0.0
    
    for num_examples, m in metrics: 
        # Priority weight
        priority_weights = {"HIGH": 2.0, "MEDIUM": 1.5, "LOW": 1.0}
        priority = m.get("priority", "LOW")
        priority_weight = priority_weights. get(priority, 1.0)
        
        # Network quality weight
        network_quality = m.get("network_quality", 1.0)
        
        # Combined weight
        combined_weight = priority_weight * network_quality * num_examples
        
        weighted_acc += m["accuracy"] * combined_weight
        total_examples += combined_weight
    
    return {"accuracy": weighted_acc / total_examples if total_examples > 0 else 0}

def fit_config(server_round: int) -> Dict: 
    """Her round için config"""
    return {
        "server_round": server_round,
        "local_epochs": 7,
    }

class PriorityFedAvg(fl.server.strategy.FedAvg):
    """
    Priority-aware FedAvg strategy
    """
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common. FitRes]],
        failures: List[Tuple[ClientProxy, fl. common.FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate with priority awareness and error handling"""
        
        print(f"\n📊 Round {server_round} Aggregation:")
        print(f"   ✅ Success: {len(results)} drones")
        print(f"   ❌ Failures: {len(failures)} drones")
        
        # Başarılı sonuçları analiz et ve boş olanları filtrele
        valid_results = []
        for client_proxy, fit_res in results:  
            metrics = fit_res.metrics
            drone_id = metrics. get("drone_id", "? ")
            priority = metrics.get("priority", "?")
            test_acc = metrics.get("test_acc", 0)
            skipped = metrics.get("skipped", False)
            
            # Parametrelerin boş olup olmadığını kontrol et
            if fit_res.parameters and len(fit_res.parameters. tensors) > 0:
                valid_results.append((client_proxy, fit_res))
                
                if skipped:
                    print(f"   ⚠️  Drone {drone_id}:  SKIPPED (connection issue)")
                else:
                    print(f"   🚁 Drone {drone_id} ({priority}): {test_acc:.2f}%")
            else:
                print(f"   ❌ Drone {drone_id}: Empty parameters (skipped in aggregation)")
        
        # Eğer hiç valid result yoksa, None döndür
        if not valid_results:
            print("   ⚠️  No valid results to aggregate!")
            return None, {}
        
        # Parent class aggregation - sadece valid results ile
        return super().aggregate_fit(server_round, valid_results, failures)

def main():
    """Flower Server - Priority-aware FL"""
    print("🌸 Flower Federated Learning Server (Network-Aware)")
    print("="*60)
    print("🎯 5 Drone + Network Challenges")
    print("="*60)
    print("\n📋 Drone Priorities:")
    print("   HIGH:     Drone 3 (Orman), Drone 4 (Dağlık)")
    print("   MEDIUM:  Drone 2 (Sanayi)")
    print("   LOW:    Drone 1 (Şehir), Drone 5 (Karma)")
    print("\n🌐 Network Conditions:")
    print("   Drone 3: 40% packet loss (worst)")
    print("   Drone 4: 35% packet loss")
    print("   Drone 2: 15% packet loss")
    print("   Drone 5: 8% packet loss")
    print("   Drone 1: 5% packet loss (best)")
    print("="*60)
    
    # Priority-aware FedAvg strategy
    strategy = PriorityFedAvg(
        fraction_fit=0.8,  # En az %80'i katılsın
        fraction_evaluate=0.8,
        min_fit_clients=3,  # Minimum 3 drone
        min_evaluate_clients=3,
        min_available_clients=5,  # 5 drone başta hazır olsun
        evaluate_metrics_aggregation_fn=priority_weighted_average,
        on_fit_config_fn=fit_config,
    )
    
    # Server'ı başlat
    print("\n🚀 Server başlatılıyor...")
    print("📡 Adres: 127.0.0.1:8080")
    print("⏳ 5 drone'un bağlanması bekleniyor.. .\n")
    
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )
    
    print("\n" + "="*60)
    print("🎉 Federated Learning tamamlandı!")
    print("="*60)

if __name__ == "__main__":
    main()