# visualize.py
import matplotlib. pyplot as plt
import numpy as np

def plot_federated_results():
    """
    Federated Learning sonuçlarını görselleştir
    """
    rounds = [1, 2, 3, 4, 5]
    
    # Drone sonuçları (loglardan)
    drone1_acc = [88.33, 88.33, 75.00, 95.00, 93.33]
    drone2_acc = [83.33, 93.33, 88.33, 95.00, 96.67]
    drone3_acc = [96.67, 98.33, 88.33, 96.67, 90.00]
    global_acc = [90.00, 91.67, 97.22, 97.22, 97.22]
    
    # Loss values
    global_loss = [0.269, 0.185, 0.088, 0.063, 0.060]
    
    # Figure oluştur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Subplot 1: Accuracy comparison
    ax1.plot(rounds, drone1_acc, 'o-', label='Drone 1 (Sehir - 70% Guvenli)', linewidth=2, markersize=8)
    ax1.plot(rounds, drone2_acc, 's-', label='Drone 2 (Orman - 70% Tehlikeli)', linewidth=2, markersize=8)
    ax1.plot(rounds, drone3_acc, '^-', label='Drone 3 (Karma - 50-50)', linewidth=2, markersize=8)
    ax1.plot(rounds, global_acc, 'D-', label='Global Model (Federated)', linewidth=3, markersize=10, color='red')
    
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Federated Drone Emergency Landing\nAccuracy per Round', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([70, 100])
    ax1.set_xticks(rounds)
    
    # Subplot 2: Global Loss
    ax2.plot(rounds, global_loss, 'o-', linewidth=3, markersize=10, color='green')
    ax2.fill_between(rounds, global_loss, alpha=0.3, color='green')
    ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Global Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Global Model Loss Reduction', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds)
    
    # Annotation for final values
    ax1.annotate(f'Final:  {global_acc[-1]:.2f}%', 
                 xy=(rounds[-1], global_acc[-1]), 
                 xytext=(rounds[-1]-0.5, global_acc[-1]-5),
                 fontsize=11, fontweight='bold', color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax2.annotate(f'Final: {global_loss[-1]:.3f}', 
                 xy=(rounds[-1], global_loss[-1]), 
                 xytext=(rounds[-1]-0.7, global_loss[-1]+0.05),
                 fontsize=11, fontweight='bold', color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig('federated_results.png', dpi=300, bbox_inches='tight')  # BOŞLUK SİLİNDİ
    print("✅ Grafik kaydedildi:  federated_results.png")
    plt.show()

def plot_drone_comparison():
    """
    Drone'ların başlangıç vs son performanslarını karşılaştır
    """
    drones = ['Drone 1\n(Sehir)', 'Drone 2\n(Orman)', 'Drone 3\n(Karma)', 'Global\nModel']
    round1 = [88.33, 83.33, 96.67, 90.00]
    round5 = [93.33, 96.67, 90.00, 97.22]
    improvement = [r5 - r1 for r1, r5 in zip(round1, round5)]
    
    x = np.arange(len(drones))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, round1, width, label='Round 1', color='skyblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, round5, width, label='Round 5', color='orange', edgecolor='black', linewidth=1.5)
    
    # Improvement annotations
    for i, imp in enumerate(improvement):
        ax.text(i, max(round1[i], round5[i]) + 2, 
                f'{imp: +.1f}%', 
                ha='center', fontsize=11, fontweight='bold',
                color='green' if imp > 0 else 'red')
    
    ax.set_xlabel('Drone / Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Federated Learning Impact\nRound 1 vs Round 5', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drones)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([75, 105])
    
    plt.tight_layout()
    plt.savefig('drone_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Grafik kaydedildi: drone_comparison.png")
    plt.show()

def print_summary():
    """
    Proje özet raporu
    """
    print("\n" + "="*70)
    print(" "*15 + "FEDERATED DRONE PROJECT - OZET RAPOR")
    print("="*70)
    
    print("\n📦 DATASET:")
    print("   • ModelNet10 (3D mesh → point cloud)")
    print("   • 900 örnek (300 per drone)")
    print("   • Her nokta bulutu:  1024 nokta (X, Y, Z)")
    
    print("\n🧠 MODEL:")
    print("   • PointNet benzeri architecture")
    print("   • Parametreler: 801,282")
    print("   • Task: Binary classification (Guvenli/Tehlikeli inis)")
    
    print("\n🚁 DRONE DAGILIMI:")
    print("   • Drone 1 (Sehir): 70% guvenli, 30% tehlikeli")
    print("   • Drone 2 (Orman): 30% guvenli, 70% tehlikeli")
    print("   • Drone 3 (Karma): 50% guvenli, 50% tehlikeli")
    
    print("\n🌸 FEDERATED LEARNING:")
    print("   • Framework: Flower 1.25.0")
    print("   • Strategy: FedAvg")
    print("   • Rounds:  5")
    print("   • Epochs per round: 5")
    
    print("\n📊 SONUCLAR:")
    print("   • Round 1: 90.00% → Round 5: 97.22% (+7.22%)")
    print("   • Loss: 0.269 → 0.060 (-77.7%)")
    print("   • En iyi drone: Drone 2 (96.67%)")
    print("   • Global model: Tum drone'lardan daha iyi!")
    
    print("\n✅ BASARILAR:")
    print("   ✓ 3 drone veri paylasmadan ogrendi")
    print("   ✓ Global model convergence basarili")
    print("   ✓ Gercekci 3D point cloud kullanildi")
    print("   ✓ MacBook M serisi MPS acceleration")
    
    print("\n🎯 KULLANIM ALANLARI:")
    print("   • Drone acil inis sistemleri")
    print("   • Otonom arac park yeri tespiti")
    print("   • Robotik navigasyon")
    print("   • LiDAR tabanli guvenlik sistemleri")
    
    print("\n" + "="*70)
    print(" "*20 + "🎉 PROJE BASARIYLA TAMAMLANDI!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("📊 Gorsellestirmeler olusturuluyor.. .\n")
    
    # Grafikler
    plot_federated_results()
    plot_drone_comparison()
    
    # Özet rapor
    print_summary()
    
    print("📁 Olusturulan dosyalar:")
    print("   • federated_results.png")
    print("   • drone_comparison.png")