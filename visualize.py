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
    print(" Grafik kaydedildi:  federated_results.png")
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
        ax.text(
            i,
            max(round1[i], round5[i]) + 2,
            f'{imp:+.1f}%',  # Corrected format specifier
            ha='center',
            fontsize=11,
            fontweight='bold',
            color='green' if imp > 0 else 'red'
        )
    
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



def plot_network_challenges():
    """
    5 Drone + Network challenges sonuçları
    """
    rounds = [1, 2, 3, 4, 5, 6]
    
    # Global accuracy (server logs'tan)
    global_acc = [93.75, 96.88, 91.67, 96.88, 98.33, 98.33]
    global_loss = [0.238, 0.073, np.nan, 0.100, np.nan, np.nan]
    
    # Network events
    network_events = {
        1: ["Drone 3: Empty params", "Drone 4: Disconnected"],
        2: ["All successful"],
        3: ["Drone 5: Empty params"],
        4: ["Drone 3: Empty params"],
        5: ["Drone 4: Disconnected"],
        6: ["All 100% accuracy! "]
    }
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # === Plot 1: Accuracy with network events ===
    ax1.plot(rounds, global_acc, 'o-', linewidth=3, markersize=12, 
             color='#2ecc71', label='Global Model')
    
    # Highlight problematic rounds
    problem_rounds = [1, 3, 4, 5]
    for r in problem_rounds:
        ax1.axvline(x=r, color='red', alpha=0.2, linestyle='--', linewidth=2)
    
    # Perfect round
    ax1.axvline(x=6, color='gold', alpha=0.3, linestyle='--', linewidth=3)
    
    # Annotations
    for i, (r, acc) in enumerate(zip(rounds, global_acc)):
        ax1.annotate(f'{acc:.1f}%', 
                    xy=(r, acc), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='darkgreen')
    
    # Network events text
    for r, events in network_events.items():
        y_pos = 88 if r in problem_rounds else 102
        color = 'red' if r in problem_rounds else 'green'
        for i, event in enumerate(events):
            ax1.text(r, y_pos - i*2, event, 
                    ha='center', fontsize=8, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
    
    ax1.set_xlabel('Federated Round', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Global Accuracy (%)', fontsize=13, fontweight='bold')
    ax1.set_title('5 Drone Federated Learning with Network Challenges\nAccuracy Despite Connection Issues', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([85, 105])
    ax1.set_xticks(rounds)
    ax1.legend(fontsize=11)
    
    # === Plot 2: Drone profiles ===
    drone_names = ['Drone 1\n(Sehir)', 'Drone 2\n(Sanayi)', 'Drone 3\n(Orman)', 
                   'Drone 4\n(Daglik)', 'Drone 5\n(Karma)']
    packet_loss = [5, 15, 40, 35, 8]
    priorities = ['LOW', 'MEDIUM', 'HIGH', 'HIGH', 'LOW']
    colors = ['#3498db', '#f39c12', '#e74c3c', '#e74c3c', '#3498db']
    
    bars = ax2.barh(drone_names, packet_loss, color=colors, edgecolor='black', linewidth=2)
    
    # Priority labels
    for i, (bar, priority) in enumerate(zip(bars, priorities)):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{priority}\n{width}% loss',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Packet Loss (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Drone Network Conditions & Priorities', fontsize=15, fontweight='bold')
    ax2.set_xlim([0, 50])
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('network_challenges.png', dpi=300, bbox_inches='tight')
    print("✅ Grafik kaydedildi:  network_challenges.png")
    plt.show()

def plot_priority_impact():
    """
    Priority system'in etkisi
    """
    drones = ['Drone 1\n(LOW)', 'Drone 2\n(MEDIUM)', 'Drone 3\n(HIGH)', 
              'Drone 4\n(HIGH)', 'Drone 5\n(LOW)']
    
    base_epochs = [7, 7, 7, 7, 7]
    adjusted_epochs = [7, 10.5, 14, 14, 7]  # Priority multipliers:  1.0, 1.5, 2.0
    
    x = np.arange(len(drones))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, base_epochs, width, label='Base Epochs', color='lightblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, adjusted_epochs, width, label='Priority-Adjusted Epochs', 
                   color='orange', edgecolor='black', linewidth=1.5)
    
    # Annotations
    for i, (base, adj) in enumerate(zip(base_epochs, adjusted_epochs)):
        if base != adj:
            ax.annotate('', xy=(i + width/2, adj), xytext=(i - width/2, base),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
            ax.text(i, max(base, adj) + 0.5, f'+{int(adj-base)}', 
                   ha='center', fontsize=11, fontweight='bold', color='red')
    
    ax.set_xlabel('Drone', fontsize=13, fontweight='bold')
    ax.set_ylabel('Epochs per Round', fontsize=13, fontweight='bold')
    ax.set_title('Priority-Based Epoch Allocation\nHIGH priority drones train 2x longer', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(drones)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 18])
    
    plt.tight_layout()
    plt.savefig('priority_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Grafik kaydedildi: priority_impact. png")
    plt.show()

def print_network_summary():
    """
    Network challenge özeti
    """
    print("\n" + "="*70)
    print(" "*15 + "NETWORK-AWARE FEDERATED LEARNING - SUMMARY")
    print("="*70)
    
    print("\n🚁 DRONE SETUP:")
    print("   Drone 1 (Sehir):    5% packet loss,  LOW priority")
    print("   Drone 2 (Sanayi):  15% packet loss, MEDIUM priority")
    print("   Drone 3 (Orman):   40% packet loss,  HIGH priority ⚠️")
    print("   Drone 4 (Daglik):  35% packet loss, HIGH priority ⚠️")
    print("   Drone 5 (Karma):    8% packet loss, LOW priority")
    
    print("\n🌐 NETWORK CHALLENGES:")
    print("   ✓ Packet loss simulation (5%-40%)")
    print("   ✓ Latency injection (0.1s-3. 0s)")
    print("   ✓ Random disconnections (1%-15%)")
    print("   ✓ Retry mechanisms (3 attempts)")
    
    print("\n🎯 PRIORITY SYSTEM:")
    print("   HIGH:    2. 0x epochs (14 instead of 7)")
    print("   MEDIUM: 1.5x epochs (10 instead of 7)")
    print("   LOW:    1.0x epochs (7 baseline)")
    
    print("\n📊 RESULTS:")
    print("   Round 1: 93.75% (Drone 3 & 4 issues)")
    print("   Round 2: 96.88% (All successful)")
    print("   Round 3: 91.67% (Drone 5 packet loss)")
    print("   Round 4: 96.88% (Drone 3 packet loss)")
    print("   Round 5: 98.33% (Drone 4 disconnected)")
    print("   Round 6: 98.33% (ALL DRONES 100%!) 🔥")
    
    print("\n✅ KEY ACHIEVEMENTS:")
    print("   ✓ 98.33% final accuracy despite network issues")
    print("   ✓ Handled 5 packet loss events gracefully")
    print("   ✓ Handled 2 disconnection events")
    print("   ✓ Priority system ensured critical drones trained longer")
    print("   ✓ Robust federated learning in adverse conditions")
    
    print("\n💡 REAL-WORLD APPLICABILITY:")
    print("   • Emergency response drones in remote areas")
    print("   • Disaster zones with damaged infrastructure")
    print("   • Military operations with jamming threats")
    print("   • Rural/mountainous deployments")
    
    print("\n" + "="*70)
    print(" "*18 + "🎉 NETWORK-RESILIENT FL SUCCESS!")
    print("="*70 + "\n")

if __name__ == "__main__":
    print("📊 Gorsellestirmeler olusturuluyor.. .\n")
    
    
    # Grafikler
    plot_federated_results()
    plot_drone_comparison()
    
    # Özet rapor
    
    print("📁 Olusturulan dosyalar:")
    print("   • federated_results.png")
    print("   • drone_comparison.png")

    plot_network_challenges()
    plot_priority_impact()
    print_network_summary()
    print_summary()

    print("\n📁 Olusturulan dosyalar:")
    print("   • network_challenges.  png")
    print("   • priority_impact. png")