# Federated Drone Emergency Landing

```markdown

Network-resilient federated learning system where 5 drones learn to detect safe emergency landing zones from 3D point clouds despite adverse network conditions — without sharing raw data.

## What It Does

- **5 drones** in different environments with varying network conditions
- Each trains locally on **3D point cloud** data (LiDAR-style)
- Share only **model weights**, not raw sensor data
- **Priority-based** federated learning (critical drones train longer)
- **Network-aware**:  Handles packet loss, latency, disconnections
- Achieves **98.33% accuracy** despite 40% packet loss

**Key Innovation:** Robust federated learning in real-world adverse network conditions with priority-based resource allocation.

## Performance

### Global Model:
```
Round 1: 93.75%
Round 2: 96.88%
Round 3: 91.67% (packet loss event)
Round 4: 96.88%
Round 5: 98.33%
Round 6: 98.33% (ALL DRONES 100%)
```

### Network Challenges Handled:
- 5 packet loss events (up to 40%)
- 2 connection drops
- Variable latency (0.1s - 3.0s)
- Priority-based recovery

### Drone Profiles:  

| Drone | Environment | Packet Loss | Priority | Final Accuracy |
|-------|-------------|-------------|----------|----------------|
| 1     | Urban       | 5%          | LOW      | 100%           |
| 2     | Industrial  | 15%         | MEDIUM   | 92.5%          |
| 3     | Forest      | 40%         | HIGH     | 100%           |
| 4     | Mountain    | 35%         | HIGH     | 100%           |
| 5     | Mixed       | 8%          | LOW      | 100%           |

**Result:** Critical drones (forest/mountain) achieved 100% despite worst network conditions through priority training. 

## Quick Start

### 1. Install Dependencies

```bash
pip install torch flwr matplotlib numpy trimesh requests tqdm
```

### 2. Download & Prepare Dataset

```bash
python download_modelnet.py    # Downloads ModelNet10 (~500MB)
python prepare_dataset. py      # Creates 5 drone datasets (1000 samples total)
```

### 3. Run Federated Learning

**Terminal 1 - Server:**

```bash
python server.py
```

**Terminals 2-6 - Drone Clients:**

```bash
python client.py 1  # Urban (good network)
python client.py 2  # Industrial (medium network)
python client.py 3  # Forest (CRITICAL - bad network)
python client.py 4  # Mountain (CRITICAL - bad network)
python client.py 5  # Mixed (good network)
```

Server waits for all 5 drones, then runs 6 federated rounds with network simulation.

### 4. Visualize Results

```bash
python visualize_network.py  # Network-aware FL results
```

## Project Structure

```
federated-drone-landing/
├── data/
│   ├── ModelNet10/          # Downloaded 3D meshes
│   ├── drone1/              # 200 point clouds (80% safe)
│   ├── drone2/              # 200 samples (60% safe)
│   ├── drone3/              # 200 samples (20% safe - challenging)
│   ├── drone4/              # 200 samples (30% safe)
│   └── drone5/              # 200 samples (50% safe)
├── download_modelnet.py     # Dataset downloader
├── prepare_dataset.py       # Point cloud generator (5 drones)
├── model.py                 # PointNet architecture (801K params)
├── dataset.py               # PyTorch DataLoader
├── train.py                 # Training/evaluation functions
├── client.py                # Flower client with network simulation
├── server.py                # Priority-aware FL server
├── visualize_network.py     # Network challenge visualization
└── README.md
```

## Architecture

### Model: PointNet Classifier

- **Input:** `[Batch, 1024, 3]` point cloud (X, Y, Z coordinates)
- **Backbone:** Shared MLPs (3→64→128→1024) + Max Pooling
- **Classifier:** FC layers (1024→512→256→2)
- **Parameters:** 801,282
- **Task:** Binary classification (safe/unsafe landing zone)

### Federated Learning

- **Framework:** Flower 1.25.0
- **Strategy:** Priority-aware FedAvg
- **Rounds:** 6
- **Base epochs:** 7 per round
- **Priority multipliers:** HIGH=2.0x, MEDIUM=1.5x, LOW=1.0x
- **Min clients per round:** 3/5 (tolerates failures)

## Network Simulation

### Per-Drone Conditions:

```
Drone 1 (Urban):      5% packet loss,  0.1-0.5s latency,  1% disconnect
Drone 2 (Industrial): 15% packet loss, 0.3-1.0s latency,  5% disconnect
Drone 3 (Forest):     40% packet loss, 1.0-3.0s latency, 15% disconnect
Drone 4 (Mountain):   35% packet loss, 0.8-2.5s latency, 12% disconnect
Drone 5 (Mixed):      8% packet loss,  0.2-0.7s latency,  2% disconnect
```

### Network Features:

- Packet loss simulation with retry mechanism (3 attempts)
- Latency injection (sleep-based delays)
- Random disconnections (skip round, rejoin next)
- Priority-based recovery (critical drones get more training)
- Graceful degradation (server aggregates available clients)

## Dataset

**ModelNet10** - 10 categories of 3D CAD models converted to point clouds

### Landing Zone Mapping:

- **Safe (4 categories):** bathtub, bed, desk, table → Flat surfaces for landing
- **Unsafe (6 categories):** chair, dresser, monitor, nightstand, sofa, toilet → Obstacles

### Point Cloud Processing: 

- Each mesh sampled to **1024 points**
- Normalized to **[-1, 1]** range
- Augmented with random rotations (training only)

### Distribution:

```
Drone 1 (Urban):      160 safe, 40 unsafe  (80% safe - parks, open areas)
Drone 2 (Industrial): 120 safe, 80 unsafe  (60% safe - flat roofs, warehouses)
Drone 3 (Forest):     40 safe, 160 unsafe  (20% safe - trees, obstacles)
Drone 4 (Mountain):   60 safe, 140 unsafe  (30% safe - rocks, uneven terrain)
Drone 5 (Mixed):      100 safe, 100 unsafe (50% safe - test distribution)
```

## Results

### Key Metrics:

- **Final Global Accuracy:** 98.33%
- **Loss Reduction:** 0.238 → 0.060 (75% reduction)
- **Network Resilience:** 5 packet loss events handled
- **Priority Impact:** HIGH priority drones trained 2x longer (14 epochs vs 7)

### Observations:

1. **Round 2:** All drones successful → peak accuracy 96.88%
2. **Round 3:** Drone 5 packet loss → accuracy dip to 91.67%
3. **Round 6:** ALL drones reached 100% local accuracy
4. **Drone 3 (Forest):** Worst network (40% loss) but 100% final accuracy through priority training
-

