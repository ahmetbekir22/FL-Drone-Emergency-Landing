#  Federated Drone Emergency Landing


```markdown

Federated learning system where 3 drones learn to detect safe landing zones from 3D point clouds without sharing data.

##  What It Does

- 3 drones in different environments (urban, forest, mixed)
- Each trains locally on 3D point cloud data
- Share only model weights, not raw data
- Achieve **97.22% accuracy** through collaboration

**Key Result:** Global federated model outperforms individual drones by learning from all environments. 

##  Performance

```
Global Model:   90.0% → 97.22% (+7.22%)
Loss:         0.269 → 0.060 (-77.7%)

Drone 1 (Urban):  88.33% → 93.33%
Drone 2 (Forest): 83.33% → 96.67%
Drone 3 (Mixed):  96.67% → 90.00%
```

**Federated advantage:** Global model (97.22%) is more robust than any single drone. 

##  Quick Start

### 1. Install
```bash
pip install torch flwr matplotlib numpy trimesh requests tqdm
```

### 2. Download Dataset
```bash
python download_modelnet.py    # Downloads ModelNet10 (~500MB)
python prepare_dataset.py       # Converts to point clouds
```

### 3. Run Federated Learning

**Terminal 1:**
```bash
python server.py
```

**Terminal 2, 3, 4:**
```bash
python client.py 1
python client.py 2
python client.py 3
```

### 4. Visualize
```bash
python visualize.py
```

##  Structure

```
├── download_modelnet.py    # Dataset downloader
├── prepare_dataset.py      # Point cloud generator
├── model.py                # PointNet (801K params)
├── dataset.py              # Data loader
├── train.py                # Training functions
├── client.py               # Drone client
├── server.py               # FL server
└── visualize.py            # Plot results
```

##  Tech Stack

- **Model:** PointNet (1024 points → safe/unsafe classification)
- **Dataset:** ModelNet10 (900 samples, 3D objects)
- **Framework:** Flower 1.25.0 + PyTorch
- **Strategy:** FedAvg, 5 rounds, 5 local epochs

##  Dataset

ModelNet10 objects mapped to landing zones:
- **Safe:** table, bed, desk, bathtub (flat surfaces)
- **Unsafe:** chair, sofa, plant, monitor (obstacles)

Each drone gets 300 samples with different safe/unsafe ratios.

##  Future Work

- Increase to 5+ drones
- Real LiDAR data integration
- Deploy on actual drone hardware
