# prepare_dataset.py
import os
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm

# Kategori mapping:  Güvenli vs Tehlikeli
SAFE_CATEGORIES = ['bathtub', 'bed', 'desk', 'table']
UNSAFE_CATEGORIES = ['chair', 'dresser', 'monitor', 'night_stand', 'sofa', 'toilet']

def sample_point_cloud(mesh_path, num_points=1024):
    """
    Mesh dosyasından point cloud örnekle
    """
    try:
        mesh = trimesh. load(mesh_path)
        
        # Eğer birden fazla mesh varsa birleştir
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # Point cloud örnekle
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        # Normalize et ([-1, 1] aralığına)
        points = points - points.mean(axis=0)
        points = points / np.abs(points).max()
        
        return points. astype(np.float32)
    
    except Exception as e: 
        print(f" Hata ({mesh_path}): {e}")
        return None

def prepare_drone_dataset(drone_id, safe_ratio=0.5, samples_per_drone=200):
    """
    Her drone için özelleştirilmiş dataset hazırla
    
    Drone 1: Şehir merkezi (80% güvenli - parklar, açık alanlar)
    Drone 2: Sanayi bölgesi (60% güvenli - düz çatılar)
    Drone 3: Orman (20% güvenli - ağaçlar, engeller)
    Drone 4: Dağlık alan (30% güvenli - kayalar, engebeli)
    Drone 5: Karma/Test (50% güvenli - her ortamdan)
    """
    
    output_dir = f"data/drone{drone_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    modelnet_path = "data/ModelNet10"
    
    # Drone'a özel dağılım
    if drone_id == 1:
        # Şehir merkezi:  %80 güvenli
        num_safe = int(samples_per_drone * 0.8)
        num_unsafe = samples_per_drone - num_safe
        env_name = "Sehir Merkezi - Acik Alanlar"
    elif drone_id == 2:
        # Sanayi:  %60 güvenli
        num_safe = int(samples_per_drone * 0.6)
        num_unsafe = samples_per_drone - num_safe
        env_name = "Sanayi Bolgesi - Duz Yuzeyler"
    elif drone_id == 3:
        # Orman: %20 güvenli
        num_safe = int(samples_per_drone * 0.2)
        num_unsafe = samples_per_drone - num_safe
        env_name = "Orman - Cok Engel"
    elif drone_id == 4:
        # Dağlık:  %30 güvenli
        num_safe = int(samples_per_drone * 0.3)
        num_unsafe = samples_per_drone - num_safe
        env_name = "Daglik Alan - Engebeli"
    else:
        # Karma:  %50-%50
        num_safe = samples_per_drone // 2
        num_unsafe = samples_per_drone - num_safe
        env_name = "Karma - Dengeli"
    
    print(f"\n Drone {drone_id} ({env_name}):")
    print(f"   Güvenli: {num_safe} örnek")
    print(f"   Tehlikeli: {num_unsafe} örnek")
    
    # Güvenli alanlar
    safe_count = 0
    for category in SAFE_CATEGORIES: 
        train_dir = os.path.join(modelnet_path, category, "train")
        off_files = list(Path(train_dir).glob("*.off"))
        
        samples_from_cat = min(len(off_files), num_safe // len(SAFE_CATEGORIES) + 10)
        selected_files = np.random.choice(off_files, samples_from_cat, replace=False)
        
        for off_file in tqdm(selected_files, desc=f"  {category} (safe)", leave=False):
            if safe_count >= num_safe: 
                break
            
            points = sample_point_cloud(str(off_file))
            if points is not None:
                np.save(f"{output_dir}/safe_{safe_count}.npy", points)
                safe_count += 1
        
        if safe_count >= num_safe: 
            break
    
    # Tehlikeli alanlar
    unsafe_count = 0
    for category in UNSAFE_CATEGORIES:
        train_dir = os.path.join(modelnet_path, category, "train")
        off_files = list(Path(train_dir).glob("*.off"))
        
        samples_from_cat = min(len(off_files), num_unsafe // len(UNSAFE_CATEGORIES) + 10)
        selected_files = np.random.choice(off_files, samples_from_cat, replace=False)
        
        for off_file in tqdm(selected_files, desc=f"  {category} (unsafe)", leave=False):
            if unsafe_count >= num_unsafe:
                break
            
            points = sample_point_cloud(str(off_file))
            if points is not None:
                np.save(f"{output_dir}/unsafe_{unsafe_count}. npy", points)
                unsafe_count += 1
        
        if unsafe_count >= num_unsafe:
            break
    
    print(f" Drone {drone_id}:  {safe_count} güvenli + {unsafe_count} tehlikeli = {safe_count + unsafe_count} toplam")

def main():
    print(" ModelNet10'dan 5 Drone Dataset'i Hazırlanıyor...")
    print("="*60)
    
    # 5 drone için veri hazırla
    for drone_id in [1, 2, 3, 4, 5]:
        prepare_drone_dataset(drone_id, samples_per_drone=200)
    
    print("\n" + "="*60)
    print(" Tüm drone dataset'leri hazır!")
    print("\n Dizin yapısı:")
    print("data/")
    print("  ├── drone1/ (Şehir - 80% güvenli)")
    print("  ├── drone2/ (Sanayi - 60% güvenli)")
    print("  ├── drone3/ (Orman - 20% güvenli)")
    print("  ├── drone4/ (Dağlık - 30% güvenli)")
    print("  └── drone5/ (Karma - 50% güvenli)")
    
    # Özet istatistikler
    print("\n Dataset Özeti:")
    for drone_id in [1, 2, 3, 4, 5]:
        drone_dir = f"data/drone{drone_id}"
        if os.path.exists(drone_dir):
            files = os.listdir(drone_dir)
            safe = len([f for f in files if f.startswith('safe')])
            unsafe = len([f for f in files if f.startswith('unsafe')])
            print(f"  Drone {drone_id}: {safe} güvenli, {unsafe} tehlikeli")

if __name__ == "__main__":
    main()