# download_modelnet.py
import os
import urllib.request
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None: 
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

def download_modelnet10():
    """Download and extract ModelNet10 dataset"""
    
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = "ModelNet10.zip"
    extract_path = "data"
    
    if os.path.exists(os.path.join(extract_path, "ModelNet10")):
        print(" ModelNet10 zaten indirilmiş!")
        return
    
    print(" ModelNet10 indiriliyor...  (~500MB, biraz zaman alabilir)")
    download_file(url, zip_path)
    
    print("\n Dosya açılıyor...")
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref. extractall(extract_path)
    
    print(" Zip dosyası temizleniyor...")
    os.remove(zip_path)
    
    print("\n ModelNet10 başarıyla indirildi!")
    print(f" Konum: {os.path.join(extract_path, 'ModelNet10')}")
    
    # Kategorileri listele
    modelnet_path = os. path.join(extract_path, "ModelNet10")
    categories = sorted([d for d in os.listdir(modelnet_path) if os.path.isdir(os.path.join(modelnet_path, d))])
    
    print(f"\n {len(categories)} kategori bulundu:")
    for cat in categories:
        train_files = len(os.listdir(os.path.join(modelnet_path, cat, "train")))
        test_files = len(os.listdir(os.path.join(modelnet_path, cat, "test")))
        print(f"   - {cat}: {train_files} train, {test_files} test")

if __name__ == "__main__":
    download_modelnet10()