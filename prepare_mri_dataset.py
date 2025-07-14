import os
import shutil
from pathlib import Path

def prepare_unconditional_dataset():
    source_dir = Path("/home/yuchenliu/Dataset/IXI/t1_np_masked_128")
    target_dir = Path("/home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional")
    
    # Create single class directories
    (target_dir / "train" / "mid_brain").mkdir(parents=True, exist_ok=True)
    (target_dir / "val" / "mid_brain").mkdir(parents=True, exist_ok=True)
    
    # Move all training data to single class
    train_files = list((source_dir / "train").glob("*.npy"))
    for file in train_files:
        shutil.copy(file, target_dir / "train" / "mid_brain" / file.name)
    
    # Move all validation data to single class  
    val_files = list((source_dir / "val").glob("*.npy"))
    for file in val_files:
        shutil.copy(file, target_dir / "val" / "mid_brain" / file.name)
    
    print(f"Created unconditional dataset with {len(train_files)} train and {len(val_files)} val samples")
    
    # Print Number of files in each directory
    print(f"Train files: {len(list((target_dir / 'train' / 'mid_brain').glob('*.npy')))}")
    print(f"Val files: {len(list((target_dir / 'val' / 'mid_brain').glob('*.npy')))}")

if __name__ == "__main__":
    prepare_unconditional_dataset()