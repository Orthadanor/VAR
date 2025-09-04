import os
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append("/home/yuchenliu/VAR")
from utils.data_mri import braTS2d

def test_braTS_preprocessing(data_path, crop_shape=(128, 128), train_split=0.95, batch_size=16):
    # Preprocess the dataset
    train_set, val_set = braTS2d(data_path, crop_shape, train_split)

    # Print dataset statistics
    print(f"[BraTS Preprocessing] Total training slices: {len(train_set)}")
    print(f"[BraTS Preprocessing] Total validation slices: {len(val_set)}")

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Save a few random slices from the training set for inspection
    save_dir = Path("/home/yuchenliu/VAR/generated_samples")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (images, _) in enumerate(train_loader):
        if i >= 5:  # Save only the first few batches
            break
        for j, img in enumerate(images):
            img = (img.numpy().squeeze() + 1) * 127.5  # Map back to [0, 255]
            img = Image.fromarray(img.astype('uint8'))
            img.save(save_dir / f"train_sample_{i * batch_size + j}.png")

    print(f"[BraTS Preprocessing] Saved a few training samples to {save_dir}")

if __name__ == "__main__":
    data_path = "/home/yuchenliu/Dataset/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
    test_braTS_preprocessing(data_path)
