import os
import random
import numpy as np
from PIL import Image
from utils.data_mri import build_braTS2d

def test_braTS2d(data_path, output_dir):
    _, train_set, _ = build_braTS2d(data_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Randomly sample slices from train_set
    sampled_indices = random.sample(range(len(train_set)), 50)
    for i, idx in enumerate(sampled_indices):
        img, _ = train_set[idx]
        img = (img.add_(1).div_(2).numpy() * 255).astype(np.uint8)  # Convert to uint8
        img = Image.fromarray(img[0], mode='L')  # Convert to grayscale PIL Image
        img.save(os.path.join(output_dir, f'sample_{i}.png'))

    print(f"[Test] Saved {len(sampled_indices)} random slices to {output_dir}")

if __name__ == "__main__":
    data_path = "../Dataset/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
    output_dir = "../VAR/generated_samples"
    test_braTS2d(data_path, output_dir)
