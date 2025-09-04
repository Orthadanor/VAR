"""
MRI Volume Dataset utilities for loading 3D volume data (.npy files with shape [H, W, num_slices])
Following the same structure as data_mri.py
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, transforms
from pathlib import Path
from typing import Tuple


def normalize_01_into_pm1(x):
    """Convert [0,1] range to [-1,1] range"""
    return x.add(x).add_(-1)


class MRIDatasetVolume(Dataset):
    """Dataset for loading 3D MRI volume data following data_mri.py structure"""
    
    def __init__(self, root_dir, transform=None, num_slices=10):
        """
        Args:
            root_dir: Root directory containing train/val subdirectories
            transform: Optional transforms to apply
            num_slices: Number of slices in each volume
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_slices = num_slices
        self.files = []
        self.class_to_idx = {'mid_brain': 0}  # Single class
        self.classes = ['mid_brain']
        
        # Collect all .npy files from mid_brain subdirectory
        class_dir = osp.join(root_dir, 'mid_brain')
        if osp.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    self.files.append(osp.join(class_dir, fname))
        
        if len(self.files) == 0:
            raise ValueError(f"No .npy files found in {class_dir}")
        
        print(f"Found {len(self.files)} volume files in {class_dir}")
        
        # Verify volume shapes
        self._verify_volumes()
    
    def _verify_volumes(self):
        """Verify that all volumes have the expected shape"""
        print("Verifying volume shapes...")
        for i, file_path in enumerate(self.files[:5]):  # Check first 5 files
            vol_data = np.load(file_path)
            expected_shape = (128, 128, self.num_slices)  # Assuming 128x128 resolution
            if vol_data.shape != expected_shape:
                print(f"Warning: Volume {osp.basename(file_path)} has shape {vol_data.shape}, expected {expected_shape}")
        
        # Check a few more files for slice count
        for file_path in self.files[5:10]:
            vol_data = np.load(file_path)
            if vol_data.shape[-1] != self.num_slices:
                print(f"Warning: Volume {osp.basename(file_path)} has {vol_data.shape[-1]} slices, expected {self.num_slices}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load volume data
        vol_data = np.load(self.files[idx]).astype(np.float32)
        
        # Ensure volume has correct shape [H, W, num_slices]
        if vol_data.shape[-1] != self.num_slices:
            raise ValueError(f"Volume {osp.basename(self.files[idx])} has {vol_data.shape[-1]} slices, expected {self.num_slices}")
        
        # Convert to torch tensor and normalize to [0, 1] if needed
        vol_tensor = torch.from_numpy(vol_data)
        
        # Normalize to [0, 1] if not already
        if vol_tensor.max() > 1.0:
            print(f"Volume {osp.basename(self.files[idx])} has max value {vol_tensor.max()}, normalizing to [0, 1]")
            vol_tensor = vol_tensor / 255.0
        
        # Transpose to [num_slices, H, W] for processing (equivalent to ToTensor for volumes)
        vol_tensor = vol_tensor.permute(2, 0, 1)  # [num_slices, H, W]
        
        # Apply transforms (including normalize_01_into_pm1)
        if self.transform:
            vol_tensor = self.transform(vol_tensor)
        
        # Return volume and class label (for compatibility)
        return vol_tensor, 0  # [num_slices, H, W], class_label


def build_mri_dataset_volume(data_path: str, final_reso: int, hflip=False, num_slices=10, mid_reso=1.125):
    """
    Build train and validation datasets for MRI volume data
    Following the same structure as build_mri_dataset_grayscale
    """
    # Build augmentations for volume data
    # Note: No resizing/cropping needed as data should already be at correct resolution
    train_aug, val_aug = [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
        # transforms.RandomCrop((final_reso, final_reso)),  # Commented out - no cropping
        # ToTensor equivalent: already done in __getitem__ with permute
        normalize_01_into_pm1,  # Convert [0,1] to [-1,1]
    ], [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
        # transforms.CenterCrop((final_reso, final_reso)),  # Commented out - no cropping
        # ToTensor equivalent: already done in __getitem__ with permute
        normalize_01_into_pm1,  # Convert [0,1] to [-1,1]
    ]
    
    # Add horizontal flip only if hflip=True
    if hflip: 
        train_aug.insert(0, transforms.RandomHorizontalFlip())
    
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # Build datasets
    train_set = MRIDatasetVolume(
        root_dir=osp.join(data_path, 'train'), 
        transform=train_aug,
        num_slices=num_slices
    )
    val_set = MRIDatasetVolume(
        root_dir=osp.join(data_path, 'val'), 
        transform=val_aug,
        num_slices=num_slices
    )
    num_classes = 1  # Single class for unconditional generation
    
    # Check data dimensions
    if len(train_set) > 0:
        sample_vol, _ = train_set[0]
        print(f'[MRI Volume Dataset] Sample volume shape: {sample_vol.shape}')
        print(f'[MRI Volume Dataset] Expected shape: [{num_slices}, {final_reso}, {final_reso}]')
        if sample_vol.shape[0] != num_slices or sample_vol.shape[1] != final_reso or sample_vol.shape[2] != final_reso:
            print(f'⚠️  WARNING: Sample volume size {sample_vol.shape} does not match expected [{num_slices}, {final_reso}, {final_reso}]')
            print(f'   Make sure your preprocessed data is already at the correct resolution!')
    
    print(f'[MRI Volume Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print(f'[Classes] {train_set.classes}')
    print_aug(train_aug, '[train] (NO resizing/cropping)')
    print_aug(val_aug, '[val] (NO resizing/cropping)')
    
    return num_classes, train_set, val_set


def print_aug(transform, label):
    """Print augmentation transforms"""
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


def test_volume_dataset():
    """Test function for volume dataset"""
    print("Testing MRI Volume Dataset...")
    
    # Test with a sample directory (you'll need to adjust this path)
    test_data_path = "/home/yuchenliu/Dataset/IXI/train_val_test_split_multislice"
    
    try:
        # Test with train/val split
        num_classes, train_dataset, val_dataset = build_mri_dataset_volume(
            data_path=test_data_path,
            final_reso=128,
            hflip=False,  # Disable horizontal flip
            num_slices=10
        )
        
        print(f"\nTrain/Val split:")
        print(f"  Train: {len(train_dataset)} volumes")
        print(f"  Val: {len(val_dataset)} volumes")
        
        # Test loading a sample
        if len(train_dataset) > 0:
            vol_tensor, label = train_dataset[0]
            print(f"Sample volume shape: {vol_tensor.shape}")
            print(f"Sample volume range: [{vol_tensor.min():.3f}, {vol_tensor.max():.3f}]")
            print(f"Sample label: {label}")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        for batch_idx, (batch_x, batch_labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}: shape {batch_x.shape}, labels {batch_labels}")
            break
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the test data path exists and contains train/mid_brain/ and val/mid_brain/ subdirectories with .npy volume files")


if __name__ == "__main__":
    test_volume_dataset()
