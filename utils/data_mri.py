import os
import os.path as osp
import numpy as np
import PIL.Image as PImage
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)


class MRIDatasetGrayscale(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.class_to_idx = {'mid_brain': 0}  # Single class
        self.classes = ['mid_brain']
        
        # Collect all .npy files
        class_dir = osp.join(root_dir, 'mid_brain')
        if osp.exists(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith('.npy'):
                    self.files.append(osp.join(class_dir, fname))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load .npy file
        data = np.load(self.files[idx]).astype(np.float32)
        
        # Keep as single channel (no duplication!)
        if len(data.shape) == 2:
            # Convert to PIL Image as grayscale
            data_uint8 = (data * 255).astype(np.uint8)
            img = PImage.fromarray(data_uint8, mode='L')  # 'L' mode for grayscale
        
        if self.transform:
            img = self.transform(img)
            
        return img, 0  # Always return class 0 (single class)


def build_mri_dataset_grayscale(data_path: str, final_reso: int, hflip=False, mid_reso=1.125):
    # Build augmentations for grayscale
    # mid_reso = round(mid_reso * final_reso)  # Not needed if no resizing
    train_aug, val_aug = [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
        # transforms.RandomCrop((final_reso, final_reso)),  # Commented out - no cropping
        transforms.ToTensor(),  # This will create 1-channel tensor for grayscale
        normalize_01_into_pm1,
    ], [
        # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
        # transforms.CenterCrop((final_reso, final_reso)),  # Commented out - no cropping
        transforms.ToTensor(),  # This will create 1-channel tensor for grayscale
        normalize_01_into_pm1,
    ]
    if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
    train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
    # Build datasets
    train_set = MRIDatasetGrayscale(root_dir=osp.join(data_path, 'train'), transform=train_aug)
    val_set = MRIDatasetGrayscale(root_dir=osp.join(data_path, 'val'), transform=val_aug)
    num_classes = 1  # Single class for unconditional generation
    
    # Check data dimensions (without augmentation, should match your preprocessed data)
    if len(train_set) > 0:
        sample_img, _ = train_set[0]
        print(f'[MRI Dataset] Sample image shape: {sample_img.shape}')
        print(f'[MRI Dataset] Expected final_reso: {final_reso}x{final_reso}')
        if sample_img.shape[-1] != final_reso or sample_img.shape[-2] != final_reso:
            print(f'⚠️  WARNING: Sample image size {sample_img.shape[-2:]} does not match expected final_reso {final_reso}')
            print(f'   Make sure your preprocessed data is already at the correct resolution!')
    
    print(f'[MRI Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
    print(f'[Classes] {train_set.classes}')
    print_aug(train_aug, '[train] (NO resizing/cropping)')
    print_aug(val_aug, '[val] (NO resizing/cropping)')
    
    return num_classes, train_set, val_set


# class MRIDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.files = []
#         self.class_to_idx = {'mid_brain': 0}  # Single class
#         self.classes = ['mid_brain']
        
#         # Collect all .npy files
#         class_dir = osp.join(root_dir, 'mid_brain')
#         if osp.exists(class_dir):
#             for fname in os.listdir(class_dir):
#                 if fname.endswith('.npy'):
#                     self.files.append(osp.join(class_dir, fname))
    
#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         # Load .npy file
#         data = np.load(self.files[idx]).astype(np.float32)
        
#         # Convert to 3-channel (RGB) by replicating grayscale
#         if len(data.shape) == 2:
#             data = np.stack([data, data, data], axis=-1)
        
#         # Convert to PIL Image for transforms
#         data_uint8 = (data * 255).astype(np.uint8)
#         img = PImage.fromarray(data_uint8)
        
#         if self.transform:
#             img = self.transform(img)
            
#         return img, 0  # Always return class 0 (single class)


# def build_mri_dataset(data_path: str, final_reso: int, hflip=False, mid_reso=1.125):
#     # Build augmentations (same as original)
#     # mid_reso = round(mid_reso * final_reso)  # Not needed if no resizing
#     train_aug, val_aug = [
#         # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
#         # transforms.RandomCrop((final_reso, final_reso)),  # Commented out - no cropping
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ], [
#         # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),  # Commented out - no resizing
#         # transforms.CenterCrop((final_reso, final_reso)),  # Commented out - no cropping
#         transforms.ToTensor(), normalize_01_into_pm1,
#     ]
#     if hflip: train_aug.insert(0, transforms.RandomHorizontalFlip())
#     train_aug, val_aug = transforms.Compose(train_aug), transforms.Compose(val_aug)
    
#     # Build datasets
#     train_set = MRIDataset(root=osp.join(data_path, 'train'), transform=train_aug)
#     val_set = MRIDataset(root=osp.join(data_path, 'val'), transform=val_aug)
#     num_classes = 1  # Single class for unconditional generation
    
#     print(f'[MRI Dataset] {len(train_set)=}, {len(val_set)=}, {num_classes=}')
#     print(f'[Classes] {train_set.classes}')
    
#     print_aug(train_aug, '[train]')
#     print_aug(val_aug, '[val]')
    
#     return num_classes, train_set, val_set

def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')