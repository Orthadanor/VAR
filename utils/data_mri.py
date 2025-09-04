import os
import os.path as osp
import numpy as np
import PIL.Image as PImage
from torch.utils.data import Dataset, random_split
from torchvision.transforms import InterpolationMode, transforms
from vqvae.braTS_aug_utils import Dataset_Training


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


def braTS2d(data_path, crop_shape=(128, 128), train_split=0.95):
    return Dataset_Training.braTS2d(data_path, crop_shape, train_split)