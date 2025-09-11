import os
import os.path as osp
from pathlib import Path
import numpy as np
import PIL.Image as PImage
from torch.utils.data import Dataset, random_split
from torchvision.transforms import InterpolationMode, transforms
from braTS_aug_utils import *


def normalize_01_into_pm1(x):
    return x.add(x).add_(-1)


class IXI2DDataset(Dataset):
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

def build_ixi_2d(data_path: str, final_reso: int, hflip=False, mid_reso=1.125):
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
    train_set = IXI2DDataset(root_dir=osp.join(data_path, 'train'), transform=train_aug)
    val_set = IXI2DDataset(root_dir=osp.join(data_path, 'val'), transform=val_aug)
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

def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')

# def braTS2d(data_path, crop_shape=(128, 128), train_split=0.95):
#     return Dataset_Training.braTS2d(data_path, crop_shape, train_split)

class BraTS2DDataset(Dataset):
    def __init__(self, root_dir, crop_shape=(128, 128), healthy_mask_crop=(128, 128, 96), transform=None):
        self.root_dir = root_dir or "../../Dataset/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
        self.output_dir = Path("../../Dataset/BraTS2023-2D-Slices")
        self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        self.list_paths_t1n = list(Path(root_dir).rglob("**/BraTS-GLI-*-*-t1n.nii.gz"))
        self.list_paths_mask_healthy = list(Path(root_dir).rglob("**/BraTS-GLI-*-*-mask-healthy.nii.gz"))

        self.crop_shape = crop_shape
        self.healthy_mask_crop = healthy_mask_crop
        self.transform = transform
        self.slices = []

        if self.output_dir.exists() and any(self.output_dir.glob("*.npy")):
            print("[BraTS2D] Preprocessed slices found. Loading from disk...")
            self.slices = list(self.output_dir.glob("*.npy"))
            print(f"[BraTS2D] Loaded {len(self.slices)} slices from disk.")
        else:
            print("[BraTS2D] No preprocessed slices found. Preprocessing dataset...")
            self.preprocess()

    def preprocess(self):
        
        print(f"Processing {len(self.list_paths_t1n)} t1n volumes and {len(self.list_paths_mask_healthy)} corresponding healthy masks")

        for idx in range(len(self.list_paths_t1n)):
            t1n_path, healthy_mask_path = self.list_paths_t1n[idx], self.list_paths_mask_healthy[idx]
            t1n, healthy_mask = nib.load(t1n_path).get_fdata(), nib.load(healthy_mask_path).get_fdata()

            referenceShape = (240, 240, 155)
            if t1n.shape != referenceShape or healthy_mask.shape != referenceShape:
                raise UserWarning(f"Invalid shape: {t1n.shape}, {healthy_mask.shape}")

            # Normalize the image to [0,1]
            t1n[t1n < 0] = 0  # Values below 0 are considered to be noise.
            # Note that only 4 samples fulfill min(t1)!=0 : GLI-01332-000, GLI-00048-001, GLI-00446-000 and BraTS2023_01655
            t1n_max_v = np.max(t1n)
            t1n /= t1n_max_v

            ############################ Take the middle section of original t1n brains #####################################
            # Process t1n images
            # Crop to whole brain (removes everything empty space)
            max_bbox_raw = compute_bbox(t1n)
            t1n_crop = t1n[max_bbox_raw]
            
            # Inspect the shape after cropping
            print("Inspect shape after cropping:", t1n_crop.shape)

            # Extract 2D slices from the middle (128, 128) section
            height, width, depth = t1n_crop.shape
            center_h, center_w = height // 2, width // 2
            half_crop_h, half_crop_w = self.crop_shape[0] // 2, self.crop_shape[1] // 2

            for z in range(depth):
                t1n_slice = t1n_crop[
                    center_h - half_crop_h : center_h + half_crop_h,
                    center_w - half_crop_w : center_w + half_crop_w,
                    z,
                ]
                if not np.any(t1n_slice > 0):
                    print("Skipping completely dark slice")
                else:
                    slice_path = self.output_dir / f"slice_{len(self.slices):05d}.npy"
                    np.save(slice_path, t1n_slice)
                    self.slices.append(slice_path)
                    

            ############################ Take the section around healthy brain masks #####################################                
            # Data augmentation: crop to region around healthy masks
            if self.healthy_mask_crop is not None:
                shape = healthy_mask.shape[-3:]
                min_bbox_mask = compute_bbox(healthy_mask)
                max_bbox_mask = []
                for i, s in enumerate(min_bbox_mask):
                    s: slice
                    d = self.healthy_mask_crop[i] - (s.stop - s.start)
                    s_n = slice(s.start - d // 2, s.stop + ceil(d / 2))
                    if s_n.start < 0:
                        s_n = slice(0, self.healthy_mask_crop[i])
                    if s_n.stop > shape[i]:
                        s_n = slice(shape[i] - self.healthy_mask_crop[i], shape[i])

                    max_bbox_mask.append(s_n)
                max_bbox_mask = tuple(max_bbox_mask)
                
                # Data augmentation: apply crop for healthy masks, pad if too small, crop if bigger
                t1n_crop_healthy, healthy_mask_crop = t1n[max_bbox_mask], healthy_mask[max_bbox_mask]
                
                t1n_crop_healthy, crop_box = pad3d(self.healthy_mask_crop, t1n_crop_healthy, max_bbox_mask)
                healthy_mask_crop, _ = pad3d(self.healthy_mask_crop, healthy_mask_crop)
                t1n_crop_healthy, healthy_mask_crop = random_crop(self.healthy_mask_crop, t1n_crop_healthy, healthy_mask_crop)
                # t1n_crop_healthy is the image cropped around healthy mask
                
                t1n_crop_healthy = t1n_crop_healthy.squeeze(0)  # Remove channel dimension
                print("Inspect shape after cropping and padding around healthy mask:", t1n_crop_healthy.shape)
                
                # Extract 2D slices
                for z in range(t1n_crop_healthy.shape[-1]):
                    t1n_slice = t1n_crop_healthy[:, :, z]
                    # Ensure t1n_slice is a NumPy array before checking for non-zero values
                    if not np.any(np.array(t1n_slice) > 0):  # Skip completely dark slices
                        print("Skipping completely dark slice")
                    else:
                        slice_path = self.output_dir / f"slice_{len(self.slices):05d}.npy"
                        np.save(slice_path, t1n_slice)
                        self.slices.append(slice_path)

        # Shuffle slices after appending
        random.shuffle(self.slices)

        # Add print statements for statistics
        print(f"[BraTS2D] Extracted and saved {len(self.slices)} slices.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_path = self.slices[idx]
        img = np.load(slice_path).astype(np.float32)

        # Convert to PIL Image as grayscale
        data_uint8 = (img * 255).astype(np.uint8)
        img = PImage.fromarray(data_uint8, mode='L')  # 'L' mode for grayscale

        if self.transform:
            img = self.transform(img)
        return img, 0

def build_braTS2d(data_path, crop_shape=(128, 128), train_split=0.95):
    
    standard_transforms = [
        transforms.ToTensor(),  # This will create 1-channel tensor for grayscale
        normalize_01_into_pm1,
    ]
    
    standard_transforms = transforms.Compose(standard_transforms)
    
    dataset = BraTS2DDataset(data_path, crop_shape=crop_shape, transform=standard_transforms)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    print(f"[BraTS2D] Total slices: {len(dataset)}, Train: {len(train_set)}, Validation: {len(val_set)}")
    return 0, train_set, val_set