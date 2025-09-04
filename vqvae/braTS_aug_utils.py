# Diverse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Datasets
from torch.utils.data import Dataset, random_split

from typing import Tuple, Optional
from torch.nn import functional as F
import random


from math import floor, ceil
import matplotlib.pyplot as plt
from matplotlib import colors  # for_ custom colormap
from scipy import ndimage  # for: getting boundaries of image
# from dataset3D import Dataset_Training, Dataset_Inference

# Training
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
# from baseline_utils import plot_3D
import nibabel as nib

# Models
# from baseline_utils import get_latest_Checkpoint  # to load last checkpoint
# from train_Pix2Pix3D import Pix2Pix3D
# from train_AE import AutoEncoder

# Processes
import subprocess
import sys

# plot single image, 5 slices
# helper function to plot transparent mask with boundary on given axis
def _plot_mask(axis, mask, color=(0.90, 0.60, 0.0, 1.0), alpha_mask=0.5, alpha_boundary=1.0):
    cmap = colors.ListedColormap([(0, 0, 0, 0), color])  # zeros: transparent  # ones: provided_color
    axis.imshow(mask, cmap=cmap, alpha=alpha_mask, interpolation="nearest")
    boundary = ndimage.morphology.binary_dilation(mask, np.ones((3, 3), dtype=bool)) != mask  # get boundaries
    axis.imshow(boundary, cmap=cmap, alpha=alpha_boundary)
    
def plot_3D(image, healthyMask=None, unhealthyMask=None, generalMask=None, steps_size=15, cmap_image="gray", dpi=100):
    """Input might be CDWH or DWH.

    COmputes center slice

    Args: [TODO]
        mask (_type_): _description_
        steps_size (int, optional): _description_. Defaults to 15.
        cmap_image (str, optional): _description_. Defaults to "gray".
    """
    shape = image.shape

    # sliceIDs = list(range(0, shape[-1], steps_size)) #all slices with distance steps_size
    middle = int(shape[-1] / 2)  # middle slice index
    sliceIDs = [middle - 2 * steps_size, middle - steps_size, middle, middle + steps_size, middle + 2 * steps_size]
    size = (shape[-3] / 60.0, shape[-2] / 60.0)  # D/120, W/120

    # Reduce dimensions CWDH -> WDH if necessary

    fig, ax = plt.subplots(figsize=(size[0] * len(sliceIDs), size[1]), nrows=1, ncols=len(sliceIDs), squeeze=True, sharey=True, dpi=dpi)

    for i, sliceID in enumerate(sliceIDs):
        # plot image
        # ax[i].set_title(sliceID)
        img = image[0, :, :, sliceID].T if len(shape) == 4 else image[:, :, sliceID].T
        ax[i].imshow(img, cmap=cmap_image, interpolation="nearest")

        # plot masks if provided
        if not healthyMask is None:
            img = healthyMask[0, :, :, sliceID].T if len(healthyMask.shape) == 4 else healthyMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.0, 0.80, 0.20, 1.0))  # green

        if not unhealthyMask is None:
            img = unhealthyMask[0, :, :, sliceID].T if len(unhealthyMask.shape) == 4 else unhealthyMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.90, 0.10, 0.10, 1.0))  # red

        if not generalMask is None:
            img = generalMask[0, :, :, sliceID].T if len(generalMask.shape) == 4 else generalMask[:, :, sliceID].T
            img = np.array(img).astype(bool)
            _plot_mask(ax[i], img, color=(0.90, 0.60, 0.0, 1.0))  # orange

    plt.show()

def compute_bbox(array, minimum=0, dist=0, zooms=(1, 1, 1)):
    """
    Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

    Args:
        minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
        dist (int): The amount of padding to be added to the cropped image. Default value is 0.
    Returns:
        ex_slice: A tuple of slice objects that need to be applied to crop the image.
        origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

    Note:
        - The computed slice removes the unused space from the image based on the minimum value.
        - The padding is added to the computed slice.
        - If the computed slice reduces the array size to zero, a ValueError is raised.
    """
    shp = array.shape
    d = np.around(dist / np.asarray(zooms)).astype(int)
    msk_bin = np.zeros(array.shape, dtype=bool)
    msk_bin[array > minimum] = 1
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)

    if cor_msk[0].shape[0] == 0:
        raise ValueError("Array would be reduced to zero size")

    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
    y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
    z0 = c_min[2] - d[2] if (c_min[2] - d[2]) > 0 else 0
    x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[0] else shp[0]
    y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[1] else shp[1]
    z1 = c_max[2] + d[2] if (c_max[2] + d[2]) < shp[2] else shp[2]

    bbox = tuple([slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1)])
    # bbox = ((x0, x1 + 1), (y0, y1 + 1), (z0, z1 + 1))

    return bbox

def pad3d(size, image, max_bbox=None) -> Tuple[torch.Tensor, Optional[Tuple[slice, slice, slice]]]:
    image = torch.Tensor(image)
    d, w, h = image.shape[-3], image.shape[-2], image.shape[-1]
    d_max, w_max, h_max = size
    d_pad = max((d_max - d) / 2, 0)
    w_pad = max((w_max - w) / 2, 0)
    h_pad = max((h_max - h) / 2, 0)
    padding = (
        int(floor(h_pad)),
        int(ceil(h_pad)),
        int(floor(w_pad)),
        int(ceil(w_pad)),
        int(floor(d_pad)),
        int(ceil(d_pad)),
    )
    x = F.pad(image, padding, value=0, mode="constant")

    if max_bbox is not None:
        max_bbox = list(max_bbox)
        for i, s in enumerate(max_bbox, 1):
            s: slice
            pad_e = padding[2 * -i + 1]
            pad_s = padding[2 * -i]
            max_bbox[i - 1] = slice(s.start - pad_s, s.stop + pad_e)
        max_bbox = tuple(max_bbox)
    return x, max_bbox

def random_crop(target_shape: Tuple[int, int, int], *arrs: torch.Tensor):
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)

def normalize(tensor):
    return (tensor * 2) - 1  # map [0,1]->[-1,1]

class Dataset_Training(Dataset):
    """Dataset for Training purposes with preprocessing and augmentation for BraTS dataset."""

    def __init__(self, root_dir, crop_shape=(128, 128, 96), center_on_mask=False):
        dataset_path = Path(root_dir) or Path("../../../Dataset/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training")
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask

        self.list_paths_t1n = list(dataset_path.rglob("**/BraTS-GLI-*-*-t1n.nii.gz"))
        self.list_paths_mask_healthy = list(dataset_path.rglob("**/BraTS-GLI-*-*-mask-healthy.nii.gz"))

    def __len__(self):
        return len(self.list_paths_mask_healthy)

    def preprocess(self, t1n: np.ndarray, healthy_mask: np.ndarray):
        referenceShape = (240, 240, 155)
        if t1n.shape != referenceShape or healthy_mask.shape != referenceShape:
            raise UserWarning(f"Invalid shape: {t1n.shape}, {healthy_mask.shape}")

        t1n[t1n < 0] = 0
        t1n_max_v = np.max(t1n)
        t1n /= t1n_max_v

        if self.center_on_mask:
            shape = healthy_mask.shape[-3:]
            min_bbox = compute_bbox(healthy_mask)
            max_bbox = []
            for i, s in enumerate(min_bbox):
                d = self.crop_shape[i] - (s.stop - s.start)
                s_n = slice(s.start - d // 2, s.stop + ceil(d / 2))
                if s_n.start < 0:
                    s_n = slice(0, self.crop_shape[i])
                if s_n.stop > shape[i]:
                    s_n = slice(shape[i] - self.crop_shape[i], shape[i])
                max_bbox.append(s_n)
            max_bbox = tuple(max_bbox)
        else:
            max_bbox = compute_bbox(t1n)

        t1n_crop = t1n[max_bbox]
        healthy_mask_crop = healthy_mask[max_bbox]

        t1n_crop, crop_box = pad3d(self.crop_shape, t1n_crop, max_bbox)
        healthy_mask_crop, _ = pad3d(self.crop_shape, healthy_mask_crop)

        t1n_crop, healthy_mask_crop = random_crop(self.crop_shape, t1n_crop, healthy_mask_crop)

        t1n_voided_healthy_crop = t1n_crop * (1 - healthy_mask_crop)

        t1n_crop = normalize(t1n_crop)
        t1n_voided_healthy_crop = normalize(t1n_voided_healthy_crop)

        t1n_crop = t1n_crop.unsqueeze(0)
        healthy_mask_crop = healthy_mask_crop.unsqueeze(0)
        t1n_voided_healthy_crop = t1n_voided_healthy_crop.unsqueeze(0)

        healthy_mask_crop = healthy_mask_crop.bool()

        return t1n_voided_healthy_crop, t1n_max_v, crop_box, t1n_crop, healthy_mask_crop

    def __getitem__(self, idx):
        t1n_path = self.list_paths_t1n[idx]
        t1n_img = nib.load(t1n_path)
        t1n = t1n_img.get_fdata()

        healthy_mask_path = self.list_paths_mask_healthy[idx]
        healthy_mask_img = nib.load(healthy_mask_path)
        healthy_mask = healthy_mask_img.get_fdata()

        t1n_voided_healthy_crop, t1n_max_v, crop_box, t1n_crop, healthy_mask_crop = self.preprocess(t1n, healthy_mask)

        sample_dict = {
            "gt_image": t1n_crop,
            "voided_healthy_image": t1n_voided_healthy_crop,
            "t1n_path": str(t1n_path),
            "healthy_mask": healthy_mask_crop,
            "healthy_mask_path": str(healthy_mask_path),
            "cropped_bbox": str(crop_box),
            "max_v": t1n_max_v,
            "name": healthy_mask_path.name[:19],
        }

        return sample_dict

    @classmethod
    def braTS2d(cls, root_dir, crop_shape=(128, 128), train_split=0.95):
        datasetTrain = cls(root_dir=root_dir, crop_shape=(240, 240, 155))
        slices = []

        for idx in range(len(datasetTrain)):
            sample = datasetTrain[idx]
            volume = sample['gt_image'].squeeze(0)
            central_crop = volume[56:184, 56:184, :]

            for z in range(central_crop.shape[-1]):
                slice_2d = central_crop[:, :, z].numpy()  # Ensure slice_2d is a numpy array
                if np.any(slice_2d > 0):
                    slices.append((slice_2d / 127.5 - 1).astype(np.float32))

        datasetTrain_cropped = cls(root_dir=root_dir, crop_shape=(128, 128, 96), center_on_mask=True)
        for idx in range(len(datasetTrain_cropped)):
            sample = datasetTrain_cropped[idx]
            volume = sample['gt_image'].squeeze(0)

            for z in range(volume.shape[-1]):
                slice_2d = volume[:, :, z].numpy()  # Ensure slice_2d is a numpy array
                if np.any(slice_2d > 0):
                    slices.append((slice_2d / 127.5 - 1).astype(np.float32))

        print(f"[BraTS2D] Total slices: {len(slices)}")

        dataset = BraTS2DDataset(slices)
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        return train_set, val_set

class BraTS2DDataset(Dataset):
    def __init__(self, slices):
        self.slices = slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        return self.slices[idx], 0