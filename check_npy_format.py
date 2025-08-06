# Inspect the data format at /home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional/train/mid_brain
import os
import os.path as osp
import numpy as np

def inspect_data_format(data_path: str):
    class_dir = osp.join(data_path)
    if not osp.exists(class_dir):
        print(f"Directory {class_dir} does not exist.")
        return
    
    files = [f for f in os.listdir(class_dir) if f.endswith('.npy')]
    if not files:
        print("No .npy files found in the directory.")
        return
    
    step = 0
    for fname in files:
        file_path = osp.join(class_dir, fname)
        data = np.load(file_path)
        step += 1
        # print(f"File: {fname}, Shape: {data.shape}, Dtype: {data.dtype}")
    print(step)
        
if __name__ == "__main__":
    inspect_data_format('/home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional/train/mid_brain')