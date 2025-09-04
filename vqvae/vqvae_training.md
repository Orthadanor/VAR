# VQVAE Training with vaex-style Features

This directory contains a re-implementation of VQVAE training with features inspired by the vaex codebase, including auto-resume functionality and advanced learning rate scheduling.


## Data Preprocessing
The data preprocessing and augmentation steps are built upon the official repo for BraTS-inpainting challenge `https://github.com/BraTS-inpainting/2025_challenge.git`. Follow the instructions on the repo to download the ```ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training``` dataset locally in the following structure:

The BraTS2023 dataset is processed in the following way to produce 2d slices for training the VQVAE:


## Files Overview

### Core Training Files
- **`vqvae_trainer.py`** - VQVAE trainer class with learning rate scheduling and gradient clipping
- **`train_vqvae_vaex_style.py`** - Main training script with auto-resume functionality
- **`tensorboard_logger.py`** - TensorBoard logging utilities
- **`metric_logger.py`** - Metric tracking and logging
- **`checkpoint_saver.py`** - Checkpoint management and auto-resume

### Key Features

#### 1. **Auto-Resume Functionality**
- Automatically detects and resumes from the latest checkpoint
- Handles interrupted training gracefully
- Saves checkpoints every 10 epochs and best model based on validation loss

#### 2. **Advanced Learning Rate Scheduling**
- **Warmup Phase**: Linear warmup from `wp0` to peak learning rate
- **Decay Phase**: Multiple scheduling options:
  - `cos`: Cosine decay (recommended)
  - `lin`: Linear decay with plateau
  - `lin0`: Linear decay with short plateau
  - `lin00`: Pure linear decay
  - `exp`: Exponential decay

#### 3. **Comprehensive Logging**
- TensorBoard integration with detailed metrics
- Speed tracking and profiling
- Gradient norm monitoring
- Codebook usage statistics

#### 4. **Training Optimizations**
- Gradient accumulation support
- Gradient clipping
- Mixed precision training (FP16/BF16)
- Model compilation support
- Distributed training ready

## Usage

### Basic Training Command

```bash
cd /home/yuchenliu/VAR
```

Set the visible CUDA devices and run the training script:

```bash
export CUDA_VISIBLE_DEVICES=7,9

python vqvae/train_vqvae_vaex_style.py \
    --data_path /home/yuchenliu/Dataset/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training \
    --batch_size 64 \
    --epochs 100 \
    --final_reso 128 \
    --vocab_size 128 \
    --z_channels 8 \
    --ch 8 \
    --lr 1e-4 \
    --sche cos \
    --warmup_epochs 5 \
    --wp0 0.005 \
    --sche_end 0.001 \
    --fp16 \
    --compile_model \
    --prof \
    --save_dir ./local_output/vqvae_vaex_style
```

## Command Line Arguments

### Data Arguments
- `--data_path`: Path to MRI data directory
- `--batch_size`: Training batch size (default: 64)
- `--num_workers`: Number of data loading workers (default: 4)
- `--final_reso`: Final image resolution (default: 128)
- `--hflip`: Enable horizontal flip augmentation

### Model Arguments
- `--vocab_size`: Codebook size (default: 512)
- `--z_channels`: Embedding dimension (default: 16)
- `--ch`: Base channel width (default: 128)
- `--beta`: Commitment loss weight (default: 1.0)
- `--share_quant_resi`: Quantization residual sharing (default: 4)
- `--v_patch_nums`: Multi-scale patch numbers (default: [1, 2, 4, 8])

### Training Arguments
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--beta1`, `--beta2`: Adam optimizer betas (default: 0.9, 0.95)
- `--weight_decay`: Weight decay (default: 0.05)
- `--codebook_weight`: Codebook loss weight (default: 1.0)
- `--perceptual_weight`: Perceptual loss weight (default: 1.0)
- `--grad_clip`: Gradient clipping norm (default: 1.0)
- `--grad_accu`: Gradient accumulation steps (default: 1)

### Learning Rate Scheduling
- `--sche`: Schedule type: `cos`, `lin`, `lin0`, `lin00`, `exp` (default: cos)
- `--warmup_epochs`: Warmup epochs (default: 5)
- `--wp0`: Initial warmup learning rate ratio (default: 0.005)
- `--sche_end`: Final learning rate ratio (default: 0.001)

### Optimization
- `--fp16`: Enable FP16 mixed precision
- `--bf16`: Enable BF16 mixed precision
- `--zero`: Enable ZeRO optimization
- `--compile_model`: Enable torch.compile

### Logging and Saving
- `--save_dir`: Checkpoint save directory
- `--val_freq`: Validation frequency in steps (default: 20)
- `--prof`: Enable profiling

## Auto-Resume Behavior

The training script automatically handles resuming from checkpoints:

1. **Checkpoint Detection**: Looks for `vqvae_epoch_*.pth` files in the save directory
2. **Latest Checkpoint**: Automatically loads the most recent checkpoint
3. **State Restoration**: Restores model, optimizer, and trainer state
4. **Resume Training**: Continues from the exact epoch and iteration

### Manual Resume
To manually specify a checkpoint:
```bash
# The script will automatically find and load the latest checkpoint
python train_vqvae_vaex_style.py --save_dir ./local_output/vqvae_vaex_style ...
```

## Output Files

The training script creates several output files:

- **`vqvae_epoch_N.pth`**: Regular checkpoints (every 10 epochs)
- **`vqvae_best.pth`**: Best model based on validation loss
- **`vqvae_final.pth`**: Final model after training
- **`vqvae_interrupted.pth`**: Checkpoint saved on interruption
- **`tensorboard_logs/`**: TensorBoard log directory

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir ./local_output/vqvae_vaex_style/tensorboard_logs
```

### Key Metrics Tracked
- **Training Losses**: Total, Reconstruction, VQ, Perceptual
- **Validation Losses**: Same as training
- **Learning Rate**: Current and scheduled learning rates
- **Gradient Norms**: Gradient clipping information
- **Speed**: Iterations per second, images per second
- **Codebook Usage**: Perplexity and usage statistics

## Learning Rate Schedule Visualization

The learning rate follows this pattern:
1. **Warmup**: Linear increase from `wp0 * lr` to `lr` over `warmup_epochs`
2. **Decay**: Decrease from `lr` to `sche_end * lr` over remaining epochs

For cosine scheduling (`--sche cos`):
- Smooth cosine decay after warmup
- Recommended for most training scenarios

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or enable gradient accumulation
2. **Slow Training**: Enable `--compile_model` or `--fp16`
3. **Poor Convergence**: Adjust learning rate or warmup epochs
4. **Checkpoint Issues**: Check save directory permissions

### Performance Tips

1. **Use Mixed Precision**: Enable `--fp16` for faster training
2. **Model Compilation**: Enable `--compile_model` for speedup
3. **Gradient Accumulation**: Use `--grad_accu` for larger effective batch sizes
4. **Profiling**: Use `--prof` to identify bottlenecks

## Comparison with Original Script

| Feature | Original | vaex-style |
|---------|----------|------------|
| Auto-resume | ❌ | ✅ |
| LR Scheduling | Basic | Advanced (cos/lin/exp) |
| Gradient Clipping | ❌ | ✅ |
| Mixed Precision | ❌ | ✅ |
| Model Compilation | ❌ | ✅ |
| Comprehensive Logging | Basic | Advanced |
| Checkpoint Management | Manual | Automatic |
| Speed Tracking | ❌ | ✅ |
| Profiling | ❌ | ✅ |

