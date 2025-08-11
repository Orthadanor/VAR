# VQVAE Training
# python train_vqvae_multiscale.py \
#     --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split \
#     --batch_size 64 \
#     --epochs 200 \
#     --final_reso 128 \
#     --vocab_size 128 \
#     --z_channels 16 \
#     --ch 64 \
#     --lr 1e-4 \
#     --val_freq 20 \
#     --save_dir ./local_output/vqvae_checkpoints_v128_z16_c64_b1_lpips

# # Convert VQVAE checkpoint format for VAR training
# python convert_vqvae_checkpoint.py

# VAR Training with your custom VQVAE
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 \
  train.py \
  --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split \
  --local_out_dir_path=./local_output/var_custom_v128_z16_c64_b1_lpips \
  --pn="1_2_4_8" \
  --depth=16 \
  --bs=256 \
  --ep=250 \
  --fp16=1 \
  --tblr=2e-4 \
  --tclip=2.0 \
  --ac=1 \
  --workers=4