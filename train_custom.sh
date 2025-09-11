# VQVAE Training
export CUDA_VISIBLE_DEVICES=2,3

python train_vqvae.py \
    --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split_single_slice \
    --batch_size 64 \
    --epochs 100 \
    --final_reso 128 \
    --vocab_size 128 \
    --z_channels 8 \
    --ch 8 \
    --lr 1e-4 \
    --sche cos \
    --warmup_epochs 5 \
    --save_dir ./local_output/vqvae_test_if_still_works_ixi

################## VQVAE for 3D #########################
# python vqvae/train_vqvae_vol.py \
#     --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split_multislice \
#     --batch_size 8 \
#     --epochs 250 \
#     --final_reso 128 \
#     --vocab_size 128 \
#     --z_channels 8 \
#     --ch 32 \
#     --lr 1e-4 \
#     --num_slices 10 \
#     --volume_lpips \
#     --no_recon \
#     --save_dir ./local_output/vqvae_test

########## OLD VERSION VQVAE TRAINING ###################
# python train_vqvae_multiscale.py \
#     --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split_single_slice \
#     --batch_size 64 \
#     --epochs 200 \
#     --final_reso 128 \
#     --vocab_size 128 \
#     --z_channels 16 \
#     --ch 64 \
#     --lr 1e-4 \
#     --val_freq 20 \
#     --save_dir ./local_output/vqvae_test_if_still_works_ixi

# # Convert VQVAE checkpoint format for VAR training
# python convert_vqvae_checkpoint.py

# VAR Training with your custom VQVAE
# export CUDA_VISIBLE_DEVICES=7,9
# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 \
#   train.py \
#   --data_path /home/yuchenliu/Dataset/IXI/train_val_test_split \
#   --local_out_dir_path=./local_output/var_custom_v128_z16_c64_lpips_d8_h8 \
#   --pn="1_2_4_8" \
#   --depth=8 \
#   --num_heads=8 \
#   --bs=256 \
#   --ep=250 \
#   --fp16=1 \
#   --tblr=2e-4 \
#   --tclip=2.0 \
#   --ac=1 \
#   --workers=4