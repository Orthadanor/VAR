prepare_mri_dataset.py

export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 \
  train.py --depth=16 --bs=8 --ep=10 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --data_path=/home/yuchen.liu/imagenette2-160


cd /home/yuchenliu/VAR

CUDA_VISIBLE_DEVICES=2,8 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 train.py \
  --depth=16 --bs=8 --ep=20 --fp16=1 --alng=1e-3 --wpe=0.1 \
  --data_path=/home/yuchenliu/Dataset/IXI/t1_np_masked_128_unconditional