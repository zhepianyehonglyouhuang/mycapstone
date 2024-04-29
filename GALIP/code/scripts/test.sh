cfg=$1
batch_size=32

pretrained_model='/content/drive/MyDrive/capstone5703/GALIP/code/saved_models/bird/GALIP_nf64_gpu8MP_True_bird_256_2024_04_07_05_11_00/state_epoch_004.pth'
multi_gpus=False
mixed_precision=True

nodes=1
num_workers=8
master_port=11277
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0 python src/test.py \
              --stamp $stamp \
              --cfg $cfg \
              --mixed_precision $mixed_precision \
              --batch_size $batch_size \
              --num_workers $num_workers \
              --multi_gpus $multi_gpus \
              --pretrained_model_path $pretrained_model \

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port=$master_port src/test.py \
#                     --stamp $stamp \
#                     --cfg $cfg \
#                     --mixed_precision $mixed_precision \
#                     --batch_size $batch_size \
#                     --num_workers $num_workers \
#                     --multi_gpus $multi_gpus \
#                     --pretrained_model_path $pretrained_model \
