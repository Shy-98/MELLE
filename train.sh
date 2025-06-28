export CUDA_VISIBLE_DEVICES=1,2
torchrun \
--nnodes=1 \
--nproc_per_node=2 \
--master_port=12345 \
DDP_main.py \
--train_json data/librispeech_train960.jsonl \
--batch_frames 50000 \
--save_dir debug_exp \
--using_postnet \
--norm layer \
--transformer_activation relu \
--prenet_activation relu \
--postnet_activation tanh \
--exp_name  melle \
--save_interval 50000