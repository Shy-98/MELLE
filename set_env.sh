conda create -n melle python==3.10 -y
conda activate melle
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy tqdm jiwer webrtcvad librosa
CUDA_VISIBLE_DEVICES=1,2 torchrun --nnodes=1 --nproc_per_node=2 --master_port=12345 DDP_main.py --train_json data/librispeech_train960.jsonl --batch_frames 50000 --save_dir debug_exp --using_postnet --norm layer --transformer_activation relu --prenet_activation relu --postnet_activation tanh --exp_name  melle --save_interval 50000