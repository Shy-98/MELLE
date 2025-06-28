conda create -n melle python==3.10 -y
conda activate melle
pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu118
pip install transformers numpy tqdm jiwer webrtcvad librosa