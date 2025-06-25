from transformers import LlamaTokenizerFast, Qwen2TokenizerFast
import numpy as np
import os
import torch
from modules.traing_utils import get_fbank_from_wav, set_seed, VADSplitter

set_seed(111)

from MELLE import MELLE
device = 'cuda:0'

text_tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            add_bos_token=True,
            add_eos_token=True,
            )
ckpt_path = 'librispeech_exp/melle_vad/step_400000.pt'
model = MELLE(
    using_rope=False,
    using_postnet=True,
    using_qwen2mlp=False,
    norm='layer',
    transformer_activation='relu',
    prenet_activation='relu',
    postnet_activation='tanh',
).to(device)
vad_splitter = VADSplitter(
            aggressiveness=3,
            sample_rate=16000,
            frame_duration=30
        )
# vad_splitter = None

checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

prompt_item =\
{"transcription": "WAS THE REPLY OF THE YOUTHFUL SALESMAN THE CAPTAIN HERE TOLD THOMPSON TO TAKE SIX CORDS WHICH WOULD LAST TILL DAYLIGHT AND AGAIN TURNED HIS ATTENTION TO THE GAME THE PILOTS HERE CHANGED PLACES WHEN DID THEY SLEEP WOOD TAKEN IN", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/5456/62043/5456-62043-0027.flac"}

generate_item =\
{"transcription": "AND THEN OF OTHER THINGS THE EVENING WAS KIND AND GENIAL AND SO WAS MY COMPANION BY DEGREES I WAXED MORE WARM AND TENDER THAN PERHAPS I HAD EVER BEEN BEFORE", "audio_path": "/root/epfs/data/LibriSpeech/LibriSpeech/train-clean-100/226/131532/226-131532-0016.flac"}

prompt_mel = get_fbank_from_wav(prompt_item['audio_path'], vad_splitter).unsqueeze(0).to(device)

mel_lengths = torch.tensor(prompt_mel.shape[1]).long().reshape(1).to(device)
if isinstance(text_tokenizer, LlamaTokenizerFast):
    txt = torch.tensor(text_tokenizer.encode(prompt_item['transcription']+" "+generate_item['transcription'])).long().reshape(1,-1).to(device)
elif isinstance(text_tokenizer, Qwen2TokenizerFast):
    txt = torch.tensor(text_tokenizer(prompt_item['transcription']+" "+generate_item['transcription'])["input_ids"]).long().reshape(1,-1).to(device) + 1
txt_lengths = torch.tensor(txt.shape[1]).long().reshape(1).to(device)
outputs = model.inference(prompt_mel, txt, max_length=2000)
np.save('test_infer.npy', outputs[0].cpu().numpy())
np.save('test_prompt.npy', prompt_mel[0].cpu().numpy())