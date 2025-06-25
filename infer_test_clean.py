from transformers import LlamaTokenizerFast, Qwen2TokenizerFast
import numpy as np
import os
import torch
from modules.traing_utils import get_fbank_from_wav, set_seed, VADSplitter
from tqdm import tqdm
import json
from MELLE import MELLE

set_seed(111)
device = 'cuda:7'
ckpt_path = 'librispeech_exp/melle/step_400000.pt'
model = MELLE(
    using_rope=False,
    using_postnet=True,
    using_qwen2mlp=False,
    norm='layer',
    transformer_activation='relu',
    prenet_activation='relu',
    postnet_activation='tanh',
).to(device)
# ============================== #
# vad_splitter = VADSplitter(
#             aggressiveness=3,
#             sample_rate=16000,
#             frame_duration=30
#         )
vad_splitter = None
# ============================== #

def read_jsonl(jsonl_path):
    all_jsonl = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            all_jsonl.append(item)
    return all_jsonl

text_tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            add_bos_token=True,
            add_eos_token=True,
            )

checkpoint = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

prompt_save_dir = ckpt_path.replace('.pt','testclean_prompt_samples')
generate_save_dir = ckpt_path.replace('.pt','testclean_generate_samples')
os.makedirs(prompt_save_dir, exist_ok=True)
os.makedirs(generate_save_dir, exist_ok=True)

prompt_datas = read_jsonl('data/librispeech_testclean_prompt.jsonl')
generate_datas = read_jsonl('data/librispeech_testclean_generate.jsonl')

pbar = tqdm(total=len(prompt_datas), desc=f"Generating ......", ncols=100)
for prompt_item, generate_item in zip(prompt_datas, generate_datas):
    prompt_mel = get_fbank_from_wav(prompt_item['audio_path'], vad_splitter).unsqueeze(0).to(device)

    mel_lengths = torch.tensor(prompt_mel.shape[1]).long().reshape(1).to(device)
    if isinstance(text_tokenizer, LlamaTokenizerFast):
        txt = torch.tensor(text_tokenizer.encode(prompt_item['transcription']+" "+generate_item['transcription'])).long().reshape(1,-1).to(device)
    elif isinstance(text_tokenizer, Qwen2TokenizerFast):
        txt = torch.tensor(text_tokenizer(prompt_item['transcription']+" "+generate_item['transcription'])["input_ids"]).long().reshape(1,-1).to(device) + 1
    txt_lengths = torch.tensor(txt.shape[1]).long().reshape(1).to(device)
    outputs = model.inference(prompt_mel, txt, max_length=2000)

    prompt_key = os.path.splitext(os.path.basename(prompt_item['audio_path']))[0]
    generate_key = os.path.splitext(os.path.basename(generate_item['audio_path']))[0]
    np.save(os.path.join(prompt_save_dir, f'{prompt_key}.npy'), prompt_mel[0].cpu().numpy())
    np.save(os.path.join(generate_save_dir, f'{generate_key}.npy'), outputs[0].cpu().numpy())
    pbar.update(1)