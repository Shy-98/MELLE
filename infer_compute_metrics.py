import os
import json
import torch
import torchaudio
from tqdm import tqdm
from jiwer import compute_measures
from modules.asv_scripts.verification import init_model as init_asv_model

from transformers import Wav2Vec2Processor, HubertForCTC

test_ckpt_path = 'librispeech_exp/melle/step_400000.pt'

prompt_save_dir = test_ckpt_path.replace('.pt','testclean_prompt_samples')
generate_save_dir = test_ckpt_path.replace('.pt','testclean_generate_samples')

asr_processor = Wav2Vec2Processor.from_pretrained("hubert-large-ls960-ft")
asr_model = HubertForCTC.from_pretrained("hubert-large-ls960-ft").cuda().eval()

asv_model_cpt = "modules/asv_scripts/wavlm_large_finetune.pth"
asv_model = init_asv_model("wavlm_large", asv_model_cpt).cuda().eval()

def compute_wer(predictions, references):
    incorrect = 0
    total = 0
    totalS, totalD, totalI = 0, 0, 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        H, S, D, I = measures["hits"], measures["substitutions"], measures["deletions"], measures["insertions"]
        totalS += S
        totalD += D
        totalI += I
        incorrect += S + D + I
        total += S + D + H

    return {
        # "wer": incorrect / float(total),
        "n_words": total,
        "n_incorrections": incorrect,
        "n_substitutions": totalS,
        "n_deletions": totalD,
        "n_insertions": totalI,
    }

def read_jsonl(jsonl_path):
    all_jsonl = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            all_jsonl.append(item)
    return all_jsonl

prompt_datas = read_jsonl('data/librispeech_testclean_prompt.jsonl')
generate_datas = read_jsonl('data/librispeech_testclean_generate.jsonl')

results = []
pbar = tqdm(total=len(prompt_datas), desc=f"Generating ......", ncols=100)
for prompt_item, generate_item in zip(prompt_datas, generate_datas):

    prompt_key = os.path.splitext(os.path.basename(prompt_item['audio_path']))[0]
    generate_key = os.path.splitext(os.path.basename(generate_item['audio_path']))[0]
    prompt_wav_path = os.path.join(prompt_save_dir, f'{prompt_key}.npy_gen.wav')
    origi_wav_path = os.path.join(prompt_save_dir, f'{generate_key}.npy_gen.wav')
    generate_wav_path = os.path.join(generate_save_dir, f'{generate_key}.npy_gen.wav')

    prompt_wav, _ = torchaudio.load(prompt_wav_path)
    origi_wav, _ = torchaudio.load(origi_wav_path)
    generate_wav, _ = torchaudio.load(generate_wav_path)
    prompt_wav = prompt_wav.cuda()
    origi_wav = origi_wav.cuda()
    generate_wav = generate_wav.cuda()
    if generate_wav.shape[1] < int(16000*0.25):
        generate_wav = torch.cat([generate_wav, generate_wav.new_zeros((1, int(16000*0.25)-generate_wav.shape[1]))], dim=1)
    # ===============ASR=============== #
    with torch.no_grad():
        logits = asr_model(generate_wav).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hubert_transcription = asr_processor.decode(predicted_ids[0])
        hubert_transcription = hubert_transcription.lower()
        hubert_wer_info = compute_wer(references=[generate_item['transcription'].lower()], predictions=[hubert_transcription])
        # ===============ASR=============== #

        # ===============ASV=============== #
        emb1 = asv_model(prompt_wav)
        emb2 = asv_model(generate_wav)
        embr = asv_model(origi_wav)
        sim_r = torch.nn.functional.cosine_similarity(emb1, emb2).cpu().item()
        sim_o = torch.nn.functional.cosine_similarity(embr, emb2).cpu().item()
        # ===============ASV=============== #

    results.append(
        {
            'wav_name': generate_key,
            'ref': generate_item['transcription'].lower(),
            "hubert_wer_info": hubert_wer_info, 
            "hubert_transcription": hubert_transcription, 
            "wer": hubert_wer_info["n_incorrections"],
            "spk_sim_r": sim_r, 
            "spk_sim_o": sim_o, 
            "spk_sim_avg": (sim_r+sim_o)/2.0
        }
    )
    pbar.update(1)

# 创建结果字典，包含不同的选择策略
res = {
    "all": results,
    "best_hubert_wer": results, 
    "best_sim_r": results,
    "best_sim_o": results,
    "best_sim_avg": results,
    "best_sorted_metric": results,
    "rand": results
}

# 计算并保存结果
result_path = test_ckpt_path.replace('.pt', 'testclean_results.txt')
os.makedirs(os.path.dirname(result_path), exist_ok=True)

# 汇总结果写入主文件
with open(result_path, "w", encoding="utf-8") as main_f:
    # for _key in res.keys():
    _key = 'all'
    words_num = sum(x["hubert_wer_info"]["n_words"] for x in res[_key])
    
    # 计算hubert WER统计
    hubert_error_words_num = sum(x["hubert_wer_info"]["n_incorrections"] for x in res[_key])
    hubert_n_substitutions = sum(x["hubert_wer_info"]["n_substitutions"] for x in res[_key])
    hubert_n_insertions = sum(x["hubert_wer_info"]["n_insertions"] for x in res[_key])
    hubert_n_deletions = sum(x["hubert_wer_info"]["n_deletions"] for x in res[_key])
    hubert_wer = hubert_error_words_num * 100.0 / words_num if words_num > 0 else 0
    
    # 计算说话人相似度
    avg_spk_sim_r = sum(x["spk_sim_r"] for x in res[_key]) / len(res[_key])
    avg_spk_sim_o = sum(x["spk_sim_o"] for x in res[_key]) / len(res[_key])
    
    # 构建结果字符串
    hubert_asr_str = (
        f"hubert wer: {hubert_wer:.2f}% | "
        f"E / N: {hubert_error_words_num} / {words_num} | "
        f"S: {hubert_n_substitutions} | "
        f"I: {hubert_n_insertions} | "
        f"D: {hubert_n_deletions}"
    )
    
    sim_str = (
        f"spk_sim_r: {avg_spk_sim_r:.4f} | "
        f"spk_sim_o: {avg_spk_sim_o:.4f}"
    )
    
    # 打印并写入主文件
    print(f"\n{_key} result:")
    print(hubert_asr_str)
    print(sim_str)
    
    main_f.write(f"{_key}\n")
    main_f.write(hubert_asr_str + "\n")
    main_f.write(sim_str + "\n\n")
    
    # 为每个策略创建详细结果文件
    detail_path = result_path.replace(".txt", f"_{_key}.txt")
    with open(detail_path, "w", encoding="utf-8") as detail_f:
        detail_f.write(f"Strategy: {_key}\n\n")
        for item in res[_key]:
            detail_line = (
                f"{item['wav_name']} | "
                f"Words: {item['hubert_wer_info']['n_words']} | "
                f"Errors: {item['hubert_wer_info']['n_incorrections']} | "
                f"Sim_R: {item['spk_sim_r']:.4f} | "
                f"Sim_O: {item['spk_sim_o']:.4f}\n"
                f"REF: {item['ref']}\n"
                f"PRED: {item['hubert_transcription']}\n"
                f"{'-'*80}\n"
            )
            detail_f.write(detail_line)
        
        detail_f.write("\nSUMMARY:\n")
        detail_f.write(hubert_asr_str + "\n")
        detail_f.write(sim_str + "\n")

print(f"Results saved to: {result_path}")