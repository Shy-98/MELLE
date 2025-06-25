import os
import json
import torch
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor
import queue
from modules.traing_utils import VADSplitter, get_fbank_from_wav

from transformers import Qwen2TokenizerFast, LlamaTokenizerFast
import numpy as np


class DynamicBatchingDataset(IterableDataset):
    def __init__(
        self,
        jsonl_path,
        max_frames=16000,
        buffer_size=5000,
        min_sample_len=10,
        max_sample_len=5000,
        shuffle_buffer=True,
        rank=0,              # 新增参数：当前进程的rank
        world_size=1,        # 新增参数：总进程数
        seed=42,             # 新增参数：随机种子
        feature_name='fbank',
        shuffer_random_key='',
        using_vad=False,
    ):
        self.audio_key = 'audio_path'
        self.feature_name = feature_name
        self.jsonl_path = jsonl_path
        self.max_frames = max_frames
        self.buffer_size = buffer_size
        self.shuffle_buffer = shuffle_buffer
        self.min_sample_len = min_sample_len
        self.max_sample_len = max_sample_len
        
        self.shuffled_path = os.path.splitext(jsonl_path)[0]+'_shuffled_tmp_'+shuffer_random_key+'.jsonl'
        
        self.rank = rank
        self.world_size = world_size
        print(f'rank {self.rank} worldsize {self.world_size}')
        self.seed = seed
        if using_vad:
            print('using vad')
            self.vad_splitter = VADSplitter(
                aggressiveness=3,
                sample_rate=16000,
                frame_duration=30
            )
        else:
            print('don\'t use vad')
            self.vad_splitter = None
        # self.text_tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
        self.text_tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer",
            add_bos_token=True,
            add_eos_token=True,
            )

    def _parse_item(self, item):
        if not os.path.exists(item[self.audio_key]):
            return None
        try:
            '''feat: torch.tensor format'''
            # feat_save_path = item[self.audio_key].replace('/root/epfs/data/LibriSpeech', '/root/epfs/data/shy_data/2025/MELLE/data/features').replace('.flac', '.pt')
            # if not os.path.exists(feat_save_path):
            #     feat = get_fbank_from_wav(item[self.audio_key], self.vad_splitter).T.unsqueeze(0)
            #     os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
            #     torch.save(feat, feat_save_path)
            # else:
            #     feat = torch.load(feat_save_path)
            feat = get_fbank_from_wav(item[self.audio_key], self.vad_splitter).T.unsqueeze(0)
            feat_len = feat.shape[2]

            if isinstance(self.text_tokenizer, LlamaTokenizerFast):
                text_ids = torch.tensor(self.text_tokenizer.encode(item['transcription']))
            elif isinstance(self.text_tokenizer, Qwen2TokenizerFast):
                text_ids = torch.tensor(self.text_tokenizer(item['transcription'])["input_ids"]) + 1

        except Exception as e:
            print(f"Error loading {item[self.audio_key]}: {e}")
            return None
        
        result = {
            "feat": feat.squeeze(0),
            "feat_len": feat_len,
            "text": text_ids.clone().detach().to(torch.long),
            "text_len": len(text_ids),
            "sample_path": item[self.audio_key],
        }
        return result
    

    def _stream_data(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # 修改1：使用rank区分不同进程的缓存文件
        shuffled_path = f"{self.shuffled_path}.rank"
        with open(shuffled_path, 'r') as f:
            lines = list(f)
            chunk_size = len(lines) // (num_workers * self.world_size)
            start = ((num_workers * self.rank) + worker_id)  * chunk_size 
            end = min((((num_workers * self.rank) + worker_id) + 1) * chunk_size, len(lines))
            print(f'loading data for worker{worker_id}/rank{self.rank}/worldsize{self.world_size} from {start} to {end} / {len(lines)}')
            for line in lines[start:end]:
                item = json.loads(line)
                parsed = self._parse_item(item)
                if parsed is not None:
                    yield parsed
            f.close()
                    
    def shuffer_data(self):
        torch.distributed.barrier()
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        shuffled_path = f"{self.shuffled_path}.rank"

        if self.rank == 0 and worker_id == 0:
            if os.path.exists(shuffled_path):
                os.system(f'rm {shuffled_path}')
            if not os.path.exists(shuffled_path):
                temp_path = shuffled_path + '.tmp'
                os.system(f'shuf {self.jsonl_path} -o {temp_path}')
                os.rename(temp_path, shuffled_path)
                print(f'Rank {self.rank} generated shuffled file')

        torch.distributed.barrier()
            

    def buffer_generate_batch(self, buffer):
        # 按特征长度排序（降序）
        buffer.sort(key=lambda x: x["feat_len"], reverse=True)
        
        # 动态组batch
        batches = []
        batch = []
        batch_frames = 0
        batch_max_feat_len = -1
        batch_max_txt_len = -1
        
        for item_idx, item in enumerate(buffer):
            batch_max_feat_len = max(batch_max_feat_len, item["feat_len"])
            batch_max_txt_len = max(batch_max_txt_len, item["text_len"])
            if (batch_max_feat_len+batch_max_txt_len) * (len(batch)+1) > self.max_frames:
            # if batch_frames + item["feat_len"] > self.max_frames:
                if batch:  # 提交当前batch
                    batches.append(batch)
                    batch = []
                    batch_frames = 0
                    batch_max_feat_len = item["feat_len"]
                    batch_max_txt_len = item["text_len"]
            
            # batch_max_sample_len = max(batch_max_sample_len, item["feat_len"])
            batch.append(item)
            batch_frames += item["feat_len"]
        
        return batches, batch, batch_frames

    def __iter__(self):
        buffer = []
        data_queue = queue.Queue(maxsize=self.buffer_size * 2)  # 缓冲队列
    
        def producer():
            for sample in self._stream_data():
                if self.min_sample_len <= sample["feat_len"] <= self.max_sample_len:
                    data_queue.put(sample)
            data_queue.put(None)  # 结束信号
        
        # 启动生产者线程
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(producer)
            
            while True:
                sample = data_queue.get()
                if sample is None:  # 结束信号
                    break
                    
                buffer.append(sample)
                # print(len(buffer))
                if len(buffer) >= self.buffer_size:
                    batches, remaining, _ = self.buffer_generate_batch(buffer)
                    buffer = []
                    for batch in batches:
                        yield self._collate_fn(batch)
                    buffer += remaining
            
            # 处理剩余数据
            if buffer:
                batches, _, _ = self.buffer_generate_batch(buffer)
                for batch in batches:
                    yield self._collate_fn(batch)
            yield None

    def _collate_fn(self, batch):
        if batch is None or len(batch)==0:
            return None
        # 特征padding
        feat_dims = batch[0]["feat"].shape[0]
        max_feat_len = max(x["feat_len"] for x in batch)
        feats = torch.zeros(len(batch), feat_dims, max_feat_len)
        feat_lens = []
        
        for i, item in enumerate(batch):
            feats[i, :, :item["feat_len"]] = item["feat"]
            feat_lens.append(item["feat_len"])
        
        # 文本padding
        max_text_len = max(x["text_len"] for x in batch)
        texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        text_lens = []
        
        for i, item in enumerate(batch):
            texts[i, :item["text_len"]] = item["text"]
            text_lens.append(item["text_len"])
        
        return {
            "mel": feats.permute(0,2,1),                    # [B, dim, T] --> [B, T, dim]
            "mel_lengths": torch.tensor(feat_lens),  # [B]
            "txt": texts,                    # [B, S] padding_idx=0
            "txt_lengths": torch.tensor(text_lens),   # [B]
        }
