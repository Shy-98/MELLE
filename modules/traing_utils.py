import logging
import os
import random
import torch
import numpy as np
import librosa
import webrtcvad
from typing import List, Tuple
import json
import argparse

# 保存参数到JSON文件
def save_args(args, path):
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

# 从JSON文件加载参数
def load_args(parser, path):
    with open(path, 'r') as f:
        args_dict = json.load(f)
    
    # 创建命名空间对象
    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)
    
    # 验证参数（可选）
    parser.parse_args(namespace=args)
    return args

def setup_logger(log_dir, local_rank):
    """配置日志系统"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if local_rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
    return logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 自定义学习率调度器
def lr_lambda(current_step, args):
    step = current_step + 1  # 转换为1-based计数
    if step <= args.warmup_steps:
        # 线性预热阶段
        warmup_ratio = step / args.warmup_steps
        warmup_lr = args.warmup_init_lr + \
                (args.lr - args.warmup_init_lr) * warmup_ratio
        return warmup_lr / args.lr  # 返回相对于peak_lr的比例
    else:
        # 逆平方根衰减阶段
        decay_factor = (args.warmup_steps ** 0.5) / (step ** 0.5)
        return decay_factor

def check_grad_flow(model, args, logger=None):
    """检查模型中所有可训练参数的梯度是否存在"""
    grads_missing = []
    params_missing = []
    
    # 遍历所有命名参数（通过 model.module 适配 DistributedDataParallel）
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        for name, param in model.module.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    grads_missing.append(name)
                elif torch.all(param.grad == 0):
                    params_missing.append(name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    grads_missing.append(name)
                elif torch.all(param.grad == 0):
                    params_missing.append(name)
    # 记录结果
    log_msg = []
    if len(grads_missing) > 0:
        log_msg.append(f"梯度未回传的参数: {grads_missing}")
    if len(params_missing) > 0:
        log_msg.append(f"梯度全零的参数: {params_missing}")
    
    # 输出到日志（仅在主进程）
    if len(log_msg) > 0 and logger and args.local_rank == 0:
        logger.info("梯度检查结果:\n" + "\n".join(log_msg))
    
    return len(grads_missing) + len(params_missing) == 0

def get_fbank_from_wav(wav_path, vad_splitter):
        wav, _ = librosa.load(wav_path, sr=16000)

        if vad_splitter is not None:
            # 执行 VAD 检测
            time_segments = vad_splitter.find_speech_segments(wav)

            if len(time_segments) > 1:
                # 提取第一个语音段的前后边界（可根据需求调整策略）
                start_time, end_time = time_segments[0][0], time_segments[-1][1]
                expand_duration = 0.1  # 扩展 100ms
                start_time = max(0, start_time - expand_duration)
                end_time = min(len(wav), end_time + expand_duration)
                # 转换为采样点索引
                start_idx = int(start_time * 16000)
                end_idx = int(end_time * 16000)
                wav = wav[start_idx:end_idx]
        fbank = logmelfilterbank(
            wav, sampling_rate=16000
        )
        fbank = torch.from_numpy(fbank).float()

        return fbank

def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=7600,
    eps=1e-10,
):
    """Compute log-Mel filterbank feature. 
    (https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/bin/preprocess.py)

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

class VADSplitter:
    def __init__(self, aggressiveness=3, sample_rate=16000, frame_duration=30):
        # 参数校验
        assert sample_rate in [8000, 16000, 32000, 48000], f"Invalid sample rate: {sample_rate}"
        assert frame_duration in [10, 20, 30], f"Invalid frame duration: {frame_duration}ms"
        assert 0 <= aggressiveness <= 3, "Aggressiveness must be 0-3"

        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)

    def _convert_audio(self, audio: np.ndarray) -> np.ndarray:
        """将音频转换为符合WebRTC VAD要求的格式"""
        # 转换为16-bit PCM
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # 补齐长度
        pad_len = (self.frame_size - (len(audio) % self.frame_size)) % self.frame_size
        return np.pad(audio, (0, pad_len), mode='constant')

    def find_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """执行鲁棒的VAD检测"""
        try:
            # 格式转换
            audio = self._convert_audio(audio)
            
            # 分帧处理
            frames = [
                audio[i:i+self.frame_size] 
                for i in range(0, len(audio), self.frame_size)
                if len(audio[i:i+self.frame_size]) == self.frame_size
            ]
            
            # 语音活动检测
            speech_flags = []
            for frame in frames:
                try:
                    speech_flags.append(
                        self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    )
                except:
                    speech_flags.append(False)
            
            # 合并连续语音段
            segments = []
            start_idx = None
            for i, is_speech in enumerate(speech_flags):
                if is_speech and start_idx is None:
                    start_idx = i
                elif not is_speech and start_idx is not None:
                    segments.append((start_idx, i-1))
                    start_idx = None
            if start_idx is not None:
                segments.append((start_idx, len(frames)-1))
            
            # 转换为时间戳
            return [
                (s * self.frame_duration / 1000, e * self.frame_duration / 1000)
                for s, e in segments
            ]
        except Exception as e:
            print(f"VAD处理失败: {str(e)}")
            return []