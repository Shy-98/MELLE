import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from MELLE import MELLE
from DDP_dataset import DynamicBatchingDataset
from functools import partial
from modules.modules import FEATURE_DIM
from modules.traing_utils import set_seed, setup_logger, lr_lambda, check_grad_flow, save_args
def main():
    # 参数配置
    parser = argparse.ArgumentParser(description="TTS Model Training")
    parser.add_argument("--train_json", type=str, default='data/librispeech_train960.jsonl')
    parser.add_argument("--batch_frames", type=int, default=2000)
    parser.add_argument("--data_buffer_size", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--max_update_step", type=int, default=400000)
    parser.add_argument("--save_dir", type=str, default="debug_exp")
    parser.add_argument("--resume", type=str, help="恢复训练的检查点路径")
    parser.add_argument("--save_interval", type=int, default=10000, help="保存间隔步数")
    parser.add_argument("--log_interval", type=int, default=50, help="日志记录间隔步数")
    parser.add_argument("--warmup_steps", type=int, default=10000, help="Warmup steps")
    parser.add_argument("--warmup_init_lr", type=float, default=1e-7)

    parser.add_argument("--feature_name", type=str, default="fbank", choices=['fbank'])
    # Model Setting
    parser.add_argument("--using_rope", action='store_true')
    parser.add_argument("--using_postnet", action='store_true')
    parser.add_argument("--using_qwen2mlp", action='store_true')
    parser.add_argument("--norm", type=str, default='rms', choices=['rms', 'layer'])
    parser.add_argument("--transformer_activation", type=str, default='relu', choices=['silu','relu','tanh'])
    parser.add_argument("--prenet_activation", type=str, default='relu', choices=['silu','relu','tanh'])
    parser.add_argument("--postnet_activation", type=str, default='tanh', choices=['silu','relu','tanh'])
    parser.add_argument("--exp_name", type=str, default='melle')
    parser.add_argument("--using_vad", action='store_true')
    #CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=12345 DDP_main.py --train_json data/librispeech_train960.jsonl --batch_frames 20000 --save_dir librispeech_exp --using_postnet --norm layer --transformer_activation relu --prenet_activation relu --postnet_activation tanh --exp_name  melle_vad_flux_0.1_bce5
    #CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=12345 DDP_main.py --train_json data/wenet4tts_emov2_zh.jsonl --batch_frames 50000 --save_dir wenet4tts_emov2_exp/

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.exp_name)
    args.log_dir = os.path.join(args.save_dir, 'logs')
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    set_seed(args.local_rank + 3704)
    print(f"Rank {args.local_rank} using GPU {torch.cuda.current_device()}")


    # 初始化系统
    logger = setup_logger(args.log_dir, args.local_rank)
    os.makedirs(args.save_dir, exist_ok=True) if args.local_rank == 0 else None
    logger.info(args) if args.local_rank == 0 else None
    save_args(args, os.path.join(args.save_dir, 'args.json')) if args.local_rank == 0 else None

    # 初始化模型
    model = MELLE(
        feature_dim=FEATURE_DIM[args.feature_name],
        using_rope=args.using_rope,
        using_postnet=args.using_postnet,
        using_qwen2mlp=args.using_qwen2mlp,
        norm=args.norm,
        transformer_activation=args.transformer_activation,
        prenet_activation=args.prenet_activation,
        postnet_activation=args.postnet_activation,
    ).to(device)
    logger.info(model) if args.local_rank == 0 else None
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[int(args.local_rank)], 
        output_device=int(args.local_rank), 
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=[0.9,0.98], weight_decay=0.01)
    
    # 初始化学习率调度器
    scheduler = LambdaLR(optimizer, partial(lr_lambda, args=args))

    # 训练状态恢复
    start_step = 0
    if args.resume:
        # 所有进程需同步加载检查点
        checkpoint = torch.load(args.resume, map_location=f'cuda:{args.local_rank}')
        
        # 动态修复键名：移除 "module." 前缀（如果检查点来自非 DDP 训练）
        state_dict = checkpoint['model']
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # 若当前是非 DDP 模式，但检查点来自 DDP 模式，需修复键名
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        else:
            # 若当前是 DDP 模式，但检查点来自非 DDP 模式，需添加 "module." 前缀
            state_dict = {('module.' + k): v for k, v in state_dict.items()}
        
        # 加载修复后的 state_dict
        model.load_state_dict(state_dict)
        
        start_step = checkpoint['step'] + 1
        
        # 仅 rank 0 打印日志
        if args.local_rank == 0:
            logger.info(f"从步骤 {start_step} 恢复训练")

    # 数据加载器
    train_dataset = DynamicBatchingDataset(
        args.train_json,
        max_frames=args.batch_frames,
        shuffle_buffer=True,
        buffer_size=args.data_buffer_size,
        rank=args.local_rank,
        world_size=torch.distributed.get_world_size(),
        seed=42 + args.local_rank,
        feature_name=args.feature_name,
        shuffer_random_key=args.exp_name,
        using_vad=args.using_vad,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
        
    )

    # 启用自动混合精度训练
    scaler = torch.amp.GradScaler('cuda', growth_interval=1000, init_scale=1.0)

    # 训练循环
    model.train()
    total_steps = start_step+1
    start_time = time.time()
    def item_to_device(batch, device):
        # 数据转移到设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            if isinstance(v, dict):
                batch[k] = item_to_device(batch[k], device)
        return batch

    epoch = args.start_epoch-1
    while True:
        logger.info(f"开始训练轮次 {epoch + 1}/{args.epochs} 开始训练步 {total_steps}/{args.max_update_step}")
        train_loader.dataset.shuffer_data()
        break_flag = False
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                logger.info(f'{batch_idx} sample is None, break {epoch+1} epoch')
                break_flag = True
            # 同步break_flag
            break_flag_tensor = torch.tensor(int(break_flag), device=device)
            torch.distributed.all_reduce(break_flag_tensor, op=torch.distributed.ReduceOp.MAX)
            break_flag = bool(break_flag_tensor.item())
            
            torch.distributed.barrier()
            if break_flag: break
            # 数据转移到设备
            batch = item_to_device(batch, device)

            # 计算总帧数用于归一化
            total_frames = batch['mel_lengths'].sum().float()
            
            # 前向计算（使用自动混合精度）
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):  # 自动混合精度
                total_loss, loss_l1, loss_l2, loss_logvar, loss_bce = model(
                    **batch
                )
            # 反向传播（使用梯度缩放）
            scaler.scale(total_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)  # 使用缩放后的优化器步骤
            scale_ = scaler.get_scale()
            scaler.update()  # 更新梯度缩放器
            # 检测梯度溢出并调整损失缩放
            new_scale = scaler.get_scale()
            if new_scale < scale_:
                logger.info(f"gradient overflow detected, ignoring gradient, setting loss scale to: {new_scale}")
            scheduler.step()
            total_steps += 1
            
            # 同步所有指标 ---------------------------------------------------
            # 将各指标转换为Tensor
            raw_total_loss_tensor = total_loss.detach()
            total_frames_tensor = total_frames.detach().clone()
            samples_tensor = torch.tensor(batch['mel'].shape[0], device=device)
            total_all_frames_tensor = torch.tensor(batch['mel'].shape[0]*(batch['mel'].shape[1]+batch['txt'].shape[1]), device=device)

            # 跨GPU求和
            torch.distributed.all_reduce(raw_total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_frames_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(samples_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_all_frames_tensor, op=torch.distributed.ReduceOp.SUM)

            # 计算全局平均值
            global_total_loss = raw_total_loss_tensor.item() / total_frames_tensor.item()
            global_total_frames = total_frames_tensor.item()
            global_total_all_frames = total_all_frames_tensor.item()
            global_samples = samples_tensor.item()

            # 同步其他损失项（以loss_l1为例）
            loss_l1_tensor = loss_l1.detach()
            torch.distributed.all_reduce(loss_l1_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss_l1 = loss_l1_tensor.item() / total_frames_tensor.item()

            loss_l2_tensor = loss_l2.detach()
            torch.distributed.all_reduce(loss_l2_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss_l2 = loss_l2_tensor.item() / total_frames_tensor.item()

            loss_logvar_tensor = loss_logvar.detach()
            torch.distributed.all_reduce(loss_logvar_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss_logvar = loss_logvar_tensor.item() / total_frames_tensor.item()

            loss_bce_tensor = loss_bce.detach()
            torch.distributed.all_reduce(loss_bce_tensor, op=torch.distributed.ReduceOp.SUM)
            global_loss_bce = loss_bce_tensor.item() / total_frames_tensor.item()

            # 日志记录
            # 定期检查梯度（例如每 100 步检查一次）
            if (total_steps % args.log_interval == 0 or total_steps > args.max_update_step) and args.local_rank == 0:
                time_per_step = (time.time() - start_time) / args.log_interval
                log_msg = (
                    f"Epoch/Step {epoch+1}/{total_steps} | "
                    f"Loss: {global_total_loss:.4f} | "
                    f"L1: {global_loss_l1:.4f} | "
                    f"L2: {global_loss_l2:.4f} | "
                    f"LogVar: {global_loss_logvar:.4f} | "
                    f"BCE: {global_loss_bce:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"ValidFrames/AllFrames/Samples: {int(global_total_frames)}/{int(global_total_all_frames)}/{int(global_samples)} | "
                    f"Time/step: {time_per_step:.3f}s"
                )
                logger.info(log_msg)
                grad_ok = check_grad_flow(model, args, logger)
                if not grad_ok:
                    logger.warning("检测到部分参数无梯度！")
                
                start_time = time.time()

            # 模型保存
            if (total_steps % args.save_interval == 0 or total_steps > args.max_update_step) and args.local_rank == 0:
                save_path = os.path.join(args.save_dir, f"step_{total_steps}.pt")
                torch.save({
                    'step': total_steps,
                    'model': model.module.state_dict(),
                }, save_path)
                logger.info(f"已保存检查点到 {save_path}")

            if total_steps > args.max_update_step:
                break

        # 每轮次结束保存
        # if args.local_rank == 0:
        #     epoch_save_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
        #     torch.save({
        #         'step': total_steps,
        #         'model': model.module.state_dict(),
        #     }, epoch_save_path)
        if total_steps > args.max_update_step:
            break
        epoch += 1
        if epoch > args.epochs and args.max_update_step <= 0:
            break
    logger.info("训练完成")

if __name__ == "__main__":
    main()