import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import json

from config import Config
from data_utils import NMTDataset, collate_fn
from model_rnn import EncoderRNN, DecoderRNN, Attention, Seq2SeqRNN
from model_transformer import TransformerNMT
from train import train_epoch 

# ==================================================================================
# 1. Define parameter grid
# ==================================================================================

# --- A. RNN 实验网格 ---
# 对应作业要求：对比 Attention (dot, general, concat)
# rnn_param_grid = {
#     'model_type': ['rnn'],
#     'learning_rate': [0.001, 0.0001],
#     'batch_size': [64, 128, 256], # Total Batch Size
#     'attn_method': ['dot', 'general', 'concat'],  # 重点对比项
#     'num_epochs': [100]
# }
rnn_param_grid = {
    'model_type': ['rnn'],
    'learning_rate': [0.001],
    'batch_size': [512], # Total Batch Size 4gpu
    'attn_method': ['concat'],  # 重点对比项
    'num_epochs': [50]
}

# --- B. Transformer 实验网格 ---
# 对应作业要求：Batch Size [32, 64, 128], LR, Norm Type, PosEnc [cite: 70, 71]
# transformer_param_grid = {
#     'model_type': ['transformer'],
    
#     # 1. 训练超参
#     'batch_size': [64, 128, 256], # Total Batch Size
#     'learning_rate': [0.0005, 0.001],
    
#     # 2. 架构消融 (Ablation)
#     'norm_type': ['layernorm', 'rmsnorm'],
#     'pos_enc_type': ['absolute', 'relative'],   # 如果你要跑 relative，加到列表里
    
#     # 3. 模型规模 (d_model, nhead) 
#     # 注意: d_model 必须能被 nhead 整除。这里为了安全，我们用元组列表表示一组规模
#     'model_scale': [
#         (256, 4),  # Small
#         (512, 8)   # Base
#     ],
    
#     'num_epochs': [100]
# }
transformer_param_grid = {
    'model_type': ['layernorm'],
    
    # 1. 训练超参
    'batch_size': [128], # Total Batch Size
    'learning_rate': [0.0001],
    
    # 2. 架构消融 (Ablation)
    'norm_type': ['layernorm'],
    'pos_enc_type': ['absolute'],   # 如果你要跑 relative，加到列表里
    
    # 3. 模型规模 (d_model, nhead) 
    # 注意: d_model 必须能被 nhead 整除。这里为了安全，我们用元组列表表示一组规模
    'model_scale': [
        (512, 8)   # Base
    ],
    
    'num_epochs': [50]
}

# ==================================================================================
# 2. 工具函数
# ==================================================================================

def generate_experiments(grid_dict):
    """
    将字典列表转换为所有可能的组合 (笛卡尔积)
    输入: {'bs': [32, 64], 'lr': [0.001]}
    输出: [{'bs': 32, 'lr': 0.001}, {'bs': 64, 'lr': 0.001}]
    """
    keys = grid_dict.keys()
    values = grid_dict.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

def get_experiment_name(params):
    """根据参数生成唯一的文件夹名"""
    if params['model_type'] == 'rnn':
        return f"rnn_{params['attn_method']}_bs{params['batch_size']}_lr{params['learning_rate']}"
    else:
        # Transformer 命名
        scale = params.get('model_scale', (512, 8))
        d_model = scale[0]
        return f"trans_{params['norm_type']}_bs{params['batch_size']}_lr{params['learning_rate']}_d{d_model}"

# ==================================================================================
# 3. 单次实验运行逻辑 (修改版：支持多卡)
# ==================================================================================

def run_single_experiment(params):
    cfg = Config()
    
    # --- 1. 动态更新 Config ---
    print(f"\n{'='*10} Setting up: {params} {'='*10}")
    for k, v in params.items():
        if k == 'model_scale':
            cfg.d_model, cfg.nhead = v 
            cfg.dim_feedforward = cfg.d_model * 4
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
        
    # --- 2. 准备保存路径 ---
    exp_name = get_experiment_name(params)
    save_path = os.path.join('checkpoints', exp_name)
    os.makedirs(save_path, exist_ok=True)
    
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json_ready_params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}
        json.dump(json_ready_params, f, indent=4)

    # --- 3. 加载数据 ---
    print(f"Loading data with Total Batch Size: {cfg.batch_size}...")
    train_dataset = NMTDataset(cfg.train_path, build_vocab=True, min_freq=cfg.min_freq)
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab
    pad_idx = src_vocab['<pad>']
    
    torch.save({'src': src_vocab, 'tgt': tgt_vocab}, os.path.join(save_path, 'vocabs.pt'))
    
    # 设置 num_workers 建议为: GPU数量 * 4
    num_workers = torch.cuda.device_count() * 4 if torch.cuda.is_available() else 2
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                              shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx),
                              num_workers=num_workers)

    # --- 4. 初始化模型 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.device = device # 确保 Config 里也是正确的 device
    
    print(f"Initializing {params['model_type']}...")
    
    if params['model_type'] == 'rnn':
        attn = Attention(cfg.rnn_hidden_dim, method=cfg.attn_method)
        enc = EncoderRNN(len(src_vocab), cfg.rnn_hidden_dim, cfg.rnn_layers, cfg.rnn_dropout)
        dec = DecoderRNN(len(tgt_vocab), cfg.rnn_hidden_dim, cfg.rnn_layers, cfg.rnn_dropout, attn)
        model = Seq2SeqRNN(enc, dec, device)
    else:
        model = TransformerNMT(
            len(src_vocab), len(tgt_vocab), cfg.d_model, cfg.nhead,
            cfg.num_encoder_layers, cfg.num_decoder_layers, 
            cfg.dim_feedforward, cfg.trans_dropout, device,
            norm_type=cfg.norm_type,
            pos_enc_type=cfg.pos_enc_type
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    #多卡并行处理
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model) # 包装模型
    
    model = model.to(device) # 移动到 GPU

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # --- 5. 训练循环 ---
    best_loss = float('inf')
    log_path = os.path.join(save_path, 'train_log.txt')
    
    print(f"Start Training: {exp_name}")
    for epoch in range(cfg.num_epochs):
        try:
            # train_epoch 函数不需要改动，DataParallel 会自动处理 forward 分发
            loss = train_epoch(model, train_loader, optimizer, criterion, 1.0, cfg, params['model_type'])
            
            log_str = f"Epoch {epoch+1}/{cfg.num_epochs} | Loss: {loss:.4f}"
            print(log_str)
            with open(log_path, 'a') as f:
                f.write(log_str + '\n')
            
            # 保存最佳模型
            if loss < best_loss:
                best_loss = loss
                
                # [关键修改]: 保存时去除 DataParallel 的 module 包装
                model_to_save = model.module if hasattr(model, 'module') else model
                
                torch.save(model_to_save.state_dict(), os.path.join(save_path, 'best_model.pt'))
                
        except KeyboardInterrupt:
            print("Interrupted. Saving current state...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break
            
    print(f"Finished: {exp_name} | Best Loss: {best_loss:.4f}\n")


# ==================================================================================
# 4. 主执行入口
# ==================================================================================

if __name__ == "__main__":
    # 1. 生成所有 RNN 组合
    rnn_experiments = list(generate_experiments(rnn_param_grid))
    
    # 2. 生成所有 Transformer 组合
    transformer_experiments = list(generate_experiments(transformer_param_grid))
    
    # 3. 合并列表
    # all_experiments = transformer_experiments
    all_experiments = rnn_experiments
    
    print(f"Total experiments to run: {len(all_experiments)}")

    # test
    # all_experiments = all_experiments[:1]
    
    for i, params in enumerate(all_experiments):
        print(f"[{i+1}/{len(all_experiments)}] Processing...")
        try:
            run_single_experiment(params)
        except Exception as e:
            print(f"Experiment failed: {params}")
            print(e)
            continue