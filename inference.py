import torch
import torch.nn as nn
import os
import json
import argparse
import math

import warnings
import logging
import os

# 1. 屏蔽 Python 的警告信息 (包括 pkg_resources 和 PyTorch 的 nested tensor 警告)
warnings.filterwarnings("ignore")

import jieba
jieba.setLogLevel(logging.ERROR)  # 屏蔽 "Building prefix dict..." 等日志


from config import Config
from data_utils import Vocab # 确保 data_utils 里有 Vocab 类定义
from model_rnn import EncoderRNN, DecoderRNN, Attention, Seq2SeqRNN
from model_transformer import TransformerNMT

# ==================================================================================
# 1. 模型加载函数
# ==================================================================================
def load_model(exp_dir, device):
    """
    自动从实验目录加载配置、词表和模型权重
    """
    config_path = os.path.join(exp_dir, 'config.json')
    vocab_path = os.path.join(exp_dir, 'vocabs.pt')
    model_path = os.path.join(exp_dir, 'best_model.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found in {exp_dir}")

    # 1. 加载配置
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    # 2. 加载词表
    vocabs = torch.load(vocab_path, map_location='cpu', weights_only = False)
    src_vocab, tgt_vocab = vocabs['src'], vocabs['tgt']

    # 3. 初始化模型结构
    print(f"Loading {params['model_type']} model from {exp_dir}...")
    
    if params['model_type'] == 'rnn':
        attn = Attention(params.get('rnn_hidden_dim', 512), method=params['attn_method'])
        enc = EncoderRNN(len(src_vocab), params.get('rnn_hidden_dim', 512), 
                         params.get('rnn_layers', 2), params.get('rnn_dropout', 0.3))
        dec = DecoderRNN(len(tgt_vocab), params.get('rnn_hidden_dim', 512), 
                         params.get('rnn_layers', 2), params.get('rnn_dropout', 0.3), attn)
        model = Seq2SeqRNN(enc, dec, device)
        
    else: # Transformer
        # 处理 model_scale 可能是 list 的情况 (json加载后 tuple 会变 list)
        if isinstance(params.get('model_scale'), list):
            d_model = params['model_scale'][0]
            nhead = params['model_scale'][1]
        else:
            d_model = params.get('d_model', 512)
            nhead = params.get('nhead', 8)
            
        model = TransformerNMT(
            len(src_vocab), len(tgt_vocab), 
            d_model=d_model, nhead=nhead,
            num_encoder_layers=params.get('num_encoder_layers', 6), 
            num_decoder_layers=params.get('num_decoder_layers', 6), 
            dim_feedforward=d_model * 4, 
            dropout=params.get('trans_dropout', 0.1), 
            device=device,
            norm_type=params.get('norm_type', 'layernorm'),
            pos_enc_type=params.get('pos_enc_type', 'absolute')
        )

    # 4. 加载权重 (处理 DataParallel 的 module. 前缀)
    state_dict = torch.load(model_path, map_location=device, weights_only = False)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, src_vocab, tgt_vocab, params

# ==================================================================================
# 2. 解码策略 (Greedy & Beam Search)
# ==================================================================================

def greedy_decode(model, src, src_vocab, tgt_vocab, max_len, device, model_type):
    """贪婪解码：每一步只选概率最大的词"""
    model.eval()
    # src: (1, seq_len)
    
    # --- RNN 推理 ---
    if model_type == 'rnn':
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src)
        
        # Decoder 输入初始化
        decoder_input = torch.tensor([[tgt_vocab['<sos>']]]).to(device)
        trg_indexes = []
        
        for _ in range(max_len):
            with torch.no_grad():
                output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
            
            # output: (batch, vocab_size)
            pred_token = output.argmax(1).item()
            
            if pred_token == tgt_vocab['<eos>']:
                break
                
            trg_indexes.append(pred_token)
            decoder_input = torch.tensor([[pred_token]]).to(device)
            
        return trg_indexes

# --- Transformer 推理 ---
    else:
        # src_mask 是 padding mask，形状 (Batch, Seq_len)
        src_mask = torch.zeros((1, src.shape[1])).type(torch.bool).to(device)
        
        with torch.no_grad():
            # [修正点]: 显式指定 src_key_padding_mask，mask 设为 None
            memory = model.transformer.encoder(
                model.pos_encoder(model.src_embedding(src) * math.sqrt(model.src_embedding.embedding_dim)), 
                mask=None, 
                src_key_padding_mask=src_mask
            )
        
        ys = torch.ones(1, 1).fill_(tgt_vocab['<sos>']).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            tgt_mask = model.generate_square_subsequent_mask(ys.size(1)).type(torch.bool).to(device)
            
            with torch.no_grad():
                # decoder 的参数位置: tgt, memory, tgt_mask (第3个位置是对的)
                out = model.transformer.decoder(
                    model.pos_encoder(model.tgt_embedding(ys) * math.sqrt(model.tgt_embedding.embedding_dim)), 
                    memory, 
                    tgt_mask
                )
            
            prob = model.fc_out(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            if next_word == tgt_vocab['<eos>']:
                break
                
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            
        # 指定只压缩第0维 (Batch维)，这样即使长度为1，也能得到 shape (1,) 的张量，tolist后是列表
        return ys.squeeze(0).tolist()[1:]

def beam_search_decode(model, src, src_vocab, tgt_vocab, max_len, device, model_type, beam_size=3):
    """Beam Search 解码"""
    if model_type == 'rnn':
        print("Warning: Beam search not implemented for RNN in this script. Using greedy.")
        return greedy_decode(model, src, src_vocab, tgt_vocab, max_len, device, model_type)

    # Transformer Beam Search
    src_mask = torch.zeros((1, src.shape[1])).type(torch.bool).to(device)
    
    with torch.no_grad():
        # [修正点]: 同上，显式指定 src_key_padding_mask
        memory = model.transformer.encoder(
            model.pos_encoder(model.src_embedding(src) * math.sqrt(model.src_embedding.embedding_dim)), 
            mask=None,
            src_key_padding_mask=src_mask
        )
    
    # 队列: (score, sequence_tensor)
    sequences = [[0.0, torch.ones(1, 1).fill_(tgt_vocab['<sos>']).type(torch.long).to(device)]]
    
    for _ in range(max_len):
        all_candidates = []
        for score, seq in sequences:
            # 如果已经结束，直接加入候选
            if seq[0, -1].item() == tgt_vocab['<eos>']:
                all_candidates.append((score, seq))
                continue
                
            tgt_mask = model.generate_square_subsequent_mask(seq.size(1)).type(torch.bool).to(device)
            with torch.no_grad():
                out = model.transformer.decoder(
                    model.pos_encoder(model.tgt_embedding(seq) * math.sqrt(model.tgt_embedding.embedding_dim)), 
                    memory, tgt_mask
                )
            
            # log_softmax 获取对数概率
            prob = torch.log_softmax(model.fc_out(out[:, -1]), dim=1) 
            topk_probs, topk_ids = torch.topk(prob, beam_size)
            
            for i in range(beam_size):
                word_idx = topk_ids[0][i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, word_idx], dim=1)
                new_score = score + topk_probs[0][i].item()
                all_candidates.append((new_score, new_seq))
        
        # 排序并选出 Top K
        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
        sequences = ordered[:beam_size]
        
        # 如果所有 beam 都生成了 <eos>，提前结束
        if all([s[1][0, -1].item() == tgt_vocab['<eos>'] for s in sequences]):
            break
            
    # 返回得分最高的那条
    best_seq = sequences[0][1].squeeze().tolist()
    # 去掉 <sos>，如果包含 <eos> 截断
    if tgt_vocab['<sos>'] in best_seq: best_seq.remove(tgt_vocab['<sos>'])
    if tgt_vocab['<eos>'] in best_seq: best_seq = best_seq[:best_seq.index(tgt_vocab['<eos>'])]
    
    return best_seq

# ==================================================================================
# 3. 主逻辑
# ==================================================================================

def translate(sentence, model, src_vocab, tgt_vocab, params, device, beam_size=1):
    model.eval()
    
    # 1. 中文分词
    tokens = list(jieba.cut(sentence))
    
    # 2. 转换为索引
    indices = [src_vocab['<sos>']] + src_vocab.lookup_indices(tokens) + [src_vocab['<eos>']]
    src_tensor = torch.tensor(indices).unsqueeze(0).to(device) # (1, seq_len)
    
    # 3. 解码
    if beam_size > 1:
        out_indices = beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, 
                                         max_len=50, device=device, model_type=params['model_type'], 
                                         beam_size=beam_size)
    else:
        out_indices = greedy_decode(model, src_tensor, src_vocab, tgt_vocab, 
                                    max_len=50, device=device, model_type=params['model_type'])
        # print(f"DEBUG Indices: {out_indices}")
    
    # 4. 转换回文本
    out_tokens = tgt_vocab.lookup_tokens(out_indices)
    return " ".join([t for t in out_tokens if t not in ['<sos>', '<eos>', '<pad>']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/mnt/afs/250010120/course/nlp/hw3/checkpoints_trans/trans_layernorm_bs128_lr0.0001_d512', help='Path to experiment folder (e.g., checkpoints/rnn_dot_...)')
    parser.add_argument('--sentence', type=str, default="总统在出行时必须使用这些交通工具。", help='Sentence to translate')
    parser.add_argument('--beam', type=int, default=1, help='Beam size (1 for greedy)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model, src_vocab, tgt_vocab, params = load_model(args.checkpoint, device)
    
    # 翻译
    print("-" * 30)
    print(f"Source: {args.sentence}")
    result = translate(args.sentence, model, src_vocab, tgt_vocab, params, device, beam_size=args.beam)
    print(f"Trans : {result}")
    print("-" * 30)