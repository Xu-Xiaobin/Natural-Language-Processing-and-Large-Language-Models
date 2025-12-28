import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import Config
from data_utils import NMTDataset, collate_fn
from model_rnn import EncoderRNN, DecoderRNN, Attention, Seq2SeqRNN
from model_transformer import TransformerNMT

def train_epoch(model, dataloader, optimizer, criterion, clip, config, model_type='rnn'):
    model.train()
    epoch_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(config.device), tgt.to(config.device)
        
        optimizer.zero_grad()
        
        if model_type == 'rnn':
            # RNN forward, tgt used for teacher forcing
            output = model(src, tgt, teacher_forcing_ratio=0.5)
            # output: (batch, len, vocab), tgt: (batch, len)
            output_dim = output.shape[-1]
            # Ignore <sos> in target for loss calc
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
        else:
            # Transformer forward
            tgt_input = tgt[:, :-1] # Input to decoder
            tgt_out = tgt[:, 1:]    # Expected output
            pad_idx = dataloader.dataset.src_vocab['<pad>']
            output = model(src, tgt_input, pad_idx)
            output = output.reshape(-1, output.shape[-1])
            tgt = tgt_out.reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

# def main():
#     cfg = Config()
#     os.makedirs(cfg.model_save_path, exist_ok=True)
    
#     # 1. Load Data
#     print("Loading data...")
#     # Assume data files exist. If not, create dummy files or error out.
#     if not os.path.exists(cfg.train_path):
#         print(f"Error: Data file {cfg.train_path} not found.")
#         return

#     train_dataset = NMTDataset(cfg.train_path, build_vocab=True, min_freq=cfg.min_freq)
#     src_vocab = train_dataset.src_vocab
#     tgt_vocab = train_dataset.tgt_vocab
    
#     # Save vocabs for inference
#     torch.save({'src': src_vocab, 'tgt': tgt_vocab}, os.path.join(cfg.model_save_path, 'vocabs.pt'))
    
#     pad_idx = src_vocab['<pad>']
#     train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
#                               shuffle=True, collate_fn=lambda b: collate_fn(b, pad_idx))

#     # 2. Initialize Model
#     MODEL_TYPE = 'transformer' # Change to 'rnn' to train RNN
    
#     print(f"Initializing {MODEL_TYPE} model...")
#     if MODEL_TYPE == 'rnn':
#         attn = Attention(cfg.rnn_hidden_dim, method=cfg.attn_method)
#         enc = EncoderRNN(len(src_vocab), cfg.rnn_hidden_dim, cfg.rnn_layers, cfg.rnn_dropout)
#         dec = DecoderRNN(len(tgt_vocab), cfg.rnn_hidden_dim, cfg.rnn_layers, cfg.rnn_dropout, attn)
#         model = Seq2SeqRNN(enc, dec, cfg.device).to(cfg.device)
#     else:
#         model = TransformerNMT(len(src_vocab), len(tgt_vocab), cfg.d_model, cfg.nhead,
#                                cfg.num_encoder_layers, cfg.num_decoder_layers, 
#                                cfg.dim_feedforward, cfg.trans_dropout, cfg.device).to(cfg.device)
        
#         # Initialize parameters
#         for p in model.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
#     criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
#     # 3. Training Loop
#     best_loss = float('inf')
    
#     for epoch in range(cfg.num_epochs):
#         train_loss = train_epoch(model, train_loader, optimizer, criterion, 1.0, cfg, MODEL_TYPE)
#         print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f}')
        
#         if train_loss < best_loss:
#             best_loss = train_loss
#             torch.save(model.state_dict(), os.path.join(cfg.model_save_path, f'{MODEL_TYPE}_best.pt'))

# if __name__ == "__main__":
#     main()
    