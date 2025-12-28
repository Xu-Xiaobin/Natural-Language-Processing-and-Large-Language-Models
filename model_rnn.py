import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        # outputs: (batch, seq_len, hidden)
        # hidden: (n_layers, batch, hidden)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            # Concat 需要将 decoder hidden 和 encoder output 拼接，所以输入是 2 * hidden
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch, hidden) -> Decoder 的上一时刻 hidden state
        # encoder_outputs: (batch, seq_len, hidden) -> Encoder 所有时刻的输出
        
        # 统一处理 hidden 的维度: (batch, 1, hidden)
        hidden = hidden[-1].unsqueeze(1)
        
        # Calculate alignment scores
        attn_energies = None # 初始化变量
        
        if self.method == 'dot':
            # Dot-product attention
            # (batch, 1, hidden) * (batch, hidden, seq_len) -> (batch, 1, seq_len)
            attn_energies = torch.bmm(hidden, encoder_outputs.transpose(1, 2))
            
        elif self.method == 'general':
            # General attention: score(h_t, h_s) = h_t^T * W * h_s
            energy = self.attn(encoder_outputs) # (batch, seq_len, hidden)
            attn_energies = torch.bmm(hidden, energy.transpose(1, 2))
            
        elif self.method == 'concat':
            # [新增修复]: Concat attention logic
            # score(h_t, h_s) = v^T * tanh(W * [h_t; h_s])
            
            src_len = encoder_outputs.shape[1]
            
            # 1. 复制 hidden 以匹配 encoder_outputs 的长度
            # hidden: (batch, 1, hidden) -> (batch, src_len, hidden)
            hidden_expanded = hidden.expand(-1, src_len, -1)
            
            # 2. 拼接
            # (batch, src_len, 2*hidden)
            concat_input = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            
            # 3. 计算 energy 并经过 tanh
            # (batch, src_len, hidden)
            energy = torch.tanh(self.attn(concat_input))
            
            # 4. 投影并调整维度
            # (batch, src_len, 1) -> (batch, 1, src_len)
            attn_energies = self.v(energy).transpose(1, 2)
        
        return F.softmax(attn_energies, dim=-1)

class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers, dropout, attention_model):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.attention = attention_model
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size + hidden_size, hidden_size, n_layers, 
                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: (batch, 1) - single token
        embedded = self.dropout(self.embedding(input))
        
        # Calculate attention weights
        weights = self.attention(hidden, encoder_outputs) # (batch, 1, src_len)
        
        # Context vector
        context = torch.bmm(weights, encoder_outputs) # (batch, 1, hidden)
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden

class Seq2SeqRNN(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to decoder is <sos>
        decoder_input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            # Teacher Forcing [cite: 65]
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs