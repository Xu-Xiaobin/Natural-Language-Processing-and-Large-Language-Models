import torch
import json
import jieba
import nltk
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 首次运行需要下载 punkt
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class Vocab:
    def __init__(self, tokens=None, min_freq=1, specials=['<pad>', '<sos>', '<eos>', '<unk>']):
        self.freqs = Counter()
        self.stoi = {}
        self.itos = []
        self.specials = specials
        self.min_freq = min_freq
        
        if tokens:
            self.build_vocab(tokens)

    def build_vocab(self, tokens):
        self.freqs.update(tokens)
        idx = 0
        for s in self.specials:
            self.stoi[s] = idx
            self.itos.append(s)
            idx += 1
            
        for token, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.stoi[token] = idx
                self.itos.append(token)
                idx += 1
                
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
    
    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices):
        return [self.itos[idx] for idx in indices]

class NMTDataset(Dataset):
    def __init__(self, filepath, src_vocab=None, tgt_vocab=None, build_vocab=False, max_len=50, min_freq=2):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        self.src_data = [] # List[List[str]]
        self.tgt_data = [] 
        
        print(f"Tokenizing {filepath}...")
        for item in self.data:
            # 中文分词 [cite: 101]
            src_tokens = list(jieba.cut(item['zh_hy']))
            # 英文分词 [cite: 100]
            tgt_tokens = nltk.word_tokenize(item['en'].lower())
            
            if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                self.src_data.append(src_tokens)
                self.tgt_data.append(tgt_tokens)
        
        if build_vocab:
            all_src = [t for sent in self.src_data for t in sent]
            all_tgt = [t for sent in self.tgt_data for t in sent]
            self.src_vocab = Vocab(all_src, min_freq=min_freq)
            self.tgt_vocab = Vocab(all_tgt, min_freq=min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_encoded = [self.src_vocab['<sos>']] + self.src_vocab.lookup_indices(self.src_data[idx]) + [self.src_vocab['<eos>']]
        tgt_encoded = [self.tgt_vocab['<sos>']] + self.tgt_vocab.lookup_indices(self.tgt_data[idx]) + [self.tgt_vocab['<eos>']]
        return torch.tensor(src_encoded), torch.tensor(tgt_encoded)

def collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
    return src_batch, tgt_batch