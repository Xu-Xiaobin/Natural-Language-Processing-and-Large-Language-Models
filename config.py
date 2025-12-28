import torch

class Config:
    def __init__(self):
        # 路径设置
        # self.train_path = '/mnt/afs/250010120/course/nlp/hw3/data/train_mixed_v2.jsonl'
        self.train_path = '/mnt/afs/250010120/course/nlp/hw3/data/train_100k_retranslated_hunyuan.jsonl'
        self.val_path = '/mnt/afs/250010120/course/nlp/hw3/data/valid_retranslated_hunyuan.jsonl'
        self.test_path = '/mnt/afs/250010120/course/nlp/hw3/data/test_retranslated_hunyuan.jsonl'
        self.model_save_path = './checkpoints/'
        
        # 数据处理
        self.src_lang = 'zh'
        self.tgt_lang = 'en'
        self.max_len = 50
        self.min_freq = 2
        
        # 模型通用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        self.num_epochs = 10
        self.learning_rate = 0.0005
        
        # RNN 特定参数 [cite: 63]
        self.rnn_hidden_dim = 512
        self.rnn_layers = 2
        self.rnn_dropout = 0.3
        self.bidirectional = False # 题目要求单向
        self.attn_method = 'dot' # 'dot', 'general', 'concat'
        
        # Transformer 特定参数 [cite: 69]
        self.d_model = 512
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.trans_dropout = 0.1
        self.norm_type = 'layernorm' # 'layernorm' or 'rmsnorm' 
        self.pos_enc_type = 'absolute' # 'absolute' or 'relative'