import os
import torch
import json
import sacrebleu
import glob
import argparse
from tqdm import tqdm

# [关键修改]: 直接从 inference.py 导入封装好的函数
from inference import load_model, translate

def evaluate_experiment(exp_path, test_data, device, beam_size=1):
    # 1. 使用 inference.py 的 load_model 自动处理参数和权重加载
    try:
        model, src_vocab, tgt_vocab, params = load_model(exp_path, device)
    except Exception as e:
        print(f"Skipping {os.path.basename(exp_path)}: {e}")
        return None

    refs = []
    preds = []
    
    exp_name = os.path.basename(exp_path)
    print(f"Evaluating {exp_name} (Beam={beam_size})...")
    
    # 2. 批量推理
    # 使用 tqdm 显示进度条
    for item in tqdm(test_data, desc=f"Translating {exp_name[:15]}...", leave=False):
        src_text = item['zh_hy']
        tgt_text = item['en']
        
        # [关键修改]: 调用 inference.py 的 translate 接口
        pred = translate(src_text, model, src_vocab, tgt_vocab, params, device, beam_size=beam_size)
        
        preds.append(pred)
        refs.append(tgt_text)
        
    # 3. 计算 BLEU
    # sacrebleu 期望 references 是一个 list of lists [[ref1_a, ref2_a], [ref1_b, ref2_b]]
    # 这里我们每个句子只有一个参考译文
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', type=int, default=1, help='Beam size for evaluation (default: 1 for greedy)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = './checkpoints_rnn'
    
    # 1. 加载测试集
    test_path = '/mnt/afs/250010120/course/nlp/hw3/data/test_retranslated_hunyuan.jsonl'
    if not os.path.exists(test_path):
        print(f"Error: {test_path} not found.")
        return

    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test samples.")
    
    # 2. 遍历所有实验文件夹
    results = {}
    exp_dirs = glob.glob(os.path.join(base_dir, '*'))
    
    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir): continue
        
        # 跳过没有 best_model.pt 的文件夹
        if not os.path.exists(os.path.join(exp_dir, 'best_model.pt')):
            continue
            
        score = evaluate_experiment(exp_dir, test_data, device, beam_size=args.beam)
        
        if score is not None:
            exp_name = os.path.basename(exp_dir)
            results[exp_name] = score
            print(f"Exp: {exp_name} | BLEU: {score:.2f}")

    # 3. 打印最终排行榜
    print("\n" + "="*50)
    print(f"Final Leaderboard (BLEU Score, Beam={args.beam})")
    print("="*50)
    
    # 按分数降序排列
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"{rank}. {score:.2f}\t{name}")
        
    # 4. 保存结果到文件 (方便写报告复制)
    with open('bleu_scores.txt', 'w') as f:
        f.write(f"Leaderboard (Beam={args.beam})\n")
        for name, score in sorted_results:
            f.write(f"{name}\t{score:.2f}\n")
    print(f"\nResults saved to bleu_scores.txt")

if __name__ == "__main__":
    main()