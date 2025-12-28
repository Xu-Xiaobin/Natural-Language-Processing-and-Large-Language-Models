import os
import matplotlib.pyplot as plt
import glob

def parse_log(log_path):
    epochs = []
    losses = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Loss:" in line:
                    # 解析格式: Epoch 1/20 | Loss: 5.4321
                    parts = line.strip().split('|')
                    # 获取 Epoch 数字
                    epoch_part = parts[0].split()[1] # "1/20"
                    epoch = int(epoch_part.split('/')[0])
                    
                    # 获取 Loss 数字
                    loss_part = parts[1].split(':')[1] # " 5.4321"
                    loss = float(loss_part)
                    
                    epochs.append(epoch)
                    losses.append(loss)
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
    return epochs, losses

def plot_single_experiment(log_file):
    """读取单个日志并画图保存"""
    # 1. 获取实验名称 (父文件夹名)
    # log_file: checkpoints/exp_name/train_log.txt
    exp_dir = os.path.dirname(log_file)
    exp_name = os.path.basename(exp_dir)
    
    # 2. 解析数据
    epochs, losses = parse_log(log_file)
    
    if not epochs:
        print(f"Skipping {exp_name}: No data found.")
        return

    # 3. 画图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Train Loss')
    
    # [修改点] 设置动态标题
    plt.title(f"Training Loss Curve: {exp_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # [修改点] 设置动态文件名并保存
    # 图片将保存在项目根目录，文件名包含实验名
    save_filename = f"{exp_name}_loss.png"
    plt.savefig(save_filename)
    plt.close() # 关闭画布，释放内存
    
    print(f"Generated: {save_filename}")

def main():
    # 查找 checkpoints 下所有的 train_log.txt
    log_files = glob.glob('checkpoints_rnn/*/train_log.txt')
    
    if not log_files:
        print("No log files found in checkpoints/")
        return

    print(f"Found {len(log_files)} logs. Generating plots...")
    
    for log_file in log_files:
        plot_single_experiment(log_file)

if __name__ == "__main__":
    main()