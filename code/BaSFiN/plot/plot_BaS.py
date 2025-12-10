import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def parse_log_file(log_file_path, kl_weights):
    """
    這個函式會：
    1. 第一遍掃描所有 trial（0、1、2），找出每個 (kl_weight, trial) 的 best_epoch。
    2. 第二遍掃描，針對每個 (kl_weight, trial) 的 best_epoch，蒐集 Train/Valid/Test AUC。
    3. 最後對三個 trial 的 AUC 做平均，回傳一個 dict：
       { kl_weight: {'train_auc': avg_train, 'valid_auc': avg_valid, 'test_auc': avg_test} }
    """
    # 第一階段：先存放每個 (kl_weight, trial) 的 best_epoch
    # 格式：{ (kl_weight, trial): best_epoch }
    kl_trial_best_epoch = {}
    
    # 第二階段：存放每個 (kl_weight, trial) 在 best_epoch 時的 AUC
    # 格式：{ (kl_weight, trial): {'train': float, 'valid': float, 'test': float} }
    kl_trial_aucs = {}
    
    # 正規表達式：不再固定 Trial 0，而是捕獲 Trial (\d+)
    early_stop_pattern = r"KL weight ([\d.]+), Trial (\d+), Early stopping triggered after \d+ epochs, best epoch was (\d+)"
    train_auc_pattern = r"KL weight ([\d.]+), Trial (\d+), Epoch \[(\d+)/200\], Train AUC: ([\d.]+)"
    valid_auc_pattern = r"KL weight ([\d.]+), Trial (\d+), Epoch \[(\d+)/200\], Valid AUC: ([\d.]+)"
    test_auc_pattern = r"KL weight ([\d.]+), Trial (\d+), Epoch \[(\d+)/200\], Test AUC: ([\d.]+)"
    
    # 先讀取整個 log 檔
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Log file {log_file_path} not found.")
        return None
    
    # 第一遍：掃描所有 Early stopping 結果，紀錄 (kl, trial) 的 best_epoch
    for line in lines:
        match = re.search(early_stop_pattern, line)
        if match:
            kl_weight = float(match.group(1))
            trial = int(match.group(2))
            best_epoch = int(match.group(3))
            # 對照最近的預定義 kl_weights，確保小數點一致
            closest_kl = min(kl_weights, key=lambda x: abs(x - kl_weight))
            if abs(kl_weight - closest_kl) < 1e-6:
                kl_trial_best_epoch[(closest_kl, trial)] = best_epoch
                # 先初始化 AUC 容器
                kl_trial_aucs.setdefault((closest_kl, trial), {'train': None, 'valid': None, 'test': None})
    
    # 第二遍：針對每行 log，把符合條件的 AUC 存入對應的 (kl, trial)
    for line in lines:
        # Train AUC
        m = re.search(train_auc_pattern, line)
        if m:
            kl_weight = float(m.group(1))
            trial = int(m.group(2))
            epoch = int(m.group(3))
            auc = float(m.group(4))
            closest_kl = min(kl_weights, key=lambda x: abs(x - kl_weight))
            key = (closest_kl, trial)
            if key in kl_trial_best_epoch and epoch == kl_trial_best_epoch[key]:
                kl_trial_aucs[key]['train'] = auc
        
        # Valid AUC
        m = re.search(valid_auc_pattern, line)
        if m:
            kl_weight = float(m.group(1))
            trial = int(m.group(2))
            epoch = int(m.group(3))
            auc = float(m.group(4))
            closest_kl = min(kl_weights, key=lambda x: abs(x - kl_weight))
            key = (closest_kl, trial)
            if key in kl_trial_best_epoch and epoch == kl_trial_best_epoch[key]:
                kl_trial_aucs[key]['valid'] = auc
        
        # Test AUC
        m = re.search(test_auc_pattern, line)
        if m:
            kl_weight = float(m.group(1))
            trial = int(m.group(2))
            epoch = int(m.group(3))
            auc = float(m.group(4))
            closest_kl = min(kl_weights, key=lambda x: abs(x - kl_weight))
            key = (closest_kl, trial)
            if key in kl_trial_best_epoch and epoch == kl_trial_best_epoch[key]:
                kl_trial_aucs[key]['test'] = auc
    
    # 第三步：對每個 kl_weight，遍歷三個 trial，計算平均 AUC
    averaged_results = {}
    for kl in kl_weights:
        # 想要針對 trial 0、1、2 做平均
        train_list, valid_list, test_list = [], [], []
        for trial in [0, 1, 2]:
            key = (kl, trial)
            if key not in kl_trial_aucs:
                # 如果某個 trial 根本沒有跑到 Early stopping，就跳過這個 kl
                train_list = []
                break
            aucs = kl_trial_aucs[key]
            # 確保三項都有值
            if aucs['train'] is None or aucs['valid'] is None or aucs['test'] is None:
                train_list = []
                break
            train_list.append(aucs['train'])
            valid_list.append(aucs['valid'])
            test_list.append(aucs['test'])
        
        # 如果剛剛有成功蒐集到三個 trial 的數據，計算平均後存入 averaged_results
        if len(train_list) == 3:
            avg_train = sum(train_list) / 3.0
            avg_valid = sum(valid_list) / 3.0
            avg_test = sum(test_list) / 3.0
            averaged_results[kl] = {
                'train_auc': avg_train,
                'valid_auc': avg_valid,
                'test_auc': avg_test
            }
        else:
            # 某些 trial 缺資料的話就跳過
            print(f"Warning: Incomplete data for KL weight {kl}, skipping.")
    
    return averaged_results

def plot_kl_auc(results, best_kl_weight, kl_weights, output_file='../plot/  kl_weight_auc_plot.png'):
    """
    以 log-scale 繪出平均後的 Train/Valid/Test AUC 曲線，並在 best_kl_weight 上畫虛線。
    """
    kl_weights = sorted(kl_weights)
    # 只挑有平均結果的 kl
    available = [kl for kl in kl_weights if kl in results]
    
    train_auc = [results[kl]['train_auc'] for kl in available]
    valid_auc = [results[kl]['valid_auc'] for kl in available]
    test_auc = [results[kl]['test_auc'] for kl in available]
    
    plt.figure(figsize=(10, 6))
    plt.plot(available, train_auc, label='Avg Train AUC', marker='o', linestyle='-')
    plt.plot(available, valid_auc, label='Avg Valid AUC', marker='s', linestyle='-')
    plt.plot(available, test_auc, label='Avg Test AUC', marker='^', linestyle='-')
    
    # 標示最佳 KL weight 的虛線
    plt.axvline(x=best_kl_weight, color='black', linestyle='--',
                label=f'Best KL Weight ({best_kl_weight:.6f})')
    
    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel('KL Weight (log scale)')
    plt.ylabel('AUC (averaged over 3 trials)')
    plt.title('Averaged AUC vs KL Weight for Train, Valid, and Test')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved as {output_file}")

def main():
    log_file_path = 'logs/final_logs/Search/BBB_BT_100_search.log'
    
    # 平均後最佳的 KL weight（你說是 0.017433）
    best_kl_weight = 0.017433

    kl_weights = [
        0.001, 0.001269, 0.00161, 0.002043, 0.002593, 0.00329, 0.004175, 0.005298, 
        0.006723, 0.008532, 0.010826, 0.013738, 0.017433, 0.022122, 0.028072, 0.035622, 
        0.045204, 0.057362, 0.07279, 0.092367, 0.11721, 0.148735, 0.188739, 0.239503,
        0.30392, 0.385662, 0.48939, 0.621017, 0.788046, 1.0
    ]
    
    # 解析並取平均
    results = parse_log_file(log_file_path, kl_weights)
    if not results:
        print("Failed to parse log file. Exiting.")
        return
    
    plot_kl_auc(results, best_kl_weight, kl_weights)

if __name__ == '__main__':
    main()
