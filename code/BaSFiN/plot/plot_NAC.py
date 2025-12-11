import re
import matplotlib.pyplot as plt

def parse_log(filepath):
    data = {
        "step1": {
            "train_auc": [], "valid_auc": [], "test_auc": [],
            "early_stop": None, "final_val_auc": None
        },
        "step2": {
            "train_auc": [], "valid_auc": [], "test_auc": [],
            "early_stop": None, "final_test_auc": None
        }
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            phase_match = re.search(r'Phase (step\d)', line)
            if not phase_match:
                continue
            phase = phase_match.group(1)

            # 抓各 AUC 值
            if 'Train:' in line:
                auc = float(re.search(r'AUC: ([0-9.]+)', line).group(1))
                data[phase]['train_auc'].append(auc)
            elif 'Valid:' in line:
                auc = float(re.search(r'AUC: ([0-9.]+)', line).group(1))
                data[phase]['valid_auc'].append(auc)
            elif 'Test:' in line and 'Valid:' not in line:
                auc = float(re.search(r'AUC: ([0-9.]+)', line).group(1))
                data[phase]['test_auc'].append(auc)

            # 抓 early stopping
            if 'early stopping triggered' in line:
                early_epoch = int(re.search(r'after (\d+) epochs', line).group(1))
                data[phase]['early_stop'] = early_epoch

            # 最終 AUC
            if 'Step 1, Validation AUC' in line:
                data['step1']['final_val_auc'] = float(re.search(r'AUC: ([0-9.]+)', line).group(1))
            elif 'Step 2, Test AUC' in line:
                data['step2']['final_test_auc'] = float(re.search(r'AUC: ([0-9.]+)', line).group(1))

    return data


def plot_auc_curves(data, phase, output_path=None):
    epochs = range(1, len(data[phase]['train_auc']) + 1)
    
    plt.figure(figsize=(10, 6))

    # 主曲線
    plt.plot(epochs, data[phase]['train_auc'], label='Train AUC', marker='o', linestyle='-', linewidth=2, markersize=5, color='blue')
    if data[phase]['valid_auc']:
        plt.plot(epochs, data[phase]['valid_auc'], label='Valid AUC', marker='s', linestyle='-', linewidth=2, markersize=5, color='orange')
    plt.plot(epochs, data[phase]['test_auc'], label='Test AUC', marker='^', linestyle='-', linewidth=2, markersize=5, color='green')

    # # 虛線：early stop
    # if data[phase]['early_stop']:
    #     plt.axvline(x=data[phase]['early_stop'], linestyle='--', color='black',
    #                 label=f"Early Stop @ Epoch {data[phase]['early_stop']}")

    # 格式設定
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title(f'{phase.upper()} AUC vs Epoch')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"已儲存圖表至 {output_path}")
    else:
        plt.show()


def main():
    log_path = 'logs/NAC/NAC_training_2025-06-08_22-50-22.log'  # ✅ 替換成你的實際檔案路徑
    data = parse_log(log_path)

    print("Step 1 最終 Validation AUC:", data['step1']['final_val_auc'])
    print("Step 2 最終 Test AUC:", data['step2']['final_test_auc'])

    plot_auc_curves(data, 'step1', output_path='../plot/step1_nac_auc.png')
    plot_auc_curves(data, 'step2', output_path='../plot/step2_nac_auc.png')


if __name__ == "__main__":
    main()
