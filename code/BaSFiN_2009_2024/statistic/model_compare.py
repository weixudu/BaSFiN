import re
import numpy as np

def parse_log_file(filepath):
    # 存放結果
    all_results = {}

    # 暫存目前年份
    current_year = None

    # 暫存目前年份的 trial aucs
    trial_aucs = {}

    # 讀取檔案
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # 偵測開始年份
            m_year = re.search(r'開始處理年份：(\d{4})', line)
            if m_year:
                # 如果換年，先儲存上一年結果
                if current_year is not None and trial_aucs:
                    all_results[current_year] = trial_aucs
                # 換年份
                current_year = int(m_year.group(1))
                trial_aucs = {}
                continue

            # 偵測 Best Test AUC
            m_auc = re.search(r'Trial\s+(\d+).*Best Test AUC\s+([0-9.]+)', line)
            if m_auc and current_year is not None:
                trial_id = int(m_auc.group(1))
                auc_val = float(m_auc.group(2))
                trial_aucs[trial_id] = auc_val

        # 檔案最後一段也要存
        if current_year is not None and trial_aucs:
            all_results[current_year] = trial_aucs

    return all_results

def print_yearly_statistics(all_results, trial_range=range(0, 5)):
    for year in sorted(all_results.keys()):
        auc_list = []
        for trial in trial_range:
            if trial in all_results[year]:
                auc_list.append(all_results[year][trial])

        if len(auc_list) < len(trial_range):
            print(f"年份 {year}: 試次不足，找到 {len(auc_list)} 個 Trial")
            continue

        mean_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        print(f"年份 {year}: mean={mean_auc:.4f}, std={std_auc:.4f}")

if __name__ == "__main__":
    # 替換成你的檔案路徑
    log_file_path = "logs/NAC+/BaSFiN_Freeze_linear_noInter_20250708_190221.log"

    results = parse_log_file(log_file_path)
    print_yearly_statistics(results)
