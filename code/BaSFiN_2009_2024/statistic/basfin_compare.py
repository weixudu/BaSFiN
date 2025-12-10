import os
import re
import statistics

def extract_stage1_best_auc_from_log(filepath):
    """
    解析檔案裡所有 Stage-1 的 Trial Best AUC
    回傳 list of float
    """
    lines = read_file_lines(filepath)
    auc_list = []
    in_stage1_trial = False

    for line in lines:
        line = line.strip()

        # 進入 Stage-1 trial 的區段
        if re.search(r"Stage-1", line):
            in_stage1_trial = True

        # 只在 Stage-1 的 Trial 狀態下抓 Best AUC
        if in_stage1_trial:
            m = re.search(r'Best AUC\s+([0-9.]+)', line)
            if m:
                auc = float(m.group(1))
                auc_list.append(auc)
                in_stage1_trial = False  # 該 trial 結束後關閉
    return auc_list


def process_all_folders(parent_folder):
    """
    走訪 parent_folder 內所有 NAC+_{year} 的子資料夾
    每個子資料夾裡找所有 BaSFiN_Freeze_ 開頭的檔案
    解析 Stage-1 的所有 Best AUC
    計算平均與標準差
    """
    results = {}

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 檢查名稱符合 NAC+_{year}
        m = re.match(r'NAC\+_(\d{4})$', folder_name)
        if not m:
            print(f"跳過非目標資料夾：{folder_name}")
            continue

        year = int(m.group(1))
        all_aucs = []

        # 找所有符合檔案
        for filename in os.listdir(folder_path):
            if filename.startswith("BaSFiN_Freeze_"):
                log_path = os.path.join(folder_path, filename)
                aucs = extract_stage1_best_auc_from_log(log_path)
                if aucs:
                    all_aucs.extend(aucs)
                else:
                    print(f"檔案 {filename} 中未找到 Stage-1 Best AUC")

        if all_aucs:
            mean = statistics.mean(all_aucs)
            std = statistics.stdev(all_aucs) if len(all_aucs) > 1 else 0.0
            results[year] = (mean, std)
            print(f"年份 {year}: mean={mean:.4f}, std={std:.4f}, trials={len(all_aucs)}")
        else:
            print(f"年份 {year}: 沒有找到任何 Stage-1 Best AUC")

    return results


def read_file_lines(filepath):
    """
    嘗試用多種編碼開啟檔案，避免 UnicodeDecodeError
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp950"]
    for enc in encodings_to_try:
        try:
            with open(filepath, encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.readlines()


if __name__ == "__main__":
    parent_folder_path = r"C:\Users\Rain\Desktop\碩論\code\logs"

    results = process_all_folders(parent_folder_path)

    print("\n=== 所有結果 ===")
    for year in sorted(results.keys()):
        mean, std = results[year]
        print(f"年份 {year}: mean={mean:.4f}, std={std:.4f}")
