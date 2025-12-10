import os
import re

def parse_auc_from_log(filepath):
    """
    讀取指定 log 檔案，找出含 Stage-1 (Step2) Test AUC 的行
    回傳 (mean, std) tuple
    """
    lines = read_file_lines(filepath)
    for line in lines:
        line = line.strip()
        m = re.search(r'Stage-1 \(Step2\) Test AUC:\s*([0-9.]+)\s*±\s*([0-9.]+)', line)
        if m:
            mean = float(m.group(1))
            std = float(m.group(2))
            return mean, std
    return None, None


def process_all_folders(parent_folder):
    """
    走訪 parent_folder 內的所有子資料夾
    子資料夾名稱格式：2013_2024
    """
    results = {}

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 從資料夾名稱取出「後面年份」
        m = re.search(r'_(\d{4})$', folder_name)
        if not m:
            print(f"無法解析年份：{folder_name}")
            continue
        year = int(m.group(1))

        # 找資料夾裡符合檔名開頭 BT_training 的檔
        for filename in os.listdir(folder_path):
            # if filename.startswith("BT_training"):
            if filename.startswith("NAC_training"):
                
                log_path = os.path.join(folder_path, filename)
                mean, std = parse_auc_from_log(log_path)
                if mean is not None and std is not None:
                    results[year] = (mean, std)
                    print(f"年份 {year}: mean={mean:.4f}, std={std:.4f}")
                else:
                    print(f"年份 {year}: 找不到 Stage-1 (Step2) Test AUC 資料 in {filename}")
                break
        else:
            print(f"資料夾 {folder_name} 中找不到 BT_training 開頭的檔案")

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
    # 如果都失敗，就忽略錯誤強制讀
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        return f.readlines()


if __name__ == "__main__":
    # 替換成你的總資料夾路徑
    # parent_folder_path = r"C:\Users\Rain\Desktop\碩論\code\logs\BT"
    
    parent_folder_path = r"C:\Users\Rain\Desktop\碩論\code\logs\NAC"

    results = process_all_folders(parent_folder_path)

    print("\n=== 所有結果 ===")
    for year in sorted(results.keys()):
        mean, std = results[year]
        print(f"年份 {year}: mean={mean:.4f}, std={std:.4f}")
