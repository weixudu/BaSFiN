import re
import pandas as pd

def load_player_mapping(csv_path):
    """讀取 mapping csv，回傳 dict: {player_id: player_name}"""
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df['player_id'], df['player_name']))
    return mapping

def replace_ids_in_text(text, mapping):
    """將文字中 np.int64(123) 的 id 換成選手名稱"""

    def replace_match(match):
        id_num = int(match.group(1))
        player_name = mapping.get(id_num)
        if player_name:
            return f"{player_name}"
        else:
            return f"UNKNOWN_ID_{id_num}"

    # np.int64(NUM)
    pattern = r"np\.int64\((\d+)\)"
    new_text = re.sub(pattern, replace_match, text)
    return new_text

def main():
    log_txt_path = "../output/pid_cofim.txt"        # 輸入檔
    mapping_csv = "../data/player_id_mapping_2009_2024.csv"
    output_txt_path = "../output/pid_cofim_name.txt"  # 輸出檔

    # 讀 mapping
    mapping = load_player_mapping(mapping_csv)

    # 讀 log
    with open(log_txt_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    # 替換
    replaced_content = replace_ids_in_text(log_content, mapping)

    # 寫出
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(replaced_content)

    print(f"✅ Done! Output written to: {output_txt_path}")

if __name__ == "__main__":
    main()
