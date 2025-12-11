from pathlib import Path
import re
import pandas as pd

# === 路徑設定 ===
LOG_PATH      = Path("logs/NAC+/BaSFiN_noInter_nofreeze_20250902_160415.log")
CSV_WIDE_PATH = Path("../output/grads_all_no_freeze_trials.csv")   # 寬格式：每個梯度一欄

# === 正規表示式 ===
HEADER_VALID_LINE_PATTERN = re.compile(
    r"Stage\s+(?P<stage>\d+)\s*\|\s*"
    r"Trial\s+(?P<trial>\d+)\s*\|\s*"
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Valid\s+AUC\s+(?P<valid_auc>[0-9.eE+-]+)"
)

HEADER_TEST_LINE_PATTERN = re.compile(
    r"Stage\s+(?P<stage>\d+)\s*\|\s*"
    r"Trial\s+(?P<trial>\d+)\s*\|\s*"
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"Test\s+AUC\s+(?P<test_auc>[0-9.eE+-]+)"
)

TEST_LINE_PATTERN = re.compile(
    r"Test\s+AUC\s+(?P<test_auc>[0-9.eE+-]+)"
)

GRAD_LINE_PATTERN = re.compile(
    r"ScoreGrad\s+(?P<scoregrad>[0-9.eE+-]+)\s*\|\s*"
    r"ModGrad\s+z\s+(?P<modgrad_z>[0-9.eE+-]+)/comp\s+(?P<modgrad_comp>[0-9.eE+-]+)/coop\s+(?P<modgrad_coop>[0-9.eE+-]+)\s*\|\s*"
    r"ParamGrad\s+bbb\s+(?P<paramgrad_bbb>[0-9.eE+-]+)/fi\s+(?P<paramgrad_fi>[0-9.eE+-]+)/anfm\s+(?P<paramgrad_anfm>[0-9.eE+-]+)"
)

# === 解析函式 ===
def parse_log_file(path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    records = []
    with open(path, "r", encoding=encoding) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    i = 0
    while i + 2 < len(lines):
        header_line = lines[i]
        test_line = lines[i + 1]
        grad_line = lines[i + 2]

        m_valid_header = HEADER_VALID_LINE_PATTERN.search(header_line)
        m_test_header  = HEADER_TEST_LINE_PATTERN.search(header_line)
        m_test_line    = TEST_LINE_PATTERN.search(test_line)
        m_grad_line    = GRAD_LINE_PATTERN.search(grad_line)

        if (m_valid_header or m_test_header) and m_test_line and m_grad_line:
            if m_valid_header:
                stage = int(m_valid_header.group("stage"))
                trial = int(m_valid_header.group("trial"))
                epoch = int(m_valid_header.group("epoch"))
                test_auc = float(m_test_line.group("test_auc"))
            else:
                stage = int(m_test_header.group("stage"))
                trial = int(m_test_header.group("trial"))
                epoch = int(m_test_header.group("epoch"))
                test_auc = float(m_test_header.group("test_auc"))

            record = {
                "stage": stage,
                "trial": trial,
                "epoch": epoch,
                "TESTAUC": test_auc,
                "scoregrad": float(m_grad_line.group("scoregrad")),
                "modgrad_z": float(m_grad_line.group("modgrad_z")),
                "modgrad_comp": float(m_grad_line.group("modgrad_comp")),
                "modgrad_coop": float(m_grad_line.group("modgrad_coop")),
                "paramgrad_bbb": float(m_grad_line.group("paramgrad_bbb")),
                "paramgrad_fi": float(m_grad_line.group("paramgrad_fi")),
                "paramgrad_anfm": float(m_grad_line.group("paramgrad_anfm")),
            }
            records.append(record)
            i += 3
        else:
            # 不符合格式就往下一行跳
            i += 1

    if not records:
        raise ValueError("Log 檔中找不到任何符合格式的區段。")

    return pd.DataFrame(records)

# === 主程式 ===
if __name__ == "__main__":
    df = parse_log_file(LOG_PATH)

    # 欄位順序
    col_order = [
        "stage", "trial", "epoch", "TESTAUC",
        "scoregrad",
        "modgrad_z", "modgrad_comp", "modgrad_coop",
        "paramgrad_bbb", "paramgrad_fi", "paramgrad_anfm",
    ]
    df = df[col_order].sort_values(["stage", "epoch"]).reset_index(drop=True)

    # 輸出 CSV
    df.to_csv(CSV_WIDE_PATH, index=False)
    print(f"✓ 已輸出寬格式 {CSV_WIDE_PATH}（{len(df)} rows）")
