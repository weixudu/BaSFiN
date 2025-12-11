from __future__ import annotations
import statistics as stats   # 放到 import 區
import re
from pathlib import Path
from typing import Dict, List, Tuple

# === 可自行修改的參數 ===
# LOG_PATH = "logs/final_logs/Best_parameter/cofim_random_10_20250623_190918.log"#best 

LOG_PATH = "logs/final_logs/Best_parameter/cofim noema.log" #NOEMA
TRIAL_IDS = [0, 1, 2,3,4,5,6,7,8,9]     

# ────────────── 新版 Step 1 Regex ──────────────
RE_EARLY_STOP_NEW = re.compile(
    r"Phase\s+step\d+,\s*"
    r"Player Dim\s*(?P<player>\d+),\s*"
    r"Intermediate Dim\s*(?P<intermediate>\d+),\s*"
    r"Dropout\s*(?P<dropout>[\d.]+),\s*"
    r"MLP Hidden\s*(?P<mlp>\d+),\s*"
    r"Need Att\s*(?P<needatt>True|False),\s*"
    r"Trial\s*(?P<trial>\d+),\s*"
    r"Early stopping triggered after\s*\d+\s*epochs,\s*best epoch was\s*(?P<epoch>\d+)"
)

# ——— ❷ 每個 Epoch 指標列 ————————————————————————
METRIC_TMPL_NEW = (
    r"Phase\s+step\d+,\s*"
    r"Player Dim \d+,\s*Intermediate Dim \d+,\s*"
    r"Dropout [\d.]+,\s*MLP Hidden \d+,\s*"
    r"Need Att (?:True|False),\s*"
    r"Trial {trial},\s*"
    r"Epoch \[{epoch}/\d+\],\s*{split} AUC: (?P<auc>[\d.]+),\s*"
    r"Acc: (?P<acc>[\d.]+),\s*Logloss: (?P<logloss>[\d.]+)"
)

# ────────────── 舊版 Step 1 Regex ──────────────
RE_EARLY_STOP_OLD = re.compile(
    r"Phase\s+step\d+,\s*Trial\s+(?P<trial>\d+),\s*"
    r"Early stopping triggered after\s+\d+\s+epochs,\s*"
    r"best epoch was\s+(?P<epoch>\d+)"
)

METRIC_TMPL_OLD = (
    r"Phase\s+step\d+,\s*Trial\s+{trial},\s*"
    r"Epoch\s*\[{epoch}/\d+],\s*{split}\s+AUC:\s*(?P<auc>[\d.]+),\s*"
    r"Acc:\s*(?P<acc>[\d.]+),\s*Logloss:\s*(?P<logloss>[\d.]+)"
)

SPLITS = ("Train", "Valid", "Test")

# ──────────────────────── Step 2 Regex ───────────────────────────
RE_STEP2 = re.compile(
    r"Test Average AUC:\s*(?P<auc>[\d.]+),\s*Avg Acc:\s*(?P<acc>[\d.]+),\s*Avg Logloss:\s*(?P<logloss>[\d.]+)"
)

# ────────────────────────────────────────────────────────────────

def parse_step1(lines: List[str]) -> Tuple[Dict[int, int], Dict[Tuple[int, int, str], Dict[str, float]]]:
    """解析 Step 1：回傳 (best_epochs, metrics)，新版找不到就換舊版"""
    best_epochs: Dict[int, int] = {}
    metrics: Dict[Tuple[int, int, str], Dict[str, float]] = {}

    # --- 先用新版找 best epoch ---
    for line in lines:
        if (m := RE_EARLY_STOP_NEW.search(line)):
            trial = int(m.group("trial"))
            epoch = int(m.group("epoch"))
            if trial in TRIAL_IDS:
                best_epochs[trial] = epoch

    # --- 新版找不到就用舊版 ---
    if not best_epochs:
        for line in lines:
            if (m := RE_EARLY_STOP_OLD.search(line)):
                trial = int(m.group("trial"))
                epoch = int(m.group("epoch"))
                if trial in TRIAL_IDS:
                    best_epochs[trial] = epoch

    # --- 用對應版本解析各 split 指標 ---
    for trial, epoch in best_epochs.items():
        found = False
        # 先試新版
        for split in SPLITS:
            regex = re.compile(METRIC_TMPL_NEW.format(trial=trial, epoch=epoch, split=split))
            for line in lines:
                if (m := regex.search(line)):
                    metrics[(trial, epoch, split.lower())] = {
                        "auc": float(m.group("auc")),
                        "acc": float(m.group("acc")),
                        "logloss": float(m.group("logloss"))
                    }
                    found = True
                    break
        # 新版都沒找到就換舊版
        if not found:
            for split in SPLITS:
                regex = re.compile(METRIC_TMPL_OLD.format(trial=trial, epoch=epoch, split=split))
                for line in lines:
                    if (m := regex.search(line)):
                        metrics[(trial, epoch, split.lower())] = {
                            "auc": float(m.group("auc")),
                            "acc": float(m.group("acc")),
                            "logloss": float(m.group("logloss"))
                        }
                        break

    return best_epochs, metrics


def average_metrics(per_trial: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """將多個 trial 的單一 split 指標取平均"""
    keys = next(iter(per_trial.values())).keys()
    return {k: sum(m[k] for m in per_trial.values()) / len(per_trial) for k in keys}


def parse_step2(lines: List[str]) -> List[Dict[str, float]]:
    """蒐集所有 Step 2 Test 結果行，回傳 list"""
    recs = []
    for line in lines:
        if (m := RE_STEP2.search(line)):
            recs.append({k: float(m.group(k)) for k in ("auc", "acc", "logloss")})
    return recs



def main() -> None:
    path = Path(LOG_PATH)
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {path}")

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # ======================= Step 1 =======================
    best_epochs, raw_metrics = parse_step1(lines)
    if not best_epochs:
        print("❌ 未找到任何 Early-Stopping / best epoch 資訊！\n")
    else:
        trial_data: Dict[int, Dict[str, Dict[str, float] | int]] = {}
        for trial, epoch in best_epochs.items():
            trial_metrics = {}
            for split in SPLITS:
                key = (trial, epoch, split.lower())
                if key in raw_metrics:
                    trial_metrics[split.lower()] = raw_metrics[key]
            trial_data[trial] = {"epoch": epoch, **trial_metrics}

        # ---- 列印各 trial ----
        print("=== Step 1 — Per-Trial Best-Epoch Metrics ===")
        test_aucs_step1 = []  
        for trial in sorted(trial_data):
            d = trial_data[trial]
            print(f"Trial {trial} — Best Epoch: {d['epoch']}")
            for split in SPLITS:
                s = split.lower()
                if s in d:
                    m = d[s]
                    print(f"  {split:<5}: AUC {m['auc']:.4f}, Acc {m['acc']:.4f}, Logloss {m['logloss']:.4f}")
                    if split == "Test":
                        test_aucs_step1.append(m["auc"])
            print()
        # ---- 計算平均 ± stdev ----
        avg_epoch = sum(d["epoch"] for d in trial_data.values()) / len(trial_data)
        mean_auc_step1  = stats.mean(test_aucs_step1)
        std_auc_step1   = stats.stdev(test_aucs_step1) if len(test_aucs_step1) > 1 else 0

        avg_metrics = {
            s.lower(): average_metrics({t: d[s.lower()] for t, d in trial_data.items()})
            for s in SPLITS
        }

        print("=== Step 1 — Averaged Best-Epoch Results ===")
        print(f"Average Best Epoch: {avg_epoch:.2f}")
        for split in SPLITS:
            s = split.lower()
            m = avg_metrics[s]
            if split == "Test":
                print(f"  {split:<5}: AUC {m['auc']:.4f} ± {std_auc_step1:.4f}, "
                    f"Acc {m['acc']:.4f}, Logloss {m['logloss']:.4f}")
            else:
                print(f"  {split:<5}: AUC {m['auc']:.4f}, Acc {m['acc']:.4f}, "
                    f"Logloss {m['logloss']:.4f}")
        print()

    # ======================= Step 2 =======================
    step2_records = parse_step2(lines)
    if step2_records:
        aucs2 = [r["auc"] for r in step2_records]
        mean_auc2 = stats.mean(aucs2)
        std_auc2  = stats.stdev(aucs2) if len(aucs2) > 1 else 0
        print("=== Step 2 — Final Test Results ===")
        print(f"Test Average AUC: {mean_auc2:.4f} ± {std_auc2:.4f}")
        print(f"Avg Acc        : {stats.mean(r['acc'] for r in step2_records):.4f}")
        print(f"Avg Logloss    : {stats.mean(r['logloss'] for r in step2_records):.4f}")
    else:
        print("⚠️  未找到 Step 2 最終結果資訊。")


if __name__ == "__main__":
    main()
