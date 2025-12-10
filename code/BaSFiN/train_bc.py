# -*- coding: utf-8 -*-
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import time, os, json, random, logging, gc, uuid, heapq        # heapq NEW
import pandas as pd
from collections import defaultdict                             # NEW
from data import Data
from bc_fim import *                                            # FIModel & utils

# ---------------------------------------------------------------
# 1. 亂數與環境
# ---------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device('cpu')

n_epochs = 200
batch_size = 32
learning_rate = 1e-4

player_dims        = [50]
intermediate_dims  = [37]
dropout_rates      = [0.364]
mlp_hidden_dims    = [53]
need_atts          = [True]         
num_trials         = 1             

patience   = 5
team_size  = 5
FOCUS_PID  = 1  
MIN_MATCH_CNT = 5

path                   = '../data/final_data/data_2013_2024.csv'
ema_tensor_path        = '../data/ema_tensor/ematensor.pt'
game_id_mapping_path   = '../data/tensor/game_id_mapping.json'

output_dir = '../output/FIM'
model_dir  = os.path.join(output_dir, 'models')
log_dir    = 'logs/feature'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
timestamp  = time.strftime("%Y%m%d_%H%M%S")

log_file = os.path.join(log_dir, f'FiN_bc_pid_1_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


def evaluate(pred, label):
    if not isinstance(pred, np.ndarray):
        pred = pred.cpu().detach().numpy()
    pred  = pred.reshape(-1)
    label = label.reshape(-1)
    auc   = metrics.roc_auc_score(label, pred)

    pred_clip = np.clip(pred, 0.001, 0.999)
    logloss   = metrics.log_loss(label, pred_clip)
    pred_bin  = (pred > 0.5).astype(int)
    acc       = (label == pred_bin).mean()
    return auc, acc, logloss

# ---------- 配對分數統計 ----------
def combine(team_size: int = 5):
    idx1, idx2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            idx1.append(i)
            idx2.append(j)
    return idx1, idx2

IDX1, IDX2 = combine(team_size)

def accumulate_pair_stats(stats_all, stats_focus,
                          pair_scores, team_A_idx, team_B_idx,
                          focus_idx, idx_to_pid,
                          focus_on_attack=True):
    ids_A_pair = team_A_idx[IDX1]; ids_B_pair = team_B_idx[IDX2]
    for s, a_idx, b_idx in zip(pair_scores.tolist(), ids_A_pair, ids_B_pair):
        a_pid, b_pid = idx_to_pid(a_idx), idx_to_pid(b_idx)  # ← 轉回原始 PID
        key = (a_pid, b_pid)
        stats_all[key][0] += s;  stats_all[key][1] += 1
        if focus_idx is not None:
            if focus_on_attack and a_idx == focus_idx:
                stats_focus[key][0] += s
                stats_focus[key][1] += 1
            elif not focus_on_attack and b_idx == focus_idx:
                stats_focus[key][0] += s
                stats_focus[key][1] += 1



def top_bottom_k(stats_dict, k: int = 5, min_cnt: int = MIN_MATCH_CNT):
    vec = [ (pair, s_cnt[0] / s_cnt[1])
            for pair, s_cnt in stats_dict.items()
            if s_cnt[1] >= min_cnt ]
    

    if not vec:                      
        return [], []

    top_k = heapq.nlargest(k,  vec, key=lambda x: x[1])
    bot_k = heapq.nsmallest(k, vec, key=lambda x: x[1])
    return top_k, bot_k

# ---------- 平均分排行榜 (門檻版) ----------
def avg_top_bottom_k(stats_dict: dict, k: int = 5, min_cnt: int = MIN_MATCH_CNT):
    records = [
        (pair, total / cnt, cnt)
        for pair, (total, cnt) in stats_dict.items()
        if cnt >= min_cnt
    ]

    if not records:           
        return [], []

    records.sort(key=lambda x: x[1], reverse=True)  
    top_k = records[:k]         
    bot_k = records[-k:][::-1]   
    return top_k, bot_k



def train_and_evaluate(player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim,
                       dataset, game_ids_train, game_ids_valid, game_ids_test,
                       ema_tensor_path, combo_idx, trial_idx,
                       need_att, phase='step1'):

    seed = SEED + trial_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = FIModel(
        n_player = dataset.n_individual,
        player_dim = player_dim,
        intermediate_dim = intermediate_dim,
        dropout_rate = dropout_rate,
        mlp_hidden_dim = mlp_hidden_dim,
        team_size = team_size,
        device = device,
        ema_tensor_path = ema_tensor_path,
        game_id_mapping_path = game_id_mapping_path,
        need_att = need_att
    ).to(device)

    optimizer  = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=2, min_lr=1e-6)
    criterion  = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1

    best_valid_auc  = 0.0
    patience_cnt    = 0
    best_epoch      = 0
    best_metrics    = None
    best_model_state= None
    best_test_preds = None
    best_test_labels= None

    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, shuffle=False)
        total_loss = 0.0

        for i, (X, y) in enumerate(batch_gen):
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
            X_tensor = torch.tensor(X, dtype=torch.long, device=device)

            probs, _, _ = model(X_tensor)
            loss = criterion(probs, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                logger.info(
                    f'Phase {phase}, Trial {trial_idx}, '
                    f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_step}], '
                    f'Loss: {loss.item():.4f}'
                )

        avg_loss = total_loss / total_step
        logger.info(f'Phase {phase}, Epoch [{epoch+1}/{n_epochs}], Avg Loss: {avg_loss:.4f}')

        # -------- Validation / Test --------
        model.eval()
        parts = ['train', 'valid', 'test'] if phase == 'step1' else ['train', 'test']
        results = {}
        game_ids_dict = {'train': game_ids_train, 'valid': game_ids_valid, 'test': game_ids_test}

        for part in parts:
            preds, labels = [], []
            batch_gen = dataset.get_batch(batch_size, part, shuffle=False)

            for X, y in batch_gen:
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.long, device=device)
                    probs, _, _ = model(X_tensor)
                    preds.append(probs)
                    labels.append(y)

            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds).cpu().numpy()
            auc, acc, logloss = evaluate(y_pred, y_true)

            results[part] = dict(auc=auc, acc=acc, logloss=logloss)
            logger.info(
                f'Phase {phase}, Trial {trial_idx}, Epoch [{epoch+1}/{n_epochs}], '
                f'{part.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}'
            )

        current_auc = results.get('valid', results['test'])['auc']
        scheduler.step(current_auc)

        if current_auc > best_valid_auc:
            best_valid_auc = current_auc
            best_epoch     = epoch + 1
            patience_cnt   = 0
            best_metrics   = results
            best_model_state = model.state_dict()
            model_path = os.path.join(model_dir,
                        f'best_model_combo{combo_idx}_trial{trial_idx}_{phase}_{timestamp}.pth')
            torch.save(best_model_state, model_path)
            logger.info(f'Phase {phase}, Trial {trial_idx}, New best AUC={best_valid_auc:.4f} '
                        f'at epoch {best_epoch}, Model saved → {model_path}')
        else:
            patience_cnt += 1
            logger.info(f'Phase {phase}, Trial {trial_idx}, Patience {patience_cnt}/{patience}')

        if patience_cnt >= patience:
            logger.info(f'Phase {phase}, Trial {trial_idx}, Early-stop at epoch {epoch+1}')
            break

    return best_valid_auc, best_metrics, best_model_state, best_test_preds, best_test_labels

def main():
    dataset = Data(path, team_size=team_size, seed=SEED)

    # ---------- 建立 index ↔ pid 轉換 ----------
    # （與原版相同）
    if isinstance(dataset.index_to_player_id, dict):
        idx2pid_dict = dataset.index_to_player_id
        pid2idx_dict = {pid: idx for idx, pid in idx2pid_dict.items()}
        def idx_to_pid(i): return idx2pid_dict.get(int(i), int(i))
    else:
        idx2pid_list = list(dataset.index_to_player_id)
        pid2idx_dict = {pid: idx for idx, pid in enumerate(idx2pid_list)}
        def idx_to_pid(i): i = int(i); return idx2pid_list[i] if i < len(idx2pid_list) else i

    focus_idx = pid2idx_dict.get(FOCUS_PID, None)
    if focus_idx is None:
        logger.warning(f'⚠️  FOCUS_PID {FOCUS_PID} 不存在，僅輸出全配對統計')
    else:
        logger.info(f'FOCUS_PID {FOCUS_PID} → 內部 index {focus_idx}')

    game_ids_train = dataset.train[:, 0]
    game_ids_valid = dataset.valid[:, 0]
    game_ids_test  = dataset.test[:, 0]

    # ---------- STEP 1 ----------
    logger.info('\n=== STEP 1: Hyper-search (單 trial) ===')
    best_results, combo_results = [], {}; combo_idx = 0

    for player_dim in player_dims:
        for intermediate_dim in intermediate_dims:
            for dropout_rate in dropout_rates:
                for mlp_hidden_dim in mlp_hidden_dims:
                    for need_att in need_atts:
                        logger.info(
                            f'\n=== COMBO {combo_idx} → '
                            f'player={player_dim}, inter={intermediate_dim}, '
                            f'dropout={dropout_rate}, mlp={mlp_hidden_dim}, need_att={need_att}'
                        )

                        aucs, metrics_list, model_states = [], [], []
                        for trial_idx in range(num_trials):
                            val_auc, metrics, model_state, *_ = train_and_evaluate(
                                player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim,
                                dataset, game_ids_train, game_ids_valid, game_ids_test,
                                ema_tensor_path,
                                combo_idx, trial_idx, need_att, phase='step1'
                            )
                            aucs.append(val_auc)
                            metrics_list.append(metrics)
                            model_states.append(model_state)

                        avg_auc = np.mean(aucs)
                        key = (player_dim, intermediate_dim,
                               dropout_rate, mlp_hidden_dim, need_att)
                        combo_results[key] = dict(
                            avg_auc=avg_auc,
                            aucs=aucs,
                            metrics=metrics_list,
                            model_states=model_states
                        )
                        best_results.append({
                            'player_dim': player_dim, 'intermediate_dim': intermediate_dim,
                            'dropout_rate': dropout_rate, 'mlp_hidden_dim': mlp_hidden_dim,
                            'need_att': need_att, 'avg_auc': avg_auc,
                            'metrics': metrics_list,
                            'best_trial_idx': int(np.argmax(aucs))
                        })
                        combo_idx += 1

    best_combo = max(best_results, key=lambda x: x['avg_auc'])
    logger.info(f"\nBest combo：{best_combo}")


    # ===== NEW 區塊結束 ===============================================
    # ---------- STEP 2 ----------
    logger.info('\n=== STEP 2: Retrain on Train+Valid ===')
    dataset.expand_training_data()
    game_ids_train = np.concatenate([game_ids_train, game_ids_valid])
    game_ids_valid = np.array([])

    test_auc, metrics, model_state, test_preds, test_labels = train_and_evaluate(
        best_combo['player_dim'], best_combo['intermediate_dim'],
        best_combo['dropout_rate'], best_combo['mlp_hidden_dim'],
        dataset, game_ids_train, game_ids_valid, game_ids_test,
        ema_tensor_path, 
        combo_idx=0, trial_idx=0, need_att=best_combo['need_att'], phase='step2'
    )

    # ---------- 載入最終模型 ----------
    best_model = FIModel(
        n_player = dataset.n_individual,
        player_dim = best_combo['player_dim'],
        intermediate_dim = best_combo['intermediate_dim'],
        dropout_rate = best_combo['dropout_rate'],
        mlp_hidden_dim = best_combo['mlp_hidden_dim'],
        team_size = team_size,
        device = device,
        ema_tensor_path = ema_tensor_path,
        game_id_mapping_path = game_id_mapping_path,
        need_att = best_combo['need_att']
    ).to(device)
    best_model.load_state_dict(model_state)
    best_model.eval()

    # -----------------------------------------------------------
    # 計算 Test Set 的配對分數統計（含 PID=1）
    # -----------------------------------------------------------
    # ===== NEW (Final Train+Valid statistics) =====================
    logger.info('\n=== Pairwise score statistics (Train+Valid, Final weights) ===')

    stats_AD_all_tv, stats_AD_focus_tv = defaultdict(lambda: [0.,0]), defaultdict(lambda: [0.,0])
    stats_DA_all_tv, stats_DA_focus_tv = defaultdict(lambda: [0.,0]), defaultdict(lambda: [0.,0])

    # -------- 判斷 valid 是否為空 --------
    parts_tv = ['train']
    if getattr(dataset, 'valid', None) is not None and dataset.valid.size > 0:
        parts_tv.append('valid')

    for part in parts_tv:
        loader = dataset.get_batch(batch_size, part, shuffle=False)
        with torch.no_grad():
            for X, _ in loader:
                X_tensor = torch.as_tensor(X, dtype=torch.long, device=device)
                _, _, _, pair1, pair2 = best_model(X_tensor, need_pairwise=True)

                team_A = X[:, 1:1+team_size]
                team_B = X[:, 1+team_size:1+10]

                for b in range(X.shape[0]):
                    accumulate_pair_stats(
                        stats_AD_all_tv, stats_AD_focus_tv,
                        pair1[b].cpu(), team_A[b], team_B[b],
                        focus_idx, idx_to_pid,
                        focus_on_attack=True      # 攻→守方向，focus 必須是攻擊
                    )
                    accumulate_pair_stats(
                        stats_DA_all_tv, stats_DA_focus_tv,
                        pair2[b].cpu(), team_B[b], team_A[b],
                        focus_idx, idx_to_pid,
                        focus_on_attack=False     # 守→攻方向，focus 必須是防守
                    )



    def _print_stats(prefix, stats_all, stats_focus):
        top, bot = top_bottom_k(stats_all)
        avg_top, avg_bot = avg_top_bottom_k(stats_all, k=3)
        top_f, bot_f = top_bottom_k(stats_focus)
        avg_top_f, avg_bot_f = avg_top_bottom_k(stats_focus, k=3)
        logger.info(prefix)
        logger.info(f'Avg-Top-3 (all)         : {avg_top}')
        logger.info(f'Avg-Bot-3 (all)         : {avg_bot}')
        logger.info(f'Avg-Top-3 (PID={FOCUS_PID}) : {avg_top_f}')
        logger.info(f'Avg-Bot-3 (PID={FOCUS_PID}) : {avg_bot_f}')

    _print_stats('---  攻 → 守  (Train+Valid Final)  ---', stats_AD_all_tv, stats_AD_focus_tv)
    _print_stats('---  守 → 攻  (Train+Valid Final)  ---', stats_DA_all_tv, stats_DA_focus_tv)
    
if __name__ == '__main__':
    main()
