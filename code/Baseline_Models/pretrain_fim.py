import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import random
import logging
from datetime import datetime
from data import Data
from co_fim2 import NAC_ANFM
from BaS import NAC_BBB
from bc_fim2 import FIModel

# ========= 基本設定 ===================================================
SEED = 42
NUM_TRIALS = 5             # 每模型每年份執行次數
BASE_SEED  = SEED           # 第一個試驗的種子

device = torch.device('cpu')  # 若需 GPU 改為 'cuda'
n_epochs = 200
batch_size = 32
learning_rate = 0.001
team_size = 5
patience = 5

# ------------ ANFM 超參數 -------------
anfm_player_dim = 31
anfm_hidden_dim = 55
anfm_need = False
anfm_drop = 0.245
anfm_mlplayer = 35

# ------------ NAC_BBB 超參數 ----------
kl_weight = 0.01519
num_samples = 100

# ------------ FIModel 超參數 ----------
bc_player_dim = 54
bc_intermediate_dim = 20
bc_drop = 0.274
bc_mlplayer = 38
bc_need = False

# ========= 公用函式 ====================================================
def set_global_seed(seed: int):
    """統一設定隨機種子，確保每次試驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def evaluate(pred, label):
    """計算 AUC、Accuracy、LogLoss。"""
    if not isinstance(pred, np.ndarray):
        pred = pred.cpu().detach().numpy()
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5).astype(int)
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss

def elbo_loss(model, X, y, kl_weight, num_samples, device):
    """
    NAC_BBB 的 ELBO（數值穩定版）
    """
    y_tensor = torch.tensor(y, dtype=torch.float, device=device)
    y_repeated = y_tensor.unsqueeze(0).repeat(num_samples, 1)

    # logits ∈ R
    logits, _ = model(X, num_samples=num_samples)

    # ✅ 正確：BCEWithLogitsLoss
    log_likelihood = -nn.BCEWithLogitsLoss(reduction='sum')(
        logits, y_repeated
    ) / num_samples

    kl_loss = model.kl_divergence()
    return -log_likelihood + kl_weight * kl_loss


# ========= 三種模型訓練函式 ===========================================
def train_anfm(dataset, criterion, total_step, ema_tensor_path, game_id_mapping_path, log_prefix='ANFM'):
    model = NAC_ANFM(
        n_player=dataset.n_individual,
        team_size=team_size,
        player_dim=anfm_player_dim,
        hidden_dim=anfm_hidden_dim,
        need_att=False,
        mlp_hidden_dim=anfm_mlplayer,
        dropout_rate=anfm_drop,
        device=device,
        ema_tensor_path=ema_tensor_path,
        game_id_mapping_path=game_id_mapping_path
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate * 0.1, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2, min_lr=1e-6)

    best_valid_auc = -float('inf')
    best_metrics = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # --------- 訓練 ---------
        model.train()
        total_loss = 0
        for i, (X, y) in enumerate(dataset.get_batch(batch_size, shuffle=False)):
            X_tensor = torch.LongTensor(X).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float, device=device)
            pred, _, _ = model(X_tensor)
            loss = criterion(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # --------- 評估 ---------
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}

        for phase in phases:
            preds, labels = [], []
            for X, y in dataset.get_batch(batch_size, phase, shuffle=False):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    pred, _, _ = model(X_tensor)
                    preds.append(pred)
                    labels.append(y)
            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)
            results[phase] = dict(auc=auc, acc=acc, logloss=logloss)

        scheduler.step(results['valid']['auc'])

        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_metrics = results
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_valid_auc, best_metrics

def train_nac_bbb(dataset, total_step, kl_weight, num_samples, log_prefix='NAC_BBB'):
    model = NAC_BBB(
        n_player=dataset.n_individual,
        team_size=team_size,
        device=device,
        prior_mu=0,
        prior_sigma=1
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6
    )

    best_valid_auc = -float('inf')
    best_metrics = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # -------- train --------
        model.train()
        for X, y in dataset.get_batch(batch_size, shuffle=False):
            X_tensor = torch.LongTensor(X).to(device)
            loss = elbo_loss(model, X_tensor, y, kl_weight, num_samples, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -------- eval --------
        model.eval()
        results = {}

        for phase in ['train', 'valid', 'test']:
            preds, labels = [], []
            for X, y in dataset.get_batch(batch_size, phase, shuffle=False):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    logits, _ = model(X_tensor, num_samples=num_samples)

                    # ✅ 評估時再 sigmoid + 平均
                    prob = torch.sigmoid(logits).mean(dim=0)

                    preds.append(prob.cpu())
                    labels.append(y)

            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)
            results[phase] = dict(auc=auc, acc=acc, logloss=logloss)

        scheduler.step(results['valid']['auc'])

        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_metrics = results
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_valid_auc, best_metrics


def train_fimodel(dataset, criterion, total_step,
                  ema_tensor_path, game_id_mapping_path, log_prefix='FIModel'):
    model = FIModel(
        n_player=dataset.n_individual,
        player_dim=bc_player_dim,
        intermediate_dim=bc_intermediate_dim,
        dropout_rate=bc_drop,
        mlp_hidden_dim=bc_mlplayer,
        team_size=team_size,
        device=device,
        ema_tensor_path=ema_tensor_path,
        game_id_mapping_path=game_id_mapping_path,
        need_att=bc_need
    ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate * 0.1, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2, min_lr=1e-6)

    best_valid_auc = -float('inf')
    best_metrics = None
    patience_counter = 0

    for epoch in range(n_epochs):
        # --------- 訓練 ---------
        model.train()
        for X, y in dataset.get_batch(batch_size, shuffle=False):
            X_tensor = torch.LongTensor(X).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float, device=device)
            pred, _, _ = model(X_tensor)
            loss = criterion(pred, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --------- 評估 ---------
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}

        for phase in phases:
            preds, labels = [], []
            for X, y in dataset.get_batch(batch_size, phase, shuffle=False):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    pred, _, _ = model(X_tensor)
                    preds.append(pred)
                    labels.append(y)
            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)
            results[phase] = dict(auc=auc, acc=acc, logloss=logloss)

        scheduler.step(results['valid']['auc'])

        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_metrics = results
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_state, best_valid_auc, best_metrics

# ========= 通用試驗執行器 =============================================
def run_trials(train_fn, dataset_kwargs, tag, year, save_dir, *train_args):
    """重複 NUM_TRIALS 次，保留 valid AUC 最佳模型。"""
    best_auc = -float('inf')
    best_state, best_metrics, best_trial_idx = None, None, -1

    for trial in range(NUM_TRIALS):
        trial_seed = BASE_SEED + trial
        set_global_seed(trial_seed)

        # 每試驗重建資料
        dataset = Data(**dataset_kwargs, seed=trial_seed)
        total_step = len(dataset.train) // batch_size + 1

        print(f"[{tag}] Year {year} Trial {trial+1}/{NUM_TRIALS} (seed={trial_seed})")

        state, valid_auc, metrics = train_fn(dataset, *train_args)

        if valid_auc > best_auc:
            best_auc = valid_auc
            best_state = state
            best_metrics = metrics
            best_trial_idx = trial

        # 若使用 GPU，可釋放顯存
        del state
        torch.cuda.empty_cache()

    # 儲存最佳權重
    best_path = os.path.join(save_dir, f'{tag}_best_{year}.pth')
    torch.save(best_state, best_path)
    print(f"[{tag}] Year {year} ➜ Best AUC={best_auc:.4f} (trial {best_trial_idx+1}) 存檔：{best_path}")

    return best_auc, best_metrics, best_trial_idx

# ========= 主程式 ====================================================
def main():
    # 需要處理的年度
    year_suffixes = ["2020", "2021", "2022", "2023", "2024"]
    criterion = nn.BCELoss()

    for year in year_suffixes:
        print("=" * 80)
        print(f"{year}")
        print("=" * 80)

        # 動態建立路徑
        start_year = str(int(year) - 11)
        path = f'../data/final_data/data_{start_year}_{year}.csv'
        ema_tensor_path = f'../data/ema_tensor/ematensor_{year}.pt'
        game_id_mapping_path = f'../data/tensor/game_id_mapping_{year}.json'
        model_save_dir = f'model/pretrain_BaSFiN_model_{year}'
        os.makedirs(model_save_dir, exist_ok=True)

        # 共同 Data 參數
        data_kwargs = dict(path=path, team_size=team_size)

        # ------------------- ANFM -------------------
        anfm_auc, anfm_metrics, anfm_trial = run_trials(
            train_anfm,
            data_kwargs,
            'anfm',
            year,
            model_save_dir,
            criterion,
            len(Data(**data_kwargs).train)//batch_size + 1,   # total_step
            ema_tensor_path,
            game_id_mapping_path
        )

        # ------------------- NAC_BBB ----------------
        bbb_auc, bbb_metrics, bbb_trial = run_trials(
            train_nac_bbb,
            data_kwargs,
            'nac_bbb',
            year,
            model_save_dir,
            len(Data(**data_kwargs).train)//batch_size + 1,   # total_step
            kl_weight,
            num_samples
        )

        # ------------------- FIModel ---------------
        fim_auc, fim_metrics, fim_trial = run_trials(
            train_fimodel,
            data_kwargs,
            'fimodel',
            year,
            model_save_dir,
            criterion,
            len(Data(**data_kwargs).train)//batch_size + 1,   # total_step
            ema_tensor_path,
            game_id_mapping_path
        )

        # ------------- freeze_config 紀錄 ----------
        freeze_dict = {
            'nac_anfm': {'best_trial': anfm_trial, 'best_auc': anfm_auc},
            'nac_bbb' : {'best_trial': bbb_trial,  'best_auc': bbb_auc},
            'fimodel' : {'best_trial': fim_trial,  'best_auc': fim_auc}
        }
        freeze_save_path = os.path.join(model_save_dir, f'freeze_config_{year}.json')
        with open(freeze_save_path, 'w', encoding='utf-8') as f:
            json.dump(freeze_dict, f, indent=4, ensure_ascii=False)
        print(f"freeze_config 已儲存：{freeze_save_path}")

        # ------------- 年度總結 ---------------------
        print(f"== {year}  Valid AUC ==")
        print(f"ANFM   : {anfm_auc:.4f} (trial {anfm_trial+1})")
        print(f"NAC_BBB: {bbb_auc:.4f} (trial {bbb_trial+1})")
        print(f"FIModel: {fim_auc:.4f} (trial {fim_trial+1})")

if __name__ == "__main__":
    main()
