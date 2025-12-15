import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import random
import logging
import os
from datetime import datetime
from data import Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 與新版一致的超參數
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED) if SEED is not None else None
torch.cuda.manual_seed_all(SEED if SEED is not None else 0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

n_epochs = 200
batch_size = 32
learning_rate = 0.001
path = '../data/final_data/data_2013_2024.csv'
team_size = 5
num_trials = 1
early_stop_patience = 5
device = torch.device('cpu')
output_dir = '../output/BT'
model_dir = os.path.join(output_dir, 'models')  # 模型保存子目錄

# 日誌設置
log_dir = 'logs/BT'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)  # 確保模型目錄存在
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f'BT_training_SGD_{current_time}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class BT(nn.Module):
    def __init__(self, n_player):
        super(BT, self).__init__()
        assert n_player > 1
        self.skill = nn.Embedding(n_player, 1)
    
    def forward(self, team):
        hero_skill = self.skill(team).sum(dim=1, keepdim=True)
        return hero_skill

class NAC(nn.Module):
    def __init__(self, n_player, team_size=5, device=torch.device('cpu')):
        super(NAC, self).__init__()
        self.n_player = n_player
        self.team_size = team_size
        self.device = device
        self.BT = BT(n_player)
    
    def forward(self, data):
        data = torch.as_tensor(data, dtype=torch.long, device=self.device)
        team_A = data[:, 1:1+self.team_size]  # [batch_size, team_size]
        team_B = data[:, 1+self.team_size:]  # [batch_size, team_size]
        return torch.sigmoid(self.BT(team_A) - self.BT(team_B)).view(-1)
    
    def get_top_players(self, index_to_player_id, top_k=10):
        with torch.no_grad():
            skills = self.BT.skill.weight.squeeze()
            top_k_values, top_k_indices = torch.topk(skills, min(top_k, len(skills)))
            top_players = []
            for idx, skill in zip(top_k_indices.tolist(), top_k_values.tolist()):
                pid = index_to_player_id.get(idx, idx)  # 映射到真實 ID，否則使用索引
                top_players.append((pid, skill))
            return top_players

def evaluate(pred, label):
    pred = pred.cpu().detach().numpy().reshape(-1)
    label = np.array(label).reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    acc = (label == (pred > 0.5)).sum() / len(label)
    return auc, acc, logloss

def train_and_evaluate(dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step1'):
    model = NAC(dataset.n_individual, team_size=dataset.team_size, device=device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=4, min_lr=1e-6)
    criterion = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1
    best_val_auc = -float('inf')  # 改為 AUC，初始化為負無窮
    best_test_auc = -float('inf')
    best_metrics = {}
    best_skill_mu = None
    best_test_preds = None
    best_test_labels = None
    patience_counter = 0

    phases = ['train', 'valid', 'test'] if len(dataset.valid) > 0 else ['train', 'test']

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for i, (X, y) in enumerate(dataset.get_batch(batch_size, 'train', shuffle=False)):
            X_tensor = torch.tensor(X, dtype=torch.long, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float, device=device)
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / total_step
        logger.info(f'Trial {trial_idx}, Epoch [{epoch+1}/{n_epochs}], Phase {phase}, Average Loss: {avg_loss:.4f}')

        model.eval()
        results = {}
        for split in phases:
            preds = []
            labels = dataset.__getattribute__(split)[:, -1]
            for X, _ in dataset.get_batch(batch_size, split, shuffle=False):
                X = torch.tensor(X, dtype=torch.long, device=device)
                with torch.no_grad():
                    pred = model(X).reshape(-1)
                    preds.append(pred)
            preds = torch.cat(preds)
            auc, acc, logloss = evaluate(preds, labels)
            results[split] = {'auc': auc, 'acc': acc, 'logloss': logloss, 'preds': preds.cpu().numpy(), 'labels': labels}
            logger.info(f'Trial {trial_idx}, Epoch [{epoch+1}/{n_epochs}], Phase {phase}, {split.capitalize()}: AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')

        # 早期停止邏輯，基於 AUC
        current_auc = results['valid']['auc'] if 'valid' in phases else results['test']['auc']
        if 'valid' in phases:
            scheduler.step(results['valid']['auc'])
            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_metrics = results
                best_metrics['epoch'] = epoch + 1
                best_skill_mu = model.BT.skill.weight.squeeze().clone().detach()
                best_test_preds = results['test']['preds']
                best_test_labels = results['test']['labels']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_trial{trial_idx}_{phase}_{current_time}.pth'))
                logger.info(f'Trial {trial_idx}, Phase {phase}, New best valid AUC: {best_val_auc:.4f} at epoch {epoch+1}')
            else:
                patience_counter += 1
                logger.info(f'Trial {trial_idx}, Phase {phase}, No improvement in valid AUC, patience counter: {patience_counter}/{early_stop_patience}')
        else:
            scheduler.step(results['test']['auc'])
            if current_auc > best_test_auc:
                best_test_auc = current_auc
                best_metrics = results
                best_metrics['epoch'] = epoch + 1
                best_skill_mu = model.BT.skill.weight.squeeze().clone().detach()
                best_test_preds = results['test']['preds']
                best_test_labels = results['test']['labels']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_trial{trial_idx}_{phase}_{current_time}.pth'))
                logger.info(f'Trial {trial_idx}, Phase {phase}, New best test AUC: {best_test_auc:.4f} at epoch {epoch+1}')
            else:
                patience_counter += 1
                logger.info(f'Trial {trial_idx}, Phase {phase}, No improvement in test AUC, patience counter: {patience_counter}/{early_stop_patience}')

        if patience_counter >= early_stop_patience:
            logger.info(f'Trial {trial_idx}, Phase {phase}, early stopping triggered after {epoch+1} epochs, best epoch was {best_metrics["epoch"]}')
            break

    return (best_val_auc if 'valid' in phases else best_test_auc), best_metrics, best_skill_mu, best_test_preds, best_test_labels

def main():
    dataset = Data(path, team_size=team_size, seed=SEED)
    logger.info("Starting expanding window training with multiple trials")

    # Step 1: Initial Training and Validation
    logger.info("\n=== Step 1: Initial Training and Validation ===")
    best_auc = -float('inf')
    best_trial = 0
    best_metrics = None
    best_skill_mu = None
    trial_results = []

    for trial_idx in range(num_trials):
        logger.info(f"Running trial {trial_idx} for Step 1")
        val_auc, metrics, skill_mu, _, _ = train_and_evaluate(
            dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step1'
        )
        trial_results.append({'auc': val_auc, 'metrics': metrics, 'skill_mu': skill_mu})
        logger.info(f"Trial {trial_idx}, Step 1, Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
             best_auc = val_auc
             best_trial = trial_idx
             best_metrics = metrics
             best_skill_mu = skill_mu

            # === Export Step 1 preds & labels (in BT.py) ===
            # 如果有 validation split 就輸出 valid，否則輸出 test
             split = 'valid' if 'valid' in metrics else 'test'
             step1_out_path = os.path.join(output_dir, f'test_avg_preds_labels_step1_SGD_{current_time}.csv')
             df_step1 = pd.DataFrame({
              'avg_preds': metrics[split]['preds'],  # 改成 avg_preds
              'labels': metrics[split]['labels']
})

             df_step1.to_csv(step1_out_path, index=False)
             logger.info(f"Saved Step 1 {split} preds and labels to {step1_out_path}")


    logger.info("\n=== Step 1 Results ===")
    logger.info(f"Best trial: {best_trial}, Best validation AUC: {best_auc:.4f}")

    # Step 2: Expanding Training Set and Re-training
    logger.info("\n=== Step 2: Expanding Training Set and Re-training ===")
    dataset.expand_training_data()
    test_results = []
    all_test_preds = []
    all_test_labels = []

    for trial_idx in range(num_trials):
        logger.info(f"Running trial {trial_idx} for Step 2")
        test_auc, metrics, skill_mu, test_preds, test_labels = train_and_evaluate(
            dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step2'
        )
        test_results.append({'auc': test_auc, 'metrics': metrics, 'skill_mu': skill_mu})
        all_test_preds.append(test_preds)
        all_test_labels.append(test_labels)
        logger.info(f"Test trial {trial_idx}, Step 2, Test AUC: {test_auc:.4f}")

    # 計算所有 trial 的平均預測概率
    avg_test_preds = np.mean(all_test_preds, axis=0)
    test_labels = all_test_labels[0]

    df_avg = pd.DataFrame({'avg_preds': avg_test_preds, 'labels': test_labels})
    file_path_avg = os.path.join(output_dir, f'test_avg_preds_labels_step2_SGD_{current_time}.csv')
    df_avg.to_csv(file_path_avg, index=False)
    logger.info(f"Saved average test preds and labels to {file_path_avg}")

    logger.info("\n=== Final Results ===")
    avg_test_auc = np.mean([r['metrics']['test']['auc'] for r in test_results])
    avg_test_acc = np.mean([r['metrics']['test']['acc'] for r in test_results])
    avg_test_logloss = np.mean([r['metrics']['test']['logloss'] for r in test_results])
    logger.info(f"Test Average AUC: {avg_test_auc:.4f}, Avg Acc: {avg_test_acc:.4f}, Avg Logloss: {avg_test_logloss:.4f}")

    # 收集選手數據
    all_data = np.vstack([dataset.train, dataset.test])
    player_games = {}
    player_wins = {}
    for row in all_data:
        pids = row[2:12].astype(int)
        team_A = pids[:5]
        team_B = pids[5:]
        label = row[-1]
        for pid in pids:
            if pid == -1:
                continue
            player_games[pid] = player_games.get(pid, 0) + 1
            if (pid in team_A and label == 1) or (pid in team_B and label == 0):
                player_wins[pid] = player_wins.get(pid, 0) + 1

    # 使用 Step 2 最佳試驗的 skill_mu
    final_skill_mu = test_results[0]['skill_mu']

    # 打印前 10 名選手
    model = NAC(dataset.n_individual, team_size=dataset.team_size, device=device).to(device)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_trial0_step2_{current_time}.pth')))
    top_players = model.get_top_players(dataset.index_to_player_id, top_k=10)
    logger.info("Top 10 players statistics:")
    for pid, skill_mu in top_players:
        index = dataset.player_id_to_index.get(pid, pid)
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        logger.info(f"Player ID {pid} - Games: {games}, Wins: {wins}, Win Rate: {win_rate:.2%}, Skill mu: {skill_mu:.4f}, Sigma: N/A")

    # 整理選手數據並存為 CSV
    player_data = []
    for index in player_games.keys():
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        pid = dataset.index_to_player_id.get(index, index)
        skill_mu = final_skill_mu[index].item() if index < len(final_skill_mu) else 0.0
        player_data.append({
            'Player_ID': pid,
            'Games': games,
            'Wins': wins,
            'Win_Rate': win_rate,
            'Skill_Mu': skill_mu,
            'Skill_Sigma': 'N/A'
        })

    player_df = pd.DataFrame(player_data)
    player_csv_path = os.path.join(output_dir, f'player_stats_BT_{current_time}.csv')
    player_df.to_csv(player_csv_path, index=False)
    logger.info(f"Player statistics saved to {player_csv_path}")

    # --------- 整體統計結果 ---------
    if len(trial_results) and len(test_results):
        stage0_aucs = [r['auc'] for r in trial_results]
        stage1_aucs = [r['auc'] for r in test_results]

        s0_mu, s0_std = np.mean(stage0_aucs), np.std(stage0_aucs)
        s1_mu, s1_std = np.mean(stage1_aucs), np.std(stage1_aucs)

        logger.info("\n========== Summary across trials ==========")
        logger.info(f"Stage-0 (Step1) Val AUC : {s0_mu:.4f} ± {s0_std:.4f}")
        logger.info(f"Stage-1 (Step2) Test AUC: {s1_mu:.4f} ± {s1_std:.4f}")
        logger.info("Best model paths per trial (Step1):")
        for trial_idx in range(len(trial_results)):
            model_path = os.path.join(
                model_dir,
                f'best_model_trial{trial_idx}_step1_{current_time}.pth'
            )
            logger.info(f"  • {model_path}")
        logger.info("Best model paths per trial (Step2):")
        for trial_idx in range(len(test_results)):
            model_path = os.path.join(
                model_dir,
                f'best_model_trial{trial_idx}_step2_{current_time}.pth'
            )
            logger.info(f"  • {model_path}")

    # 收集選手數據
    all_data = np.vstack([dataset.train, dataset.test])
    player_games = {}
    player_wins = {}
    for row in all_data:
        pids = row[2:12].astype(int)
        team_A = pids[:5]
        team_B = pids[5:]
        label = row[-1]
        for pid in pids:
            if pid == -1:
                continue
            player_games[pid] = player_games.get(pid, 0) + 1
            if (pid in team_A and label == 1) or (pid in team_B and label == 0):
                player_wins[pid] = player_wins.get(pid, 0) + 1

    # 使用 Step 2 最佳試驗的 skill_mu
    final_skill_mu = test_results[0]['skill_mu']  # 假設 num_trials=1，取第一個試驗

    # 打印前 10 名選手
    model = NAC(dataset.n_individual, team_size=dataset.team_size, device=device).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_trial0_step2_{current_time}.pth')))
    top_players = model.get_top_players(dataset.index_to_player_id, top_k=10)
    logger.info("Top 10 players statistics:")
    for pid, skill_mu in top_players:
        index = dataset.player_id_to_index.get(pid, pid)  # 映射回索引
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        logger.info(f"Player ID {pid} - Games: {games}, Wins: {wins}, Win Rate: {win_rate:.2%}, Skill mu: {skill_mu:.4f}, Sigma: N/A")

    # 整理選手數據並存為 CSV
    player_data = []
    for index in player_games.keys():
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        pid = dataset.index_to_player_id.get(index, index)  # 映射到真實 ID
        skill_mu = final_skill_mu[index].item() if index < len(final_skill_mu) else 0.0
        player_data.append({
            'Player_ID': pid,
            'Games': games,
            'Wins': wins,
            'Win_Rate': win_rate,
            'Skill_Mu': skill_mu,
            'Skill_Sigma': 'N/A'
        })

    # 保存選手數據到 CSV
    player_df = pd.DataFrame(player_data)
    player_csv_path = os.path.join(output_dir, f'player_stats_{current_time}.csv')
    player_df.to_csv(player_csv_path, index=False)
    logger.info(f"Player statistics saved to {player_csv_path}")

if __name__ == "__main__":
    main()