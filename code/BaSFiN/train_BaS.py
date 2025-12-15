from matplotlib.font_manager import weight_dict
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
from scipy.stats import spearmanr
import random
import logging
import os
from datetime import datetime
from BaS import NAC_BBB
from data import Data
import pandas as pd
import csv

# 追蹤的 Player IDs
TRACK_PLAYER_IDS = [513, 132, 623, 510, 582, 1254, 1209, 615, 1092, 1076]
# 設置隨機種子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# 超參數
device = torch.device('cpu')
n_epochs = 200
batch_size = 32
learning_rate =  0.00833
path = '../data/final_data/data_2013_2024.csv'
team_size = 5
prior_mu = 0
prior_sigma = 1
num_samples = 100
early_stop_patience = 5
num_trials = 1
output_dir = '../output/BaS'
model_dir = os.path.join(output_dir, 'models')  

# 設置日誌
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)  # 創建模型保存目錄
os.makedirs('logs/BBB', exist_ok=True)  # 確保日誌目錄存在
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join('logs/BBB', f'BaS_LR0.0001_{current_time}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def elbo_loss(model, X, y, kl_weight, num_samples, device):
    y_tensor = torch.tensor(y, dtype=torch.float, device=device)
    y_repeated = y_tensor.unsqueeze(0).repeat(num_samples, 1)
    prob, _ = model(X, num_samples=num_samples)
    log_likelihood = -nn.BCELoss(reduction='sum')(prob, y_repeated) / num_samples
    kl_loss = model.kl_divergence()
    return -log_likelihood + kl_weight * kl_loss

def evaluate(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    acc = (label == (pred > 0.5)).sum() / len(label)
    return auc, acc, logloss

def save_player_skills_csv(skill_mus_list, skill_sigmas_list, dataset, phase, stage, kl_weight, output_dir, current_time):

    all_skills_data = []
    
    for trial_idx, (skill_mu, skill_sigma) in enumerate(zip(skill_mus_list, skill_sigmas_list)):
        for index in range(len(skill_mu)):
            pid = dataset.index_to_player_id.get(index, f"unknown_{index}")
            mu_value = skill_mu[index].item() if index < len(skill_mu) else 0.0
            sigma_value = skill_sigma[index].item() if index < len(skill_sigma) else 0.0
            
            all_skills_data.append({
                'Phase': phase,
                'Trial': trial_idx,
                'Player_ID': pid,
                'Player_Index': index,
                'Skill_Mu': mu_value,
                'Skill_Sigma': sigma_value,
                'KL_Weight': kl_weight
            })
    
    skills_df = pd.DataFrame(all_skills_data)
    skills_csv_path = os.path.join(output_dir, f'player_skills_{phase}_kl{kl_weight}_{current_time}.csv')
    skills_df.to_csv(skills_csv_path, index=False)
    logger.info(f"Saved {phase} player skills to {skills_csv_path}")
    return skills_csv_path

def train_and_evaluate(kl_weight, dataset, n_epochs, batch_size, learning_rate, num_samples, device, early_stop_patience, idx, trial_idx,
                        track_sigma=False, track_pids=None, sigma_tracking_path=None):
    model = NAC_BBB(dataset.n_individual, team_size=dataset.team_size, device=device, prior_mu=0, prior_sigma=1).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-6)
    total_step = len(dataset.train) // batch_size + 1
    best_valid_auc = float('-inf')
    best_test_auc = float('-inf')
    best_metrics = {}
    best_results = {}
    best_skill_mu = None
    best_skill_sigma = None
    patience_counter = 0

    phases = ['train', 'valid', 'test'] if len(dataset.valid) > 0 else ['train', 'test']

    sigma_log = []  # 如果要追蹤 sigma，儲存這裡

    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, type='train', shuffle=False)
        total_loss = 0

        for i, (X, y) in enumerate(batch_gen):
            X = torch.tensor(X, dtype=torch.long, device=device)
            loss = elbo_loss(model, X, y, kl_weight=kl_weight, num_samples=num_samples, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 新增：追蹤 sigma 收斂變化
            if track_sigma:
                with torch.no_grad():
                    sigma_tensor = torch.log1p(torch.exp(model.BT.rho)).detach().cpu()
                    for pid in track_pids:
                        index = dataset.player_id_to_index.get(pid, -1)
                        if index != -1 and index < len(sigma_tensor):
                            sigma_val = sigma_tensor[index].item()
                            sigma_log.append({
                                'epoch': epoch + 1,
                                'batch': i + 1,
                                'player_id': pid,
                                'player_index': index,
                                'sigma': sigma_val
                            })

        avg_loss = total_loss / total_step
        logger.info(f'KL weight {kl_weight}, Trial {trial_idx}, Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

        model.eval()
        results = {}

        for phase in phases:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)
            for i, (X, y) in enumerate(batch_gen):
                X = torch.tensor(X, dtype=torch.long, device=device)
                with torch.no_grad():
                    prob, z = model(X, num_samples=num_samples)
                    prob_mean = torch.mean(prob, dim=0)
                    preds.append(prob_mean.cpu().numpy())
                    labels.append(y.flatten())
            preds = np.concatenate(preds)
            labels = np.concatenate(labels)
            auc, acc, logloss = evaluate(preds, labels)
            results[phase] = {'auc': auc, 'acc': acc, 'logloss': logloss, 'preds': preds, 'labels': labels}
            logger.info(f'KL weight {kl_weight}, Trial {trial_idx}, Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')

        improved = False
        if 'valid' in phases:
            scheduler.step(results['valid']['auc'])
            if results['valid']['auc'] > best_valid_auc:
                best_valid_auc = results['valid']['auc']
                improved = True
        else:
            scheduler.step(results['test']['auc'])
            if results['test']['auc'] > best_test_auc:
                best_test_auc = results['test']['auc']
                improved = True

        if improved:
            best_metrics = results
            best_results = {phase: {'preds': results[phase]['preds'], 'labels': results[phase]['labels']} for phase in phases}
            best_skill_mu = model.BT.mu.clone().detach()
            best_skill_sigma = torch.log1p(torch.exp(model.BT.rho)).clone().detach()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, f'best_model_kl{idx}_trial{trial_idx}_{current_time}.pth'))
            logger.info(f'KL weight {kl_weight}, Trial {trial_idx}, New best {"valid" if "valid" in phases else "test"} AUC at epoch {epoch + 1}')

            # 如果是最佳模型且啟用 sigma 追蹤，儲存記錄
            if track_sigma and sigma_tracking_path:
                with open(sigma_tracking_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'batch', 'player_id', 'player_index', 'sigma'])
                    writer.writeheader()
                    for row in sigma_log:
                        writer.writerow(row)
                logger.info(f"Saved sigma tracking data to {sigma_tracking_path}")
        else:
            patience_counter += 1
            logger.info(f'KL weight {kl_weight}, Trial {trial_idx}, No improvement, patience: {patience_counter}/{early_stop_patience}')

        if patience_counter >= early_stop_patience:
            logger.info(f'KL weight {kl_weight}, Trial {trial_idx}, Early stopping at epoch {epoch + 1}')
            break

    return (best_valid_auc if 'valid' in phases else best_test_auc, 
            best_metrics['test']['logloss'], 
            best_metrics, 
            best_skill_mu, 
            best_skill_sigma, 
            best_results['test']['preds'], 
            best_results['test']['labels'])


def main():
    dataset = Data(path, team_size=team_size, seed=SEED)
    logger.info("Starting Expanding Window training with multiple trials")

    kl_weight_candidates = [0.05191]
    logger.info(f"KL weight candidates: {[round(x, 6) for x in kl_weight_candidates]}")

    # Step 1: Initial Training and Validation
    logger.info("\n=== Step 1: Initial Training and Validation ===")
    best_kl_weight = None
    best_kl_avg_auc = float('-inf')
    best_kl_avg_logloss = float('inf')
    best_kl_results = None
    best_kl_skill_mu = None
    best_kl_skill_sigma = None
    kl_results = {}
    
    # 儲存第一階段所有KL權重的技能值
    stage1_all_skills = {}

    for idx, kl_weight in enumerate(kl_weight_candidates):
        logger.info(f"\n=== Testing KL weight: {kl_weight} (index {idx}) ===")
        aucs = []
        loglosses = []
        results_list = []
        skill_mus = []
        skill_sigmas = []

        for trial_idx in range(num_trials):
            logger.info(f"Running trial {trial_idx} for KL weight {kl_weight}")
            val_auc, val_logloss, best_metrics, skill_mu, skill_sigma, _, _ = train_and_evaluate(
                kl_weight, dataset, n_epochs, batch_size, learning_rate, num_samples, device, early_stop_patience, idx, trial_idx
            )
            aucs.append(val_auc)
            loglosses.append(val_logloss)
            results_list.append(best_metrics)
            skill_mus.append(skill_mu)
            skill_sigmas.append(skill_sigma)

        # 保存第一階段的技能值
        stage1_skills_path = save_player_skills_csv(
            skill_mus, skill_sigmas, dataset, 'stage1', f'kl_weight_{kl_weight}', 
            kl_weight, output_dir, current_time
        )
        stage1_all_skills[kl_weight] = stage1_skills_path

        avg_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        avg_logloss = np.mean(loglosses)
        kl_results[kl_weight] = {
            'avg_auc': avg_auc,
            'avg_logloss': avg_logloss,
            'aucs': aucs,
            'loglosses': loglosses,
            'results': results_list,
            'skill_mus': skill_mus,
            'skill_sigmas': skill_sigmas
        }
        logger.info(f"KL weight {kl_weight}, Average AUC: {avg_auc:.4f}, Average Logloss: {avg_logloss:.4f}, AUCs: {aucs}")

        if avg_auc > best_kl_avg_auc:
         best_kl_avg_auc = avg_auc
         best_kl_avg_logloss = avg_logloss
         best_kl_weight = kl_weight
         best_kl_results = kl_results[kl_weight]
         best_kl_skill_mu = skill_mus[0]
         best_kl_skill_sigma = skill_sigmas[0]

        # === 匯出 Step 1 標籤 ===
        # 注意：results_list 最後一次跑的 best_metrics 已含 preds 和 labels
        step1_labels = best_metrics['valid']['labels'] if 'valid' in best_metrics else best_metrics['test']['labels']
        step1_preds = best_metrics['valid']['preds'] if 'valid' in best_metrics else best_metrics['test']['preds']

        step1_labels_path = os.path.join(output_dir, f'step1_preds_labels_kl{kl_weight}_0.001{current_time}.csv')
        df_step1 = pd.DataFrame({'avg_preds': step1_preds, 'labels': step1_labels})
        df_step1.to_csv(step1_labels_path, index=False)
        logger.info(f"Saved Step 1 preds and labels to {step1_labels_path}")


    logger.info("\n=== Step 1 Results ===")
    logger.info(f"Best KL weight: {best_kl_weight}")
    logger.info(f"Best average validation AUC: {best_kl_avg_auc:.4f} ± {std_auc:.4f}, Average Logloss: {best_kl_avg_logloss:.4f}")

    # 統計所有比賽的玩家出場次數與勝場數
    all_data_step1 = np.vstack([dataset.train, dataset.valid, dataset.test])
    player_games_step1 = {}
    player_wins_step1 = {}
    for row in all_data_step1:
        pids = row[2:12].astype(int)
        team_A = pids[:5]
        team_B = pids[5:]
        label = row[-1]
        for pid in pids:
            if pid == -1:
                continue
            player_games_step1[pid] = player_games_step1.get(pid, 0) + 1
            if (pid in team_A and label == 1) or (pid in team_B and label == 0):
                player_wins_step1[pid] = player_wins_step1.get(pid, 0) + 1

    skill_mu_np = best_kl_skill_mu.cpu().numpy()
    skill_sigma_np = best_kl_skill_sigma.cpu().numpy()

    logger.info("\n=== Step 1: Top 10 Players by Skill (mu) ===")
    skill_mu_np = best_kl_skill_mu.cpu().numpy()
    skill_sigma_np = best_kl_skill_sigma.cpu().numpy()
    top_indices = np.argsort(-skill_mu_np)[:10]

    for index in top_indices:
        pid = dataset.index_to_player_id.get(index, f"unknown_{index}")
        mu_val = skill_mu_np[index]
        sigma_val = skill_sigma_np[index]
        games = player_games_step1.get(pid, 0)
        wins  = player_wins_step1.get(pid, 0)
        win_rate = (wins / games) if games > 0 else 0.0
        logger.info(f"Player ID {pid} - Games: {games}, Wins: {wins}, "
                    f"Win Rate: {win_rate:.2%}, Skill mu: {mu_val:.4f}, Sigma: {sigma_val:.4f}")


    # Step 2: Expanding Training Set and Re-training
    logger.info("\n=== Step 2: Expanding Training Set and Re-training ===")
    dataset.expand_training_data()

    logger.info(f"Re-training with best KL weight {best_kl_weight}")
    aucs = []
    loglosses = []
    results_list = []
    skill_mus = []
    skill_sigmas = []
    all_test_preds = []
    all_test_labels = []

    for trial_idx in range(num_trials):
        logger.info(f"Running trial {trial_idx} for expanded training set")
        
        # 是否啟用 sigma 追蹤（僅限 trial 0）
        track_sigma = (trial_idx == 0)
        sigma_tracking_path = os.path.join(output_dir, f'sigma_tracking_stage2_best_trial_{current_time}.csv') if track_sigma else None

        test_auc, test_logloss, best_metrics, skill_mu, skill_sigma, test_preds, test_labels = train_and_evaluate(
            best_kl_weight, dataset, n_epochs, batch_size, learning_rate, num_samples, device, 
            early_stop_patience, 0, trial_idx,
            track_sigma=track_sigma,
            track_pids=TRACK_PLAYER_IDS,
            sigma_tracking_path=sigma_tracking_path
        )

        aucs.append(test_auc)
        loglosses.append(test_logloss)
        results_list.append(best_metrics)
        skill_mus.append(skill_mu)
        skill_sigmas.append(skill_sigma)
        all_test_preds.append(test_preds)
        all_test_labels.append(test_labels)

    # 保存第二階段的技能值
    stage2_skills_path = save_player_skills_csv(
        skill_mus, skill_sigmas, dataset, 'stage2', 'expanded_training', 
        best_kl_weight, output_dir, current_time
    )

    # 計算所有 trial 的平均預測概率
    avg_test_preds = np.mean(all_test_preds, axis=0)
    test_labels = all_test_labels[0]  # 標籤在所有 trial 中是相同的

    # 保存平均預測概率和標籤到 CSV
    df_avg = pd.DataFrame({'avg_preds': avg_test_preds, 'labels': test_labels})
    file_path_avg = os.path.join(output_dir, f'test_avg_preds_labels_kl{best_kl_weight}_{current_time}.csv')
    df_avg.to_csv(file_path_avg, index=False)
    logger.info(f"Saved average test preds and labels to {file_path_avg}")

    avg_auc = np.mean(aucs)
    avg_logloss = np.mean(loglosses)
    logger.info(f"\n=== Step 2 Results ===")
    logger.info(f"Average Test AUC: {avg_auc:.4f}, Average Test Logloss: {avg_logloss:.4f}")

    # 創建技能值摘要
    logger.info("\n=== Creating Skills Summary ===")
    
    # 合併所有階段的技能值數據
    all_skills_data = []
    
    # 讀取第一階段數據
    for kl_weight, csv_path in stage1_all_skills.items():
        if os.path.exists(csv_path):
            stage1_df = pd.read_csv(csv_path)
            all_skills_data.append(stage1_df)
    
    # 讀取第二階段數據
    if os.path.exists(stage2_skills_path):
        stage2_df = pd.read_csv(stage2_skills_path)
        all_skills_data.append(stage2_df)
    
    # 合併所有數據
    if all_skills_data:
        combined_skills_df = pd.concat(all_skills_data, ignore_index=True)
        combined_skills_path = os.path.join(output_dir, f'all_player_skills_combined_{current_time}.csv')
        combined_skills_df.to_csv(combined_skills_path, index=False)
        logger.info(f"Saved combined player skills to {combined_skills_path}")

    # Final results analysis
    logger.info("\n=== Final Results ===")
    for phase in ['train', 'test']:
        avg_auc = np.mean([r[phase]['auc'] for r in results_list])
        avg_acc = np.mean([r[phase]['acc'] for r in results_list])
        avg_logloss = np.mean([r[phase]['logloss'] for r in results_list])
        avg_auc_std = np.std([r[phase]['auc'] for r in results_list])
        logger.info(f"{phase.capitalize()} Average AUC: {avg_auc:.4f} ± {avg_auc_std:.4f}, Avg Acc: {avg_acc:.4f}, Avg Logloss: {avg_logloss:.4f}")
    skill_mean = skill_mus[0].mean().item()
    skill_std = skill_sigmas[0].mean().item()
    logger.info(f"All players skill mean (mu): {skill_mean:.4f}, average uncertainty (sigma): {skill_std:.4f}")

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

    # 整理選手數據並存為 CSV
    player_data = []
    for index in player_games.keys():
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        pid = dataset.index_to_player_id.get(index, None)
        if pid is None:
            logger.warning(f"Index {index} not in index_to_player_id")
            continue
        skill_mu = skill_mus[0][index].item() if index < len(skill_mus[0]) else 0.0
        skill_sigma = skill_sigmas[0][index].item() if index < len(skill_sigmas[0]) else 0.0
        player_data.append({
            'Player_ID': pid,
            'Games': games,
            'Wins': wins,
            'Win_Rate': win_rate,
            'Skill_Mu': skill_mu,
            'Skill_Sigma': skill_sigma
        })

    player_df = pd.DataFrame(player_data)
    player_csv_path = os.path.join(output_dir, f'player_stats_{current_time}.csv')
    player_df.to_csv(player_csv_path, index=False)
    logger.info(f"Player statistics saved to {player_csv_path}")

    logger.info(f"Player IDs in player_games: min={min(player_games.keys())}, max={max(player_games.keys())}")
    top_players = NAC_BBB(dataset.n_individual, team_size=dataset.team_size, device=device, prior_mu=0, prior_sigma=1).to(device)
    top_players.load_state_dict(torch.load(os.path.join(model_dir, f'best_model_kl{kl_weight_candidates.index(best_kl_weight)}_trial0_{current_time}.pth')))
    top_players = top_players.get_top_players(dataset.index_to_player_id, top_k=10)
    logger.info("Top 10 players statistics:")
    for pid, skill_mu in top_players:
        index = dataset.player_id_to_index.get(pid, -1)
        if index == -1:
            logger.warning(f"Player ID {pid} not in player_id_to_index")
            continue
        games = player_games.get(index, 0)
        wins = player_wins.get(index, 0)
        win_rate = wins / games if games > 0 else 0.0
        skill_sigma = skill_sigmas[0][index].item()
        logger.info(f"Player ID {pid} - Games: {games}, Wins: {wins}, Win Rate: {win_rate:.2%}, Skill mu: {skill_mu:.4f}, Sigma: {skill_sigma:.4f}")

    valid_players = []
    for index, games in player_games.items():
        wins = player_wins.get(index, 0)
        if games > 0:
            pid = dataset.index_to_player_id.get(index, None)
            if pid is None:
                logger.warning(f"Index {index} not in index_to_player_id")
                continue
            skill_mu = skill_mus[0][index].item()
            valid_players.append((pid, wins / games, skill_mu))

    if len(valid_players) > 1:
        corr, p_value = spearmanr([x[1] for x in valid_players], [x[2] for x in valid_players])
        logger.info(f"Spearman's Rank Correlation Coefficient between win rate and skill mu: {corr:.4f}, p-value: {p_value:.4f}")
    else:
        logger.info("Insufficient valid players for Spearman's correlation")
    
    # 技能值文件摘要
    logger.info("\n=== Skills Files Summary ===")
    logger.info("Generated skill files:")
    for kl_weight, csv_path in stage1_all_skills.items():
        logger.info(f"Stage 1 (KL={kl_weight}): {csv_path}")
    logger.info(f"Stage 2: {stage2_skills_path}")
    logger.info(f"Combined: {combined_skills_path}")

if __name__ == "__main__":
    main()
