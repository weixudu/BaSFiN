import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import random
import logging
import os
from datetime import datetime
import pandas as pd
from data import Data

def combine(team_size=5):
    index1, index2 = [], []
    for i in range(team_size):
        for j in range(team_size):
            if i == j:
                continue  
            index1.append(i)  
            index2.append(j) 
    return index1, index2

class BT(nn.Module):
    def __init__(self, n_player):
        super(BT, self).__init__()
        assert n_player > 1 
        self.skill = nn.Embedding(n_player, 1)

    def forward(self, team):
        n_match = len(team)
        hero_skill = self.skill(team).view(n_match, -1)
        team_skill = hero_skill.sum(dim=1, keepdim=True)
        return team_skill  

class ANFM(nn.Module):
    def __init__(self, n_player, team_size, hidden_dim, need_att=True):
        super(ANFM, self).__init__()
        assert(n_player > 1 and team_size > 1)  
        self.n_player = n_player
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_player, hidden_dim)
        self.index1, self.index2 = combine(5)
        self.need_att = need_att  

        self.attenM = AttM(n_player, team_size, hidden_dim, reduce=True)
        dropout = nn.Dropout(0.2) 
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 50), nn.ReLU(), dropout,  
            nn.Linear(50, 1, bias=True), nn.ReLU(),  
        )

    def forward(self, team):
        n_match = len(team) 
        a = team[:, self.index1]  
        b = team[:, self.index2] 
        a = self.embedding(a)
        b = self.embedding(b)
        order2 = self.MLP(a * b).squeeze()  
        if self.need_att:
            normal = self.attenM(a, b, dim=2) 
            order2 = order2 * normal  
        order2 = order2.sum(dim=1, keepdim=True)
        return order2

class AttM(nn.Module):
    def __init__(self, n_player, length=5, hidden_dim=10, reduce=False):
        super(AttM, self).__init__()
        assert (n_player > 1 and length > 1)  
        self.n_player = n_player
        self.hidden_dim = hidden_dim
        self.length1 = length 
        self.length2 = length if not reduce else length - 1  
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, team1, team2, dim=2):
        assert team1.shape == team2.shape
        length1, length2 = self.length1, self.length2
        team1 = team1.view(-1, length1, length2, self.hidden_dim)
        team2 = team2.view(-1, length1, length2, self.hidden_dim)
       
        score = (self.W(team1) * team2).sum(dim=3)  
        score = F.softmax(score, dim=dim)
        return score.view(-1, length1 * length2)


class Blade_chest(nn.Module):
    def __init__(self, n_player, team_size, hidden_dim, method='inner', need_att=True):
        super(Blade_chest, self).__init__()
        assert(n_player > 1 and team_size > 1)  
        assert method in ['inner', 'dist']  
        self.team_size = team_size
        self.blade = nn.Embedding(n_player, hidden_dim) 
        self.chest = nn.Embedding(n_player, hidden_dim)  
        self.index1 = np.repeat([i for i in range(team_size)], team_size)  # [0, 0, 0, 1, 1, 1, 2, 2, 2]
        self.index2 = np.tile([i for i in range(team_size)], team_size)    # [0, 1, 2, 0, 1, 2, 0, 1, 2]
        self.attenM = AttM(n_player, team_size, hidden_dim) 
        self.method = method  
        self.need_att = need_att  
        dropout = nn.Dropout(0.2)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, 50), nn.ReLU(), dropout,
            nn.Linear(50, 1, bias=True), nn.ReLU(),
        )

    def forward(self, team_A, team_B):
        a = team_A[:, self.index1] 
        b = team_B[:, self.index2]  
        a_blade = self.blade(a) 
        b_chest = self.chest(b)  
        if self.method == 'inner':
            a_beat_b = self.inner(a_blade, b_chest)  
        else:
            a_beat_b = self.dist(a_blade, b_chest)  
        return a_beat_b  

    def inner(self, a_blade, b_chest):
        interact = a_blade * b_chest
        a_beat_b = self.MLP(interact).squeeze()  # [batch_size, 25]

        normal = self.attenM(a_blade, b_chest, dim=2) if self.need_att else 1
        a_beat_b = a_beat_b * normal
        return a_beat_b.sum(dim=1, keepdim=True)

    def dist(self, a_blade, b_chest):
        interact = (a_blade - b_chest) ** 2
        a_beat_b = self.MLP(interact).squeeze()
        normal = self.attenM(a_blade, b_chest, dim=2) if self.need_att else 1
        a_beat_b = a_beat_b * normal
        return a_beat_b.sum(dim=1, keepdim=True)

class NAC(nn.Module):
    def __init__(self, n_player, team_size=5, hidden_dim=10, need_att=True, device=torch.device('cpu')):
        super(NAC, self).__init__()
        assert(n_player > 1 and team_size > 1)  
        self.n_player = n_player
        self.team_size = team_size
        self.hidden_dim = hidden_dim
        self.device = device  
        self.BT = BT(n_player) 
        self.Coop = ANFM(n_player, team_size, hidden_dim, need_att)  
        self.Comp = Blade_chest(n_player, team_size, hidden_dim, need_att=need_att) 

    def forward(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.LongTensor(data).to(self.device)
        elif data.device != self.device:
            data = data.to(self.device)
        team_A = data[:, 1:1+self.team_size] 
        team_B = data[:, 1+self.team_size:] 
        A_coop = self.Coop(team_A) + self.BT(team_A) 
        B_coop = self.Coop(team_B) + self.BT(team_B)  
        A_comp = self.Comp(team_A, team_B)  
        B_comp = self.Comp(team_B, team_A) 
        adv = A_comp - B_comp
        probs = torch.sigmoid(A_coop - B_coop + adv).view(-1) 
        return probs



# 超參數設置，與原始代碼一致
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

n_epochs = 200
batch_size = 32
learning_rate = 0.001
path = "../data/final_data/data_2013_2024.csv"
team_size = 5
num_trials = 1
early_stop_patience = 5
device = torch.device('cpu')
hidden_dim = 50 

log_dir = 'logs/NAC'
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f'NAC_training_{current_time}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def evaluate(pred, label):
    pred = pred.cpu().detach().numpy().reshape(-1)
    label = np.array(label).reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    acc = (label == (pred > 0.5)).sum() / len(label)
    return auc, acc, logloss


def train_and_evaluate(dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step1'):
    model = NAC(dataset.n_individual, team_size=dataset.team_size, hidden_dim=hidden_dim, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=10, min_lr=1e-6)
    criterion = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1
    best_val_auc = -float('inf')
    best_test_auc = -float('inf')
    best_metrics = {}
    best_skill_mu = None
    patience_counter = 0

    phases = ['train', 'valid', 'test'] if len(dataset.valid) > 0 else ['train', 'test']

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for i, (X, y) in enumerate(dataset.get_batch(batch_size, 'train', shuffle=False)):
            X = torch.tensor(X, dtype=torch.long, device=device)
            y_tensor = torch.tensor(y, dtype=torch.float, device=device)
            loss = criterion(model(X), y_tensor)
            optimizer.zero_grad()
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
            auc, acc, logloss = evaluate(torch.cat(preds), labels)
            results[split] = {'auc': auc, 'acc': acc, 'logloss': logloss}
            logger.info(f'Trial {trial_idx}, Epoch [{epoch+1}/{n_epochs}], Phase {phase}, {split.capitalize()}: AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')

        current_auc = results['valid']['auc'] if 'valid' in phases else results['test']['auc']
        if 'valid' in phases:
            scheduler.step(results['valid']['auc'])
            if current_auc > best_val_auc:
                best_val_auc = current_auc
                best_metrics = results
                best_metrics['epoch'] = epoch + 1
                best_skill_mu = model.BT.skill.weight.squeeze().clone().detach()
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_trial{trial_idx}_{phase}_{current_time}.pth'))
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
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_trial{trial_idx}_{phase}_{current_time}.pth'))
                logger.info(f'Trial {trial_idx}, Phase {phase}, New best test AUC: {best_test_auc:.4f} at epoch {epoch+1}')
            else:
                patience_counter += 1
                logger.info(f'Trial {trial_idx}, Phase {phase}, No improvement in test AUC, patience counter: {patience_counter}/{early_stop_patience}')

        if patience_counter >= early_stop_patience:
            logger.info(f'Trial {trial_idx}, Phase {phase}, early stopping triggered after {epoch+1} epochs, best epoch was {best_metrics["epoch"]}')
            break

    return (best_val_auc if 'valid' in phases else best_test_auc), best_metrics, best_skill_mu

def main(path=path):
    dataset = Data(path, team_size=team_size, seed=SEED)
    logger.info("Starting expanding window training with multiple trials")

    logger.info("\n=== Step 1: Initial Training and Validation ===")
    best_auc = -float('inf')
    best_trial = 0
    best_metrics = None
    best_skill_mu = None
    trial_results = []
    trial_best_paths_step1 = []

    for trial_idx in range(num_trials):
        logger.info(f"Running trial {trial_idx} for Step 1")
        val_auc, metrics, skill_mu = train_and_evaluate(
            dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step1'
        )
        trial_results.append({'auc': val_auc, 'metrics': metrics, 'skill_mu': skill_mu})
        logger.info(f"Trial {trial_idx}, Step 1, Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_trial = trial_idx
            best_metrics = metrics
            best_skill_mu = skill_mu

        # 預設模型存在在 log_dir
        trial_best_paths_step1.append(
            os.path.join(log_dir, f'best_model_trial{trial_idx}_step1_{current_time}.pth')
        )

    logger.info("\n=== Step 1 Results ===")
    logger.info(f"Best trial: {best_trial}, Best validation AUC: {best_auc:.4f}")

    # Step 2: Expanding Training Set and Re-training
    logger.info("\n=== Step 2: Expanding Training Set and Re-training ===")
    dataset.expand_training_data()
    test_results = []
    trial_best_paths_step2 = []

    for trial_idx in range(num_trials):
        logger.info(f"Running trial {trial_idx} for Step 2")
        test_auc, metrics, skill_mu = train_and_evaluate(
            dataset, n_epochs, batch_size, learning_rate, device, early_stop_patience, trial_idx, phase='step2'
        )
        test_results.append({'auc': test_auc, 'metrics': metrics, 'skill_mu': skill_mu})
        logger.info(f"Test trial {trial_idx}, Step 2, Test AUC: {test_auc:.4f}")

        trial_best_paths_step2.append(
            os.path.join(log_dir, f'best_model_trial{trial_idx}_step2_{current_time}.pth')
        )

    logger.info("\n=== Final Results ===")
    avg_test_auc = np.mean([r['metrics']['test']['auc'] for r in test_results])
    avg_test_acc = np.mean([r['metrics']['test']['acc'] for r in test_results])
    avg_test_logloss = np.mean([r['metrics']['test']['logloss'] for r in test_results])
    logger.info(f"Test Average AUC: {avg_test_auc:.4f}, Avg Acc: {avg_test_acc:.4f}, Avg Logloss: {avg_test_logloss:.4f}")

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
        for path in trial_best_paths_step1:
            logger.info(f"  • {path}")

        logger.info("Best model paths per trial (Step2):")
        for path in trial_best_paths_step2:
            logger.info(f"  • {path}")

if __name__ == "__main__":
    main()