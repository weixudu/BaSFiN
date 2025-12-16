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
from co_fim import NAC_ANFM
from BaS import NAC_BBB
from bc_fim import FIModel

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# Configuration
device = torch.device('cpu')
n_epochs = 200
batch_size = 32
learning_rate = 0.001
team_size = 5
patience = 5
path = '../data/final_data/data_2013_2024.csv'
ema_tensor_path = '../data/ema_tensor/ematensor.pt'
game_id_mapping_path = '../data/tensor/game_id_mapping.json'
model_save_dir = 'model/pretrain_BaSFiN_model'

# Best hyperparameters
anfm_player_dim = 31
anfm_hidden_dim = 55
anfm_need = False
anfm_drop = 0.245
anfm_mlplayer = 35

kl_weight = 0.01519
num_samples = 100

bc_player_dim = 54
bc_intermediate_dim = 20
bc_drop = 0.274
bc_mlplayer = 38
bc_need = False


# Setup logging
log_dir = 'logs/pretrain_FIM'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'pretrain_0.0001p2_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def evaluate(pred, label):
    if not isinstance(pred, np.ndarray):
        pred = pred.cpu().detach().numpy()
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss

def elbo_loss(model, X, y, kl_weight, num_samples, device=torch.device('cuda:0')):
    y_tensor = torch.tensor(y, dtype=torch.float, device=device)  
    y_repeated = y_tensor.unsqueeze(0).repeat(num_samples, 1)
    prob, _ = model(X, num_samples=num_samples)
    log_likelihood = -nn.BCELoss(reduction='sum')(prob, y_repeated) / num_samples
    kl_loss = model.kl_divergence()
    return -log_likelihood + kl_weight * kl_loss

def train_anfm(dataset, criterion, total_step):
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

    optimizer = optim.SGD(model.parameters(), lr=learning_rate*0.1, weight_decay=0.005, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)

    best_valid_auc = 0.0
    patience_counter = 0
    early_stopped = False
    best_metrics = None
    
    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, shuffle=False)
        total_loss = 0
        
        for i, (X, y) in enumerate(batch_gen):
            X_tensor = torch.LongTensor(X).to(device)
            y_tensor = torch.Tensor(y).to(device)
            pred, _,_ = model(X_tensor)
            loss = criterion(pred, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                logger.info(f'ANFM Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / total_step
        logger.info(f'ANFM Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')
        
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}
        
        for phase in phases:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)
            
            for X, y in batch_gen:
                X_tensor = torch.LongTensor(X).to(device)
                with torch.no_grad():
                    pred, _,_ = model(X_tensor)
                    preds.append(pred)
                    labels.append(y)
            
            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)
            results[phase] = {'auc': auc, 'acc': acc, 'logloss': logloss}
            logger.info(f'ANFM Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')
        
        scheduler.step(results['valid']['auc'])

        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_metrics = results
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'anfm.pth'))
            logger.info(f'ANFM New best valid AUC: {best_valid_auc:.4f} at epoch {epoch + 1}')
        else:
            patience_counter += 1
            logger.info(f'ANFM No improvement, patience counter: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            early_stopped = True
            logger.info(f'ANFM Early stopping triggered after {epoch + 1} epochs')
            break
    
    return model, best_valid_auc, best_metrics, early_stopped

def train_nac_bbb(dataset, total_step):
    model = NAC_BBB(
        n_player=dataset.n_individual,
        team_size=team_size,
        device=device,
        prior_mu=0,
        prior_sigma=1
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)
    
    best_valid_auc = float('-inf')
    patience_counter = 0
    early_stopped = False
    best_metrics = None
    phases = ['train', 'valid', 'test']

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

            if i == 0 and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                logger.info(f"GPU Memory: Allocated = {allocated:.2f} MB, Reserved = {reserved:.2f} MB")

        avg_loss = total_loss / total_step
        logger.info(f'KL weight {kl_weight}, Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

        model.eval()
        results = {}

        for phase in phases:
            preds = []
            zs = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)
            for i, (X, y) in enumerate(batch_gen):
                X = torch.tensor(X, dtype=torch.long, device=device)
                with torch.no_grad():
                    prob, z = model(X, num_samples=num_samples)
                    prob_mean = torch.mean(prob, dim=0)
                    z_mean = torch.mean(z, dim=0)
                    preds.append(prob_mean.cpu().numpy())
                    zs.append(z_mean.cpu().numpy())
                    labels.append(y.flatten())
            y_true = np.concatenate(labels)
            y_pred = np.concatenate(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)
            results[phase] = {'auc': auc, 'acc': acc, 'logloss': logloss}
            logger.info(f'KL weight {kl_weight}, Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')

        scheduler.step(results['valid']['auc'])

        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_metrics = results
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'nac_bbb.pth'))
            logger.info(f'NAC_BBB New best valid AUC: {best_valid_auc:.4f} at epoch {epoch + 1}')
        else:
            patience_counter += 1
            logger.info(f'NAC_BBB No improvement, patience counter: {patience_counter}/{patience}')
        
        if patience_counter >= patience:
            early_stopped = True
            logger.info(f'NAC_BBB Early stopping triggered after {epoch + 1} epochs')
            break

    return model, best_valid_auc, best_metrics, early_stopped

def train_fimodel(dataset, criterion, total_step, game_id_mapping, game_ids_dict):
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
        need_att= bc_need
    )

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate*0.1, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)
    
    best_valid_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_metrics = None
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, shuffle=False)
        total_loss = 0

        for i, (X, y) in enumerate(batch_gen):
            y_tensor = torch.Tensor(y).to(device)
            X_tensor = torch.LongTensor(X).to(device)
            # 檢查 game_ids_dict['train'] 是否可迭代
            if not isinstance(game_ids_dict['train'], (list, np.ndarray, torch.Tensor)):
                raise ValueError(f"game_ids_dict['train'] is not iterable, got {type(game_ids_dict['train'])}")
            batch_game_ids = torch.LongTensor(game_ids_dict['train'][i * batch_size:(i + 1) * batch_size]).to(device)

            # 前向傳播
            pred, _,_ = model(X_tensor)
            loss = criterion(pred, y_tensor)

            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                logger.info(f'Intermediate Dim {bc_intermediate_dim}, '
                            f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / total_step
        logger.info(f'Intermediate Dim {bc_intermediate_dim},  '
                    f'Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

        # 評估階段
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}

        for phase in phases:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)

            for i, (X, y) in enumerate(batch_gen):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    # 檢查 game_ids_dict[phase] 是否可迭代
                    if not isinstance(game_ids_dict[phase], (list, np.ndarray, torch.Tensor)):
                        raise ValueError(f"game_ids_dict['{phase}'] is not iterable, got {type(game_ids_dict[phase])}")
                    batch_game_ids = torch.LongTensor(game_ids_dict[phase][i * batch_size:(i + 1) * batch_size]).to(device)
                    pred, _,_ = model(X_tensor)
                    preds.append(pred)
                    labels.append(y)

            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)

            results[phase] = {
                'auc': auc,
                'acc': acc,
                'logloss': logloss
            }

            logger.info(f'Intermediate Dim {bc_intermediate_dim}, '
                        f'Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')
        
        scheduler.step(results['valid']['auc'])
        # 早停邏輯
        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = results
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'fimodel.pth'))
            logger.info(f'Intermediate Dim {bc_intermediate_dim}, '
                        f'New best valid AUC: {best_valid_auc:.4f} at epoch {best_epoch}')
        else:
            patience_counter += 1
            logger.info(f'Intermediate Dim {bc_intermediate_dim},  '
                        f'No improvement in valid AUC, patience counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            early_stopped = True
            logger.info(f'Intermediate Dim {bc_intermediate_dim}, '
                        f'Early stopping triggered after {epoch + 1} epochs, best epoch was {best_epoch}')
            break
    
    return model, best_valid_auc, best_metrics, early_stopped

def main():
    # Load data
    dataset = Data(path, team_size=team_size, seed=SEED)
    with open(game_id_mapping_path, 'r', encoding='utf-8') as f:
        game_id_mapping = json.load(f)
    
    game_ids_train = dataset.train[:, 0]
    game_ids_valid = dataset.valid[:, 0]
    game_ids_test = dataset.test[:, 0]
    game_ids_dict = {'train': game_ids_train, 'valid': game_ids_valid, 'test': game_ids_test}
    
    logger.info(f'Training started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80)
    logger.info(f'Dataset info: Train={len(dataset.train)}, Valid={len(dataset.valid)}, Test={len(dataset.test)}')
    logger.info(f'Total unique players: {dataset.n_individual}')
    logger.info(f"Train player IDs: min={dataset.train[:, 1:11].min()}, max={dataset.train[:, 1:11].max()}")
    logger.info(f"Valid player IDs: min={dataset.valid[:, 1:11].min()}, max={dataset.valid[:, 1:11].max()}")
    logger.info(f"Train label ratio: {np.mean(dataset.train[:, -1]):.4f}")
    logger.info(f"Valid label ratio: {np.mean(dataset.valid[:, -1]):.4f}")
    
    criterion = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1
    
    # Create model save directory
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Train all models
    anfm_model, anfm_auc, anfm_metrics, anfm_early_stopped = train_anfm(dataset, criterion, total_step)
    logger.info(f'ANFM Valid AUC: {anfm_auc:.4f}')
    
    # Reset random seed and dataset
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dataset = Data(path, team_size=team_size, seed=SEED)
    
    nac_bbb_model, bbb_auc, bbb_metrics, bbb_early_stopped = train_nac_bbb(dataset, total_step)
    logger.info(f'NAC_BBB Valid AUC: {bbb_auc:.4f}')
    
    # Reset random seed and dataset
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dataset = Data(path, team_size=team_size, seed=SEED)
    
    fimodel, fim_auc, fim_metrics, fim_early_stopped = train_fimodel(
        dataset, criterion, total_step, game_id_mapping, game_ids_dict
    )
    logger.info(f'FIModel Valid AUC: {fim_auc:.4f}')
    
    # Save freeze configuration
    freeze_dict = {
        'nac_anfm': anfm_early_stopped,
        'nac_bbb': bbb_early_stopped,
        'fimodel': fim_early_stopped
    }
    logger.info(f'Freeze recommendations: {freeze_dict}')
    
    freeze_save_path = os.path.join(model_save_dir, 'freeze_config.json')
    with open(freeze_save_path, 'w') as f:
        json.dump(freeze_dict, f, indent=4)
    logger.info(f'Saved freeze recommendations to {freeze_save_path}')
    
    # Log final results
    logger.info('\n' + '=' * 80)
    logger.info('Training completed for all models!')
    logger.info('Summary of best results:')
    
    logger.info(f'ANFM (hidden_dim={anfm_hidden_dim}):')
    for phase in ['train', 'valid', 'test']:
        logger.info(f"{phase.capitalize()}: AUC={anfm_metrics[phase]['auc']:.4f}, Acc={anfm_metrics[phase]['acc']:.4f}, Logloss={anfm_metrics[phase]['logloss']:.4f}")
    
    logger.info(f'NAC_BBB (kl_weight={kl_weight}):')
    for phase in ['train', 'valid', 'test']:
        logger.info(f"{phase.capitalize()}: AUC={bbb_metrics[phase]['auc']:.4f}, Acc={bbb_metrics[phase]['acc']:.4f}, Logloss={bbb_metrics[phase]['logloss']:.4f}")
    
    logger.info(f'FIModel (intermediate_dim={bc_intermediate_dim}:')
    for phase in ['train', 'valid', 'test']:
        logger.info(f"{phase.capitalize()}: AUC={fim_metrics[phase]['auc']:.4f}, Acc={fim_metrics[phase]['acc']:.4f}, Logloss={fim_metrics[phase]['logloss']:.4f}")
    
    logger.info(f'Training completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    main()
