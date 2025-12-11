import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
from data import Data
import random
import logging
from bc_fim import *
import gc
import uuid

# Set random seed
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
learning_rate = 0.0001
patience = 5
team_size = 5
path = '../data/final_data/data_2013_2024.csv'
ema_tensor_path = '../data/ema_tensor/ematensor.pt'
game_id_mapping_path = '../data/tensor/game_id_mapping.json'
num_trials = 3
num_combinations = 100

# Set up logging
log_dir = 'logs/feature'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'FiN_pairwise_0.0001p2_random_search_{timestamp}.log')

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
    if type(pred) != np.ndarray:
        pred = pred.cpu().detach().numpy()
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    pred = np.clip(pred, 0.001, 0.999)
    logloss = metrics.log_loss(label, pred)
    pred = (pred > 0.5) * 1
    acc = (label == pred).sum() / len(label)
    return auc, acc, logloss

def train_and_evaluate(player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim, dataset, 
                      game_ids_train, game_ids_valid, game_ids_test, 
                      ema_tensor_path, game_id_mapping, combo_idx, trial_idx, need_att):
    seed = SEED + combo_idx * num_trials + trial_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = FIModel(
        n_player=dataset.n_individual,
        player_dim=player_dim,
        intermediate_dim=intermediate_dim,
        dropout_rate=dropout_rate,
        mlp_hidden_dim=mlp_hidden_dim,
        team_size=team_size,
        device=device,
        ema_tensor_path=ema_tensor_path,
        game_id_mapping_path=game_id_mapping_path,
        need_att=need_att
    )
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)
    criterion = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1

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
            batch_game_ids = torch.LongTensor(game_ids_train[i * batch_size:(i + 1) * batch_size]).to(device)

            # Forward pass
            probs, _, _ = model(X_tensor)  
            loss = criterion(probs, y_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                           f'Dropout {dropout_rate}, Need Att={need_att}, '
                           f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                           f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / total_step
        logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                   f'Dropout {dropout_rate}, Need Att={need_att}, '
                   f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                   f'Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

        # Evaluation phase
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}
        game_ids_dict = {'train': game_ids_train, 'valid': game_ids_valid, 'test': game_ids_test}

        for phase in phases:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)

            for i, (X, y) in enumerate(batch_gen):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    batch_game_ids = torch.LongTensor(game_ids_dict[phase][i * batch_size:(i + 1) * batch_size]).to(device)
                    probs, _, _ = model(X_tensor)  
                    preds.append(probs)
                    labels.append(y)

            y_true = np.concatenate(labels)
            y_pred = np.concatenate(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)

            results[phase] = {
                'auc': auc,
                'acc': acc,
                'logloss': logloss
            }

            logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                       f'Dropout {dropout_rate}, Need Att={need_att}, '
                       f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                       f'Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, '
                       f'Acc: {acc:.4f}, Logloss: {logloss:.4f}')
        
        scheduler.step(results['valid']['auc'])
        # Early stopping logic
        if results['valid']['auc'] > best_valid_auc:
            best_valid_auc = results['valid']['auc']
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = results
            best_model_state = model.state_dict()
            logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                       f'Dropout {dropout_rate}, Need Att={need_att}, '
                       f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                       f'New best valid AUC: {best_valid_auc:.4f} at epoch {best_epoch}')
        else:
            patience_counter += 1
            logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                       f'Dropout {dropout_rate}, Need Att={need_att}, '
                       f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                       f'No improvement in valid AUC, patience counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            logger.info(f'Player Dim {player_dim}, Intermediate Dim {intermediate_dim}, '
                       f'Dropout {dropout_rate}, Need Att={need_att}, '
                       f'MLP Hidden {mlp_hidden_dim}, Trial {trial_idx}, '
                       f'Early stopping triggered after {epoch + 1} epochs, best epoch was {best_epoch}')
            break

    return best_valid_auc, best_metrics, best_model_state

def main():
    # Load game_id_mapping
    with open(game_id_mapping_path, 'r', encoding='utf-8') as f:
        game_id_mapping = json.load(f)

    dataset = Data(path, team_size=team_size, seed=SEED)
    index_to_player_id = dataset.index_to_player_id
    logger.info(f'Training started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80)
    logger.info(f'Dataset info: Train={len(dataset.train)}, Valid={len(dataset.valid)}, Test={len(dataset.test)}')
    logger.info(f'Total unique players: {dataset.n_individual}')

    game_ids_train = dataset.train[:, 0]
    game_ids_valid = dataset.valid[:, 0]
    game_ids_test = dataset.test[:, 0]

    best_results = []
    combo_results = {}
    combo_idx = 0

    # Random search over 100 combinations
    for combo_idx in range(num_combinations):
        # Randomly sample parameters
        player_dim = random.randint(16, 64)
        intermediate_dim = random.randint(16, 64)
        dropout_rate = random.uniform(0.1, 0.4)
        mlp_hidden_dim = random.randint(16, 64)
        need_att = random.choice([True, False])

        logger.info(f'\n=== Testing combination {combo_idx + 1}/{num_combinations}: '
                    f'Player Dim={player_dim}, Intermediate Dim={intermediate_dim}, '
                    f'Dropout={dropout_rate:.3f}, Need Att={need_att}, MLP Hidden={mlp_hidden_dim} ===')
        aucs = []
        metrics_list = []
        model_states = []

        for trial_idx in range(num_trials):
            logger.info(f'Running trial {trial_idx} for Player Dim={player_dim}, '
                        f'Intermediate Dim={intermediate_dim}, Dropout={dropout_rate:.3f}, '
                        f'Need Att={need_att}, MLP Hidden={mlp_hidden_dim}')
            val_auc, metrics, model_state = train_and_evaluate(
                player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim, 
                dataset, game_ids_train, game_ids_valid, game_ids_test,
                ema_tensor_path, game_id_mapping, combo_idx, trial_idx, need_att
            )
            aucs.append(val_auc)
            metrics_list.append(metrics)
            model_states.append(model_state)

        avg_auc = np.mean(aucs)
        combo_results[(player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim, need_att)] = {
            'avg_auc': avg_auc,
            'aucs': aucs,
            'metrics': metrics_list,
            'model_states': model_states
        }
        logger.info(f'Player Dim={player_dim}, Intermediate Dim={intermediate_dim}, '
                    f'Dropout={dropout_rate:.3f}, Need Att={need_att}, '
                    f'MLP Hidden={mlp_hidden_dim}, Average AUC: {avg_auc:.4f}, AUCs: {aucs}')

        best_results.append({
            'player_dim': player_dim,
            'intermediate_dim': intermediate_dim,
            'dropout_rate': dropout_rate,
            'mlp_hidden_dim': mlp_hidden_dim,
            'need_att': need_att,
            'avg_auc': avg_auc,
            'metrics': metrics_list,
            'best_trial_idx': np.argmax(aucs)
        })

    logger.info('\n' + '=' * 80)
    logger.info('Training completed for all hyperparameter combinations!')
    logger.info('Summary of best results per combination:')

    best_avg_auc = 0
    best_combo = None
    for result in best_results:
        logger.info(f"Player Dim: {result['player_dim']}, Intermediate Dim: {result['intermediate_dim']}, "
                   f"Dropout: {result['dropout_rate']:.3f}, Need Att: {result['need_att']}, "
                   f"MLP Hidden: {result['mlp_hidden_dim']}, Average Valid AUC: {result['avg_auc']:.4f}")
        if result['avg_auc'] > best_avg_auc:
            best_avg_auc = result['avg_auc']
            best_combo = (result['player_dim'], result['intermediate_dim'], 
                         result['dropout_rate'], result['mlp_hidden_dim'], result['need_att'])

    logger.info(f"\nBest combination: Player Dim={best_combo[0]}, Intermediate Dim={best_combo[1]}, "
                f"Dropout={best_combo[2]:.3f}, MLP Hidden={best_combo[3]}, "
                f"Need Att={best_combo[4]} with Average Valid AUC: {best_avg_auc:.4f}")
    logger.info('Detailed metrics for best combination:')

    best_result = next(r for r in best_results if (r['player_dim'], r['intermediate_dim'], 
                                                  r['dropout_rate'], r['mlp_hidden_dim'], r['need_att']) == best_combo)
    best_trial_idx = best_result['best_trial_idx']
    best_metrics = best_result['metrics'][best_trial_idx]

    for phase in ['train', 'valid', 'test']:
        metrics = best_metrics[phase]
        logger.info(f"{phase.capitalize()}: AUC={metrics['auc']:.4f}, "
                   f"Acc={metrics['acc']:.4f}, Logloss={metrics['logloss']:.4f}")

    # Load best model and print weights of blade_emb, chest_emb, and mlp_pairwise_multiplication
    best_model = FIModel(
        n_player=dataset.n_individual,
        player_dim=best_combo[0],
        intermediate_dim=best_combo[1],
        dropout_rate=best_combo[2],
        mlp_hidden_dim=best_combo[3],
        need_att=best_combo[4],
        team_size=team_size,
        device=device,
        ema_tensor_path=ema_tensor_path,
        game_id_mapping_path=game_id_mapping_path
    ).to(device)
    best_model.load_state_dict(combo_results[best_combo]['model_states'][best_trial_idx])

if __name__ == "__main__":
    main()