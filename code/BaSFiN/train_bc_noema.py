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
from bc_fim_noema import *  # 改成 noema 版本
import gc
import uuid
import pandas as pd


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
player_dims = [54]
intermediate_dims = [20]
dropout_rates = [0.274]
mlp_hidden_dims = [38]
need_atts = [False]  


patience = 5
team_size = 5
path = '../data/final_data/data_2013_2024.csv'
# 移除 EMA 相關路徑
# ema_tensor_path = '../data/ema_tensor/ematensor.pt'
# game_id_mapping_path = '../data/tensor/game_id_mapping.json'
num_trials = 5
output_dir = '../output/FIM_noema'  # 改名成 noema
model_dir = os.path.join(output_dir, 'models')


# Set up logging
log_dir = 'logs/feature_noema'  # 改名
os.makedirs(log_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'FiN_bc_noema_ReLU10_{timestamp}.log')  # 改名


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
                      combo_idx, trial_idx, need_att, phase='step1'):
    # 移除 ema_tensor_path, game_id_mapping 參數
    seed = SEED + trial_idx
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
        need_att=need_att
        # 移除 ema_tensor_path 和 game_id_mapping_path
    )
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)
    criterion = nn.BCELoss()
    total_step = len(dataset.train) // batch_size + 1


    best_valid_auc = 0.0
    patience_counter = 0
    best_epoch = 0
    best_metrics = None
    best_model_state = None
    best_test_preds = None
    best_test_labels = None


    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, shuffle=False)
        total_loss = 0


        for i, (X, y) in enumerate(batch_gen):
            y_tensor = torch.Tensor(y).to(device)
            X_tensor = torch.LongTensor(X).to(device)
            # 移除 batch_game_ids，因為 noema 版本不需要


            probs, _, _ = model(X_tensor) 
            loss = criterion(probs, y_tensor)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            total_loss += loss.item()


            if (i + 1) % 10 == 0:
                logger.info(f'Phase {phase}, Trial {trial_idx}, '
                           f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')


        avg_loss = total_loss / total_step
        logger.info(f'Phase {phase}, Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.4f}')


        model.eval()
        parts = ['train', 'valid', 'test'] if phase == 'step1' else ['train', 'test']
        results = {}
        game_ids_dict = {'train': game_ids_train, 'valid': game_ids_valid, 'test': game_ids_test}


        for part in parts:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, part, shuffle=False)


            for i, (X, y) in enumerate(batch_gen):
                with torch.no_grad():
                    X_tensor = torch.LongTensor(X).to(device)
                    # 移除 batch_game_ids
                    probs, _, _ = model(X_tensor)
                    preds.append(probs)
                    labels.append(y)


            y_true = np.concatenate(labels)
            y_pred = torch.cat(preds).cpu().numpy()
            auc, acc, logloss = evaluate(y_pred, y_true)


            results[part] = {
                'auc': auc,
                'acc': acc,
                'logloss': logloss
            }


            if part == 'test' and (phase == 'step2' or results.get('valid', {}).get('auc', 0) > best_valid_auc):
                best_test_preds = y_pred
                best_test_labels = y_true


            logger.info(f'Phase {phase}, Trial {trial_idx}, '
                       f'Epoch [{epoch+1}/{n_epochs}], {part.capitalize()} AUC: {auc:.4f}, '
                       f'Acc: {acc:.4f}, Logloss: {logloss:.4f}')


        current_auc = results.get('valid', results['test'])['auc']
        scheduler.step(current_auc)


        if current_auc > best_valid_auc:
            best_valid_auc = current_auc
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = results
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(model_dir, f'best_model_combo{combo_idx}_trial{trial_idx}_{phase}_{timestamp}.pth'))
            logger.info(f'Phase {phase}, Trial {trial_idx}, '
                       f'New best AUC: {best_valid_auc:.4f} at epoch {best_epoch}, '
                       f'Model saved to {os.path.join(model_dir, f"best_model_combo{combo_idx}_trial{trial_idx}_{phase}_{timestamp}.pth")}')
        else:
            patience_counter += 1
            logger.info(f'Phase {phase}, Trial {trial_idx}, '
                       f'No improvement in AUC, patience counter: {patience_counter}/{patience}')


        if patience_counter >= patience:
            logger.info(f'Phase {phase}, Trial {trial_idx}, '
                       f'Early stopping triggered after {epoch+1} epochs, best epoch was {best_epoch}')
            break


    return best_valid_auc, best_metrics, best_model_state, best_test_preds, best_test_labels


def main():
    # 移除 game_id_mapping 載入
    # with open(game_id_mapping_path, 'r', encoding='utf-8') as f:
    #     game_id_mapping = json.load(f)


    dataset = Data(path, team_size=team_size, seed=SEED)
    index_to_player_id = dataset.index_to_player_id
    logger.info(f'Training started at {time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80)
    logger.info(f'Dataset info: Train={len(dataset.train)}, Valid={len(dataset.valid)}, Test={len(dataset.test)}')
    logger.info(f'Total unique players: {dataset.n_individual}')


    game_ids_train = dataset.train[:, 0]
    game_ids_valid = dataset.valid[:, 0]
    game_ids_test = dataset.test[:, 0]


    # Step 1: Initial Training and Validation
    logger.info("\n=== Step 1: Initial Training and Validation (NO-EMA) ===")
    best_results = []
    combo_results = {}
    combo_idx = 0


    for player_dim in player_dims:
        for intermediate_dim in intermediate_dims:
            for dropout_rate in dropout_rates:
                for mlp_hidden_dim in mlp_hidden_dims:
                    for need_att in need_atts:
                        logger.info(f'\n=== Testing combination: Player Dim={player_dim}, '
                                    f'Intermediate Dim={intermediate_dim}, '
                                    f'Dropout={dropout_rate}, MLP Hidden={mlp_hidden_dim}, '
                                    f'Need Att={need_att} (combo_idx={combo_idx}) ===')
                        aucs = []
                        metrics_list = []
                        model_states = []
                        all_test_preds = []
                        all_test_labels = []


                        for trial_idx in range(num_trials):
                            logger.info(f'Running trial {trial_idx} for Player Dim={player_dim}, '
                                        f'Intermediate Dim={intermediate_dim}, '
                                        f'Dropout={dropout_rate}, MLP Hidden={mlp_hidden_dim}, '
                                        f'Need Att={need_att}')
                            val_auc, metrics, model_state, test_preds, test_labels = train_and_evaluate(
                                player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim, 
                                dataset, game_ids_train, game_ids_valid, game_ids_test,
                                combo_idx, trial_idx, need_att, phase='step1'
                                # 移除 ema_tensor_path, game_id_mapping
                            )
                            aucs.append(val_auc)
                            metrics_list.append(metrics)
                            model_states.append(model_state)
                            all_test_preds.append(test_preds)
                            all_test_labels.append(test_labels)


                        avg_test_preds = np.mean(all_test_preds, axis=0)
                        test_labels = all_test_labels[0]
                        df_avg = pd.DataFrame({'avg_preds': avg_test_preds, 'labels': test_labels})
                        file_path_avg = os.path.join(output_dir, f'test_avg_preds_labels_combo{combo_idx}_step1_{timestamp}.csv')
                        df_avg.to_csv(file_path_avg, index=False)
                        logger.info(f'Saved average test preds and labels for combo {combo_idx} (Step 1) to {file_path_avg}')


                        avg_auc = np.mean(aucs)
                        combo_results[(player_dim, intermediate_dim, dropout_rate, mlp_hidden_dim, need_att)] = {
                            'avg_auc': avg_auc,
                            'aucs': aucs,
                            'metrics': metrics_list,
                            'model_states': model_states
                        }
                        logger.info(f'Player Dim={player_dim}, Intermediate Dim={intermediate_dim}, '
                                    f'Dropout={dropout_rate}, MLP Hidden={mlp_hidden_dim}, '
                                    f'Need Att={need_att}, Average AUC: {avg_auc:.4f}, AUCs: {aucs}')


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
                        combo_idx += 1


    best_avg_auc = 0
    best_combo = None
    for result in best_results:
        logger.info(f"Player Dim: {result['player_dim']}, Intermediate Dim: {result['intermediate_dim']}, "
                   f"Dropout: {result['dropout_rate']}, MLP Hidden: {result['mlp_hidden_dim']}, "
                   f"Need Att: {result['need_att']}, Average Valid AUC: {result['avg_auc']:.4f}")
        if result['avg_auc'] > best_avg_auc:
            best_avg_auc = result['avg_auc']
            best_combo = (result['player_dim'], result['intermediate_dim'], 
                         result['dropout_rate'], result['mlp_hidden_dim'], result['need_att'])


    logger.info(f"\nBest combination: Player Dim={best_combo[0]}, Intermediate Dim={best_combo[1]}, "
               f"Dropout={best_combo[2]}, MLP Hidden={best_combo[3]}, Need Att={best_combo[4]} "
               f"with Average Valid AUC: {best_avg_auc:.4f}")


    # Step 2: Expanding Training Set and Re-training
    logger.info("\n=== Step 2: Expanding Training Set and Re-training (NO-EMA) ===")
    dataset.expand_training_data()  # Merge training and validation sets
    game_ids_train = np.concatenate([game_ids_train, game_ids_valid])
    game_ids_valid = np.array([])  # Clear validation set


    logger.info(f'After expansion: Train={len(dataset.train)}, Valid={len(dataset.valid)}, Test={len(dataset.test)}')
    
    test_results = []
    all_test_preds = []
    all_test_labels = []


    for trial_idx in range(num_trials):
        logger.info(f'Running trial {trial_idx} for best combination in Step 2')
        test_auc, metrics, model_state, test_preds, test_labels = train_and_evaluate(
            best_combo[0], best_combo[1], best_combo[2], best_combo[3],
            dataset, game_ids_train, game_ids_valid, game_ids_test,
            combo_idx=0, trial_idx=trial_idx, 
            need_att=best_combo[4], phase='step2'
            # 移除 ema 相關參數
        )
        test_results.append({'auc': test_auc, 'metrics': metrics, 'model_state': model_state})
        all_test_preds.append(test_preds)
        all_test_labels.append(test_labels)
        logger.info(f'Test trial {trial_idx}, Step 2, Test AUC: {test_auc:.4f}')


    avg_test_preds = np.mean(all_test_preds, axis=0)
    test_labels = all_test_labels[0]
    df_avg = pd.DataFrame({'avg_preds': avg_test_preds, 'labels': test_labels})
    file_path_avg = os.path.join(output_dir, f'test_avg_preds_labels_step2_{timestamp}.csv')
    df_avg.to_csv(file_path_avg, index=False)
    logger.info(f'Saved average test preds and labels for Step 2 to {file_path_avg}')


    logger.info("\n=== Final Results (Step 2, NO-EMA) ===")
    avg_test_auc = np.mean([r['metrics']['test']['auc'] for r in test_results])
    avg_test_acc = np.mean([r['metrics']['test']['acc'] for r in test_results])
    avg_test_logloss = np.mean([r['metrics']['test']['logloss'] for r in test_results])
    logger.info(f"Test Average AUC: {avg_test_auc:.4f}, Avg Acc: {avg_test_acc:.4f}, Avg Logloss: {avg_test_logloss:.4f}")


    best_result = best_results[np.argmax([r['avg_auc'] for r in best_results])]
    best_trial_idx = best_result['best_trial_idx']
    best_metrics = best_result['metrics'][best_trial_idx]


    logger.info("\n=== Best Step 1 Metrics (NO-EMA) ===")
    for phase in ['train', 'valid', 'test']:
        metrics = best_metrics[phase]
        logger.info(f"{phase.capitalize()}: AUC={metrics['auc']:.4f}, "
                   f"Acc={metrics['acc']:.4f}, "
                   f"Logloss={metrics['logloss']:.4f}")


    # Load best model (NO-EMA 版本)
    best_model = FIModel(
        n_player=dataset.n_individual,
        player_dim=best_combo[0],
        intermediate_dim=best_combo[1],
        dropout_rate=best_combo[2],
        mlp_hidden_dim=best_combo[3],
        team_size=team_size,
        device=device,
        need_att=best_combo[4]
        # 移除 ema 相關參數
    ).to(device)
    best_model.load_state_dict(combo_results[best_combo]['model_states'][best_trial_idx])


if __name__ == "__main__":
    main()
