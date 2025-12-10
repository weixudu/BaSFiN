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
import json
import gc
from BaSFiN_noInter import NAC
from data import Data

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
team_size = 5
prior_mu = 0.0
prior_sigma = 1.0
num_samples = 100
early_stop_patience = 5
learning_rate = 0.00005   # Fixed learning rate
freeze_modules = True
num_trials = 5
num_combinations = 121

# Best hyperparameters
anfm_player_dim = 49
anfm_hidden_dim = 29
anfm_need = True
anfm_drop = 0.169
anfm_mlplayer = 56
kl_weight = 0.017433288221999882
bc_player_dim = 50
bc_intermediate_dim = 37
bc_drop = 0.364
bc_mlplayer = 53
bc_need = True


path               = "../data/final_data/data_2013_2024.csv"
ema_tensor_path    = "../data/ema_tensor/ematensor.pt"
game_id_mapping_path = "../data/tensor/game_id_mapping.json"
model_save_dir     = "model/pretrain_BaSFiN_model"
save_dir           = "model/NAC_plus"
log_dir            = "logs/NAC+"

# Set up logging
log_dir = 'logs/NAC+'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'BaSFiN_128_noInter_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def compute_gradient_norm(model):
    norms = {}
    for name, module in [('nac_bbb', model.nac_bbb), ('fimodel', model.fimodel), ('nac_anfm', model.nac_anfm)]:
        total_norm = 0
        for p in module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        norms[name] = total_norm ** 0.5
    return norms

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

def train_and_evaluate(dataset, n_epochs, batch_size, learning_rate, num_samples, device, 
                      early_stop_patience, use_pretrain, freeze_modules, prob_dim,
                      dropout, combo_idx, trial_idx, bc_need):
    # Set random seed
    seed = SEED + trial_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize model
    model = NAC(
        n_player=dataset.n_individual,
        bc_player_dim=bc_player_dim,
        anfm_player_dim=anfm_player_dim,
        team_size=team_size,
        anfm_hidden_dim=anfm_hidden_dim,
        intermediate_dim=bc_intermediate_dim,
        prob_dim=prob_dim,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        kl_weight=kl_weight,
        dropout=dropout,
        need_att=anfm_need,
        num_samples=num_samples,
        anfm_drop=anfm_drop,
        fim_drop=bc_drop,
        anfm_mlp=anfm_mlplayer,
        fim_mlp=bc_mlplayer,
        device=device,
        ema_tensor_path=ema_tensor_path,
        game_id_mapping_path=game_id_mapping_path,
        model_save_dir=model_save_dir,
        bc_need_att=bc_need
    ).to(device)

    # Load pre-trained weights
    if use_pretrain:
        logger.info("Loading pre-trained weights")
        try:
            model.nac_bbb.load_state_dict(torch.load(os.path.join(model_save_dir, 'nac_bbb.pth'), 
                                                    map_location=device, weights_only=True))
            state_dict = torch.load(
                os.path.join(model_save_dir, 'fimodel.pth'),
                map_location=device,
                weights_only=True
            )

            model.fimodel.load_state_dict(state_dict, strict=False)

            state_dict = torch.load(
                os.path.join(model_save_dir, 'anfm.pth'),
                map_location=device,
                weights_only=True
            )

            model.nac_anfm.load_state_dict(state_dict, strict=False)

        except FileNotFoundError as e:
            logger.error(f"Pre-trained weight file not found: {e}")
            return None, None, None, None, None
        except RuntimeError as e:
            logger.error(f"Failed to load pre-trained weights: {e}")
            return None, None, None, None, None

    # Freeze modules
    if use_pretrain and freeze_modules:
        logger.info("Loading freeze configuration from freeze_config.json")
        freeze_config_path = os.path.join(model_save_dir, 'freeze_config.json')
        if os.path.exists(freeze_config_path):
            with open(freeze_config_path, 'r') as f:
                freeze_dict = json.load(f)
            for module_name in ['nac_bbb', 'fimodel', 'nac_anfm']:
                should_freeze = freeze_dict.get(module_name, False)
                if should_freeze:
                    module = getattr(model, module_name)
                    for param in module.parameters():
                        param.requires_grad = False
                    logger.info(f"Froze {module_name} (early stopped during pre-training)")
                else:
                    logger.info(f"Did not freeze {module_name} (not early stopped)")
        else:
            logger.warning("Freeze configuration file not found, no modules will be frozen")

    # Ensure final_mlp is trainable
    for param in model.final_mlp.parameters():
        param.requires_grad = True
    trainable_params = any(p.requires_grad for p in model.parameters())
    if not trainable_params:
        logger.error("No trainable parameters found, even after ensuring final_mlp is trainable")
        return None, None, None, None, None
    else:
        logger.info("Trainable parameters found, proceeding with training")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=learning_rate, weight_decay=0.005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                    patience=2, min_lr=1e-6)
    total_step = len(dataset.train) // batch_size + 1

    best_val_auc = float('-inf')
    best_model_state = None
    best_results = None
    best_skill_mu = None
    best_skill_sigma = None
    best_grad_norms = None
    patience_counter = 0
    early_stopped = False

    for epoch in range(n_epochs):
        model.train()
        batch_gen = dataset.get_batch(batch_size, 'train', shuffle=False)
        total_loss = 0

        for i, (X, y) in enumerate(batch_gen):
            X = torch.LongTensor(X).to(device)
            probs, _, _, _ = model(X, training=True)
            loss = model.elbo_loss(probs, y, num_samples=num_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / total_step
        logger.info(f'Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')
        grad_norms = compute_gradient_norm(model)
        logger.info(f'Trial {trial_idx}, Epoch [{epoch + 1}/{n_epochs}], Gradient Norms: '
                    f'nac_bbb: {grad_norms["nac_bbb"]:.4f}, '
                    f'fimodel: {grad_norms["fimodel"]:.4f}, '
                    f'nac_anfm: {grad_norms["nac_anfm"]:.4f}')
            
        model.eval()
        phases = ['train', 'valid', 'test']
        results = {}
        for phase in phases:
            preds = []
            labels = []
            batch_gen = dataset.get_batch(batch_size, phase, shuffle=False)
            for X, y in batch_gen:
                X = torch.LongTensor(X).to(device)
                with torch.no_grad():
                    prob, _, _, _ = model(X, training=True)  
                    prob_mean = torch.mean(prob, dim=0)      # [batch_size]
                    preds.append(prob_mean.cpu().numpy())
                    labels.append(y.flatten())              
            y_true = np.concatenate(labels)
            y_pred = np.concatenate(preds)
            auc, acc, logloss = evaluate(y_pred, y_true)

            results[phase] = {'auc': auc, 'acc': acc, 'logloss': logloss}
            logger.info(f'Epoch [{epoch + 1}/{n_epochs}], {phase.capitalize()} AUC: {auc:.4f}, Acc: {acc:.4f}, Logloss: {logloss:.4f}')
        
        scheduler.step(results['valid']['auc'])
        if results['valid']['auc'] > best_val_auc:
            best_val_auc = results['valid']['auc']
            best_model_state = model.state_dict()
            best_results = results
            best_results['epoch'] = epoch + 1
            best_skill_mu = model.nac_bbb.BT.mu.detach().cpu()
            best_skill_sigma = torch.log1p(torch.exp(model.nac_bbb.BT.rho)).detach().cpu()
            best_grad_norms = compute_gradient_norm(model)
            patience_counter = 0
            torch.save(best_model_state, os.path.join(save_dir, f'nac_prob{prob_dim}_drop{dropout}_lr{learning_rate}_trial{trial_idx}.pth'))
            logger.info(f'New best valid AUC: {best_val_auc:.4f} at epoch {epoch + 1}')
        else:
            patience_counter += 1
            logger.info(f'No improvement, patience counter: {patience_counter}/{early_stop_patience}')

        if patience_counter >= early_stop_patience:
            early_stopped = True
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            logger.info(f'Best validation AUC at epoch {best_results["epoch"]}: {best_val_auc:.4f}')
            for phase in phases:
                logger.info(f'Best {phase.capitalize()} metrics at epoch {best_results["epoch"]}: '
                           f'AUC: {best_results[phase]["auc"]:.4f}, '
                           f'Accuracy: {best_results[phase]["acc"]:.4f}, '
                           f'Logloss: {best_results[phase]["logloss"]:.4f}')
            break

    if best_model_state is None:
        logger.error("No valid model state saved")
        return None, None, None, None, None

    # Save final MLP weights
    skill_mean = best_skill_mu.mean().item()
    skill_std = best_skill_sigma.mean().item()

    info = {
        'val_auc': best_val_auc,
        'results': best_results,
        'epoch': best_results['epoch'],
        'grad_norms': best_grad_norms,
        'skill_mu_mean': skill_mean,
        'skill_sigma_mean': skill_std,
        'prob_dim': prob_dim,
        'dropout': dropout,
        'learning_rate': learning_rate
    }
    return best_val_auc, best_results, best_skill_mu, best_skill_sigma, info

def main():
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        return
    if not os.path.exists(ema_tensor_path):
        logger.error(f"EMA tensor file not found: {ema_tensor_path}")
        return
    if not os.path.exists(game_id_mapping_path):
        logger.error(f"Game ID mapping file not found: {game_id_mapping_path}")
        return
    for model_file in ['nac_bbb.pth', 'fimodel.pth', 'anfm.pth']:
        if not os.path.exists(os.path.join(model_save_dir, model_file)):
            logger.error(f"Pre-trained model file not found: {model_file}")
            return

    dataset = Data(path, team_size=team_size, seed=SEED)
    logger.info(f"n_individual: {dataset.n_individual}, max_player_id: {dataset.max_player_id}")
    logger.info(f"Length of index_to_player_id: {len(dataset.index_to_player_id)}")
    game_ids_train = dataset.train[:, 0]
    game_ids_valid = dataset.valid[:, 0]
    game_ids_test = dataset.test[:, 0]
    logger.info(f"Type of game_ids_train: {type(game_ids_train)}, Shape: {game_ids_train.shape}")
    logger.info(f"Type of game_ids_valid: {type(game_ids_valid)}, Shape: {game_ids_valid.shape}")
    logger.info(f"Type of game_ids_test: {type(game_ids_test)}, Shape: {game_ids_test.shape}")
    logger.info(f"Train label ratio: {np.mean(dataset.train[:, -1]):.4f}")
    logger.info(f"Valid label ratio: {np.mean(dataset.valid[:, -1]):.4f}")
    logger.info(f"Test label ratio: {np.mean(dataset.test[:, -1]):.4f}")

    logger.info(f'Training started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 80)

    best_results = []
    combo_results = {}
    combo_idx = 0

    # for _ in range(num_combinations):
            # Randomly sample parameters
            # prob_dim = random.randint(8, 32)
    for prob_dim in range(8, 129):
        dropout = 0.2

        logger.info(f'\n=== Testing combination {combo_idx + 1}/{num_combinations}: '
                    f'prob_dim={prob_dim}, dropout={dropout:.3f}, learning_rate={learning_rate} ===')
        aucs = []
        metrics_list = []
        skill_mus = []
        skill_sigmas = []
        infos = []

        for trial_idx in range(num_trials):
            logger.info(f'Running trial {trial_idx} for prob_dim={prob_dim}, dropout={dropout:.3f}, learning_rate={learning_rate}')
            val_auc, results, skill_mu, skill_sigma, info = train_and_evaluate(
                dataset, n_epochs, batch_size, learning_rate, num_samples, device, 
                early_stop_patience, use_pretrain=True, freeze_modules=freeze_modules, 
                prob_dim=prob_dim, dropout=dropout, 
                combo_idx=combo_idx, trial_idx=trial_idx, bc_need=bc_need
            )
            if val_auc is not None:
                aucs.append(val_auc)
                metrics_list.append(results)
                skill_mus.append(skill_mu)
                skill_sigmas.append(skill_sigma)
                infos.append(info)

        if aucs:
            avg_auc = np.mean(aucs)
            combo_results[(prob_dim, dropout, learning_rate)] = {
                'avg_auc': avg_auc,
                'aucs': aucs,
                'metrics': metrics_list,
                'skill_mus': skill_mus,
                'skill_sigmas': skill_sigmas,
                'infos': infos
            }
            logger.info(f'prob_dim={prob_dim}, dropout={dropout:.3f}, learning_rate={learning_rate}, '
                        f'Average AUC: {avg_auc:.4f}, AUCs: {aucs}')
            best_results.append({
                'prob_dim': prob_dim,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'avg_auc': avg_auc,
                'metrics': metrics_list,
                'best_trial_idx': np.argmax(aucs) if aucs else 0
            })
        combo_idx += 1

    logger.info('\n' + '=' * 80)
    logger.info('Training completed for all hyperparameter combinations!')
    logger.info('Summary of best results per combination:')

    best_avg_auc = float('-inf')
    best_combo = None
    for result in best_results:
        logger.info(f"Prob Dim: {result['prob_dim']}, Dropout: {result['dropout']:.3f}, "
                    f"Learning Rate: {result['learning_rate']}, "
                    f"Average Valid AUC: {result['avg_auc']:.4f}")
        if result['avg_auc'] > best_avg_auc:
            best_avg_auc = result['avg_auc']
            best_combo = (result['prob_dim'], result['dropout'], result['learning_rate'])

    if best_combo is None:
        logger.error("No valid results obtained from any trials")
        return

    logger.info(f"\nBest combination: prob_dim={best_combo[0]}, dropout={best_combo[1]:.3f}, "
                f"learning_rate={best_combo[2]} with Average Valid AUC: {best_avg_auc:.4f}")
    logger.info('Detailed metrics for best combination:')

    best_result = next(r for r in best_results if (r['prob_dim'], r['dropout'], r['learning_rate']) == best_combo)
    best_trial_idx = best_result['best_trial_idx']
    best_info = combo_results[best_combo]['infos'][best_trial_idx]

    logger.info(f"Best trial index: {best_trial_idx}")
    logger.info(f"Best validation AUC: {best_info['val_auc']:.4f}")
    logger.info(f"Best Epoch: {best_info['epoch']}")
    logger.info("Performance metrics:")
    for split in ['train', 'valid', 'test']:
        logger.info(f"{split.capitalize()} AUC: {best_info['results'][split]['auc']:.4f}, "
                    f"Accuracy: {best_info['results'][split]['acc']:.4f}, "
                    f"Logloss: {best_info['results'][split]['logloss']:.4f}")

    logger.info("Gradient norms at best epoch:")
    logger.info(f"nac_bbb: {best_info['grad_norms']['nac_bbb']:.4f}, "
                f"fimodel: {best_info['grad_norms']['fimodel']:.4f}, "
                f"nac_anfm: {best_info['grad_norms']['nac_anfm']:.4f}")
    logger.info(f"Mean skill mu across all players: {best_info['skill_mu_mean']:.4f}, "
                f"Mean sigma: {best_info['skill_sigma_mean']:.4f}")
    
if __name__ == "__main__":
    main()