import os
import sys
import json
import random
import logging
import gc
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics as metrics

from BaSFiN_noInter import NAC
from data import Data

force_no_freeze = False

device = torch.device("cpu")
n_epochs = 200
batch_size = 32
team_size = 5
num_samples = 100
learning_rate = 1e-5
early_stop_patience = 5
NUM_TRIALS = 5

prior_mu = 0.0
prior_sigma = 1.0
kl_weight = 0.017433288221999882

prob_dim = 64
dropout = 0.2

anfm_player_dim = 49
anfm_hidden_dim = 29
anfm_need = True
anfm_drop = 0.169
anfm_mlplayer = 56

bc_player_dim = 50
bc_intermediate_dim = 37
bc_drop = 0.364
bc_mlplayer = 53
bc_need = True

# 日誌設置
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ========= 隨機種子工具 =========
def set_global_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# 工具函式
def compute_param_grad_norm(model: NAC) -> Dict[str, float]:
    norms = {}
    for name, sub in [("nac_bbb", model.nac_bbb),
                      ("fimodel", model.fimodel),
                      ("nac_anfm", model.nac_anfm)]:
        sq = 0.0
        for p in sub.parameters():
            if p.grad is not None:
                sq += p.grad.data.norm(2).item() ** 2
        norms[name] = sq ** 0.5
    return norms

def evaluate_metrics(pred: np.ndarray, label: np.ndarray) -> Tuple[float, float, float]:
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    logloss = metrics.log_loss(label, np.clip(pred, 1e-3, 0.999))
    acc = (label == (pred > 0.5).astype(int)).mean()
    return auc, acc, logloss

def avg_grad_signals(model: NAC,
                     dataset: Data,
                     batch_size: int,
                     phase: str,
                     device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:

    model.eval()
    mod_sum = torch.zeros(3, device=device)
    param_sum = torch.zeros(3, device=device)
    score_sum = 0.0
    n_batch = 0

    for X, _ in dataset.get_batch(batch_size, phase, shuffle=False):
        X = torch.as_tensor(X, dtype=torch.long, device=device)
        with torch.set_grad_enabled(True):
            _, mod_g, feat_g, par_g = model(
                X,
                training=False,
                return_module_contrib=True,
                return_feature_contrib=True,
                return_param_contrib=True,
            )
        mod_sum += mod_g.abs().sum(dim=0)          # (3,)
        param_sum += par_g.abs()                   # (3,) already scalar per module
        score_sum += feat_g.abs().sum()            # scalar
        n_batch += 1
        model.zero_grad(set_to_none=True)

    avg_mod = (mod_sum / (n_batch * batch_size)).cpu().numpy()          # (3,)
    avg_param = (param_sum / n_batch).cpu().numpy()                     # (3,)
    avg_score = (score_sum / (n_batch * batch_size * 15)).item()        # scalar
    return avg_mod, avg_param, avg_score

# 訓練 / 評估主流程
def train_and_evaluate(  
    dataset: Data,
    config: Dict[str, str],
    stage_idx: int,
    trial_idx: int,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_samples: int,
    device: torch.device,
    early_stop_patience: int,
    use_pretrain: bool,
    freeze_modules: bool,
    prob_dim: int,
    dropout: float,
    bc_need: bool,
    force_no_freeze: bool,
    end_year: str,
    trial_seed: int,
    prev_best_model_path: Optional[str] = None,
):
    # 從 config 中提取變數
    ema_tensor_path = config['ema_tensor_path']
    game_id_mapping_path = config['game_id_mapping_path']
    model_save_dir = config['model_save_dir']
    save_dir = config['save_dir']
    log_dir = config['log_dir']

    # 設置日誌檔案
    log_file = os.path.join(log_dir, f"BaSFiN_Freeze_noInter_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 固定隨機種子
    set_global_seed(trial_seed)


    # 建立模型
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
        bc_need_att=bc_need,
    ).to(device)

    # 載入預訓練
    if stage_idx == 0 and use_pretrain:
        try:
            model.nac_bbb.load_state_dict(torch.load(os.path.join(model_save_dir, f"nac_bbb_best_{end_year}.pth"),
                                                    map_location=device, weights_only=True))
            state_dict = torch.load(
                os.path.join(model_save_dir, f"fimodel_best_{end_year}.pth"),
                map_location=device
            )
            model.fimodel.load_state_dict(state_dict, strict=False)

            state_dict = torch.load(
                os.path.join(model_save_dir, f"anfm_best_{end_year}.pth"),
                map_location=device
            )
            model.nac_anfm.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.error(f"Load pre-trained failed: {e}")
            return None, None, None, None, None
    elif stage_idx == 1 and prev_best_model_path:
        try:
            model.load_state_dict(torch.load(prev_best_model_path, map_location=device, weights_only=True))
            logger.info(f"Loaded Stage-0 best model: {prev_best_model_path}")
        except Exception as e:
            logger.error(f"Load Stage-0 model failed: {e}")
            return None, None, None, None, None

    # 凍結策略
    if stage_idx == 0:                      # Stage-0：只訓練 final MLP
        for sub in (model.nac_bbb, model.fimodel, model.nac_anfm):
            for p in sub.parameters():
                p.requires_grad = False
        logger.info("Stage-0: all sub-modules frozen.")
    else:                                   # Stage-1：全部解凍共同訓練
        for sub in (model.nac_bbb, model.fimodel, model.nac_anfm):
            for p in sub.parameters():
                p.requires_grad = True
        logger.info("Stage-1: all sub-modules unfrozen.")


    # 確保 final_mlp 可訓練
    for p in model.final_mlp.parameters():
        p.requires_grad = True
    if not any(p.requires_grad for p in model.parameters()):
        logger.error("No trainable parameters – abort.")
        return None, None, None, None, None

    # Optimizer / Scheduler
    if stage_idx == 1:
        optimizer = optim.AdamW([
            {
                'params': model.nac_bbb.parameters(),
                'lr': 0.001,
                'weight_decay': 0
            },
            {
                'params': list(model.fimodel.parameters()) + list(model.nac_anfm.parameters()),
                'lr': learning_rate*0.1,
                'weight_decay': 0.005
            },
            {
                'params': model.final_mlp.parameters(),
                'lr': learning_rate*0.1,
                'weight_decay': 0.005
            }
        ])
    else:
        # Stage-0 只訓練 final_mlp
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=0.005
        )


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                     patience=2, min_lr=1e-6)

    total_step = len(dataset.train) // batch_size + 1
    best_metric = float("-inf")
    best_epoch = -1
    best_state = best_results = best_norms = best_mod = best_score = None
    patience_cnt = 0

    phases = ["train", "valid", "test"] if len(dataset.valid) else ["train", "test"]
    key_phase = "valid" if "valid" in phases else "test"

    # Epoch Loop
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        for step, (X, y) in enumerate(dataset.get_batch(batch_size, "train", shuffle=False), 1):
            X = torch.as_tensor(X, dtype=torch.long, device=device)
            probs, *_ = model(X, training=True)
            loss = model.elbo_loss(probs, y, num_samples=num_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch}/{n_epochs} | "
                            f"Step {step}/{total_step} | Loss {loss.item():.4f}")

        logger.info(f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch} | "
                    f"Avg Loss {epoch_loss/total_step:.4f}")

        # 參數梯度 Norm
        grad_norms = compute_param_grad_norm(model)

        # Evaluation
        model.eval()
        results: Dict[str, Dict[str, Any]] = {}
        with torch.no_grad():
            for ph in phases:
                preds, labels = [], []
                for X, y in dataset.get_batch(batch_size, ph, shuffle=False):
                    X = torch.as_tensor(X, dtype=torch.long, device=device)
                    prob, *_ = model(X, training=False)
                    preds.append(prob.mean(dim=0).cpu().numpy())
                    labels.append(y.flatten())
                auc, acc, logloss = evaluate_metrics(np.concatenate(preds),
                                                     np.concatenate(labels))
                results[ph] = {"auc": auc, "acc": acc, "logloss": logloss}

        # 梯度貢獻
        avg_mod_contrib, avg_param_contrib, avg_score_contrib = avg_grad_signals(
            model, dataset, batch_size, key_phase, device)

        test_auc = results.get("test", {}).get("auc", float("nan"))

        logger.info(
            f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch} | "
            f"{key_phase.capitalize()} AUC {results[key_phase]['auc']:.4f} | "
            f"Acc {results[key_phase]['acc']:.4f} | Logloss {results[key_phase]['logloss']:.4f} |"
            f"Test AUC {test_auc:.4f} |"
            f"ScoreGrad {avg_score_contrib:.3e} | "
            f"ModGrad z {avg_mod_contrib[0]:.2e}/comp {avg_mod_contrib[1]:.2e}/coop {avg_mod_contrib[2]:.2e} | "
            f"ParamGrad bbb {avg_param_contrib[0]:.2e}/fi {avg_param_contrib[1]:.2e}/anfm {avg_param_contrib[2]:.2e}"
        )


        # LR Scheduler & early-stop
        scheduler.step(results[key_phase]["auc"])
        if results[key_phase]["auc"] > best_metric:
            best_metric = results[key_phase]["auc"]
            best_epoch = epoch
            best_state = model.state_dict()
            best_results = {**results, "epoch": epoch}
            best_norms = grad_norms
            best_mod = avg_mod_contrib
            best_score = avg_score_contrib
            patience_cnt = 0
            save_path = os.path.join(save_dir,
                                     f"nac_prob{prob_dim}_drop{dropout}_lr{learning_rate}_stage{stage_idx}_trial{trial_idx}.pth")
            torch.save(best_state, save_path)
            logger.info(f"*** New best {key_phase} AUC {best_metric:.4f} (epoch {epoch}) saved -> {save_path}")
        else:
            patience_cnt += 1
            logger.info(f"No improvement -> patience {patience_cnt}/{early_stop_patience}")
            if patience_cnt >= early_stop_patience:
                logger.info(f"Early stop @ epoch {epoch}. Best AUC {best_metric:.4f} (epoch {best_epoch})")
                break

    
    logger.removeHandler(file_handler)
    file_handler.close()

    # Return Best
    if best_state is None:
        logger.error("Training failed – no best state.")
        return None, None, None, None, None

    info = {
        "val_auc": best_metric,
        "results": best_results,
        "epoch": best_epoch,
        "grad_norms": best_norms,
        "module_contrib": best_mod.tolist(),
        "score_contrib": best_score,
        "prob_dim": prob_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "stage": stage_idx,
    }
    return best_metric, best_results, None, None, info

# main
def main():
    year_suffixes = ["2020", "2021", "2022", "2023", "2024"]
    BASE_SEED = 42                
    set_global_seed(BASE_SEED)    
    for end_year in year_suffixes:
        start_year = str(int(end_year) - 11)


        # 配置字典
        config = {
            'path': f"../data/final_data/data_{start_year}_{end_year}.csv",
            'ema_tensor_path': f"../data/ema_tensor/ematensor_{end_year}.pt",
            'game_id_mapping_path': f"../data/tensor/game_id_mapping_{end_year}.json",
            'model_save_dir': f"model/pretrain_BaSFiN_model_{end_year}",
            'save_dir': f"model/NAC_plus_{end_year}",
            'log_dir': f"logs/NAC+_{end_year}"
        }

        # 創建資料夾
        os.makedirs(config['log_dir'], exist_ok=True)
        os.makedirs(config['save_dir'], exist_ok=True)

        logger.info("\n" + "="*80)
        logger.info(f"Starting processing for year: {end_year}")
        logger.info("="*80)

        # 檢查檔案
        for p in [config['path'], config['ema_tensor_path'], config['game_id_mapping_path']]:
            if not os.path.exists(p):
                logger.error(f"Error: Required file not found: {p}")
                continue
        for f in [f"nac_bbb_{end_year}.pth", f"fimodel_{end_year}.pth", f"anfm_{end_year}.pth"]:
            if not os.path.exists(os.path.join(config['model_save_dir'], f)):
                logger.error(f"Error: Pre-trained model missing: {os.path.join(config['model_save_dir'], f)}")
                return

        # 訓練流程
        stage0_aucs, stage1_aucs = [], []
        trial_best_paths = []

        for trial_idx in range(NUM_TRIALS):
            trial_seed = BASE_SEED + trial_idx
            set_global_seed(trial_seed)  
            dataset = Data(config['path'], team_size=team_size, seed=trial_seed)
            logger.info(f"[Trial {trial_idx}] n_individual={dataset.n_individual} | "
                        f"train/valid/test = {len(dataset.train)}/{len(dataset.valid)}/{len(dataset.test)}")
            logger.info("="*90)

            # Stage-0
            logger.info(f"\n=== Trial {trial_idx} | Stage-0 : fine-tune head ===")
            best_model_path, best_val_auc = None, float("-inf")
            metric, results, _, _, info = train_and_evaluate(
                dataset=dataset,
                config=config,
                stage_idx=0,
                trial_idx=trial_idx,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_samples=num_samples,
                device=device,
                early_stop_patience=early_stop_patience,
                use_pretrain=True,
                freeze_modules=True,
                prob_dim=prob_dim,
                dropout=dropout,
                bc_need=bc_need,
                trial_seed=trial_seed,
                force_no_freeze=force_no_freeze,
                end_year=end_year
            )
            if metric is not None:
                best_val_auc = metric
                best_model_path = os.path.join(
                    config['save_dir'],
                    f"nac_prob{prob_dim}_drop{dropout}_lr{learning_rate}_stage0_trial{trial_idx}.pth"
                )
            else:
                logger.error("Stage-0 failed – abort trial")
                continue

            # Stage-1
            logger.info(f"\n=== Trial {trial_idx} | Stage-1 : expand data & full fine-tune ===")
            dataset.expand_training_data()
            best_test_auc = float("-inf")
            metric, results, _, _, info = train_and_evaluate(
                dataset=dataset,
                config=config,
                stage_idx=1,
                trial_idx=trial_idx,
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_samples=num_samples,
                device=device,
                early_stop_patience=early_stop_patience,
                use_pretrain=False,
                freeze_modules=False,
                prob_dim=prob_dim,
                dropout=dropout,
                bc_need=bc_need,
                trial_seed=trial_seed,
                force_no_freeze=force_no_freeze,
                prev_best_model_path=best_model_path,
                end_year=end_year
            )
            if metric is not None:
                best_test_auc = metric

            # 統計
            stage0_aucs.append(best_val_auc)
            stage1_aucs.append(best_test_auc)
            trial_best_paths.append(best_model_path)
            logger.info(f"=== Trial {trial_idx} completed : "
                        f"Best Val AUC {best_val_auc:.4f} | Best Test AUC {best_test_auc:.4f} ===")

        # 年份結束後，log summary
        if stage0_aucs:
            s0_mu, s0_std = np.mean(stage0_aucs), np.std(stage0_aucs)
            s1_mu, s1_std = np.mean(stage1_aucs), np.std(stage1_aucs)
            logger.info("\n========== Trial Summary ==========")
            logger.info(f"Stage-0 Val AUC: {s0_mu:.4f} ± {s0_std:.4f}")
            logger.info(f"Stage-1 Test AUC: {s1_mu:.4f} ± {s1_std:.4f}")
            logger.info("Best model paths per trial:")
            for p in trial_best_paths:
                logger.info(f"  • {p}")

if __name__ == "__main__":
    main()