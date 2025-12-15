import os, sys, json, random, logging, gc
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import sklearn.metrics as metrics

from BaSFiN_noInter import NAC               
from data import Data

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
######### stage-0 是否強制不凍結#######
force_no_freeze   = False

device = torch.device("cpu")
n_epochs         = 200
batch_size       = 32
team_size        = 5
num_samples      = 100
learning_rate    = 5e-5
early_stop_patience = 5
NUM_TRIALS = 5


prior_mu   = 0.0
prior_sigma= 1.0
kl_weight  = 0.017433288221999882

prob_dim   = 64
dropout    = 0.2

anfm_player_dim   = 49
anfm_hidden_dim   = 29
anfm_need         = True
anfm_drop         = 0.169
anfm_mlplayer     = 56

bc_player_dim     = 50
bc_intermediate_dim = 37
bc_drop           = 0.364
bc_mlplayer       = 53
bc_need           = True



# path               = "../data/final_data/data_2013_2024.csv"
# ema_tensor_path    = "../data/ema_tensor/ematensor.pt"
# game_id_mapping_path = "../data/tensor/game_id_mapping.json"
# model_save_dir     = "NAC_gpu/pretrain_BaSFiN_model"
# save_dir           = "NAC_gpu/NAC_plus"
# log_dir            = "logs/NAC+"

path               = "../data/final_data/data_2013_2024.csv"
ema_tensor_path    = "../data/ema_tensor/ematensor.pt"
game_id_mapping_path = "../data/tensor/game_id_mapping.json"
model_save_dir     = "model/pretrain_BaSFiN_model"
save_dir           = "NAC_gpu/NAC_plus"
log_dir            = "logs/NAC+"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# ------------------- Logger ------------------- #
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = os.path.join(log_dir, f"BaSFiN_noInter_SGD_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(log_file, encoding="utf-8")],
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# ====================================================== #
#                       工具函式                         #
# ====================================================== #
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
    pred  = pred.reshape(-1)
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
    """
    回傳
      • avg_mod_grad  : ndarray(3,)   – z / comp / coop 特徵貢獻 (|grad| 平均)
      • avg_param_grad: ndarray(3,)   – nac_bbb / fimodel / anfm 參數 L1-norm (平均)
      • avg_score_grad: float         – 15 維特徵 |grad| 全域平均
    """
    model.eval()
    mod_sum   = torch.zeros(3, device=device)
    param_sum = torch.zeros(3, device=device)
    score_sum = 0.0
    n_batch   = 0

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
        mod_sum   += mod_g.abs().sum(dim=0)          # (3,)
        param_sum += par_g.abs()                     # (3,)  already scalar per module
        score_sum += feat_g.abs().sum()              # scalar
        n_batch   += 1
        model.zero_grad(set_to_none=True)

    avg_mod   = (mod_sum / (n_batch * batch_size)).cpu().numpy()          # (3,)
    avg_param = (param_sum / n_batch).cpu().numpy()                       # (3,)
    avg_score = (score_sum / (n_batch * batch_size * 15)).item()          # scalar
    return avg_mod, avg_param, avg_score


# ====================================================== #
#                    訓練 / 評估主流程                   #
# ====================================================== #
def train_and_evaluate(
    dataset: Data,
    *,
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
    prev_best_model_path: Optional[str] = None,
):

    # ------- 固定隨機種子 (per trial) ------- #
    seed = SEED + trial_idx
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ---------------- 建立模型 ---------------- #
    model = NAC(
        n_hero=dataset.n_individual,
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

    # ---------- 載入預訓練 ---------- #
    if stage_idx == 0 and use_pretrain:
        try:
            for sub, fn in zip(
                (model.nac_bbb, model.fimodel, model.nac_anfm),
                ("nac_bbb.pth", "fimodel.pth", "anfm.pth")
            ):
                state = torch.load(
                    os.path.join(model_save_dir, fn),
                    map_location=device,
                    weights_only=True,
                )
                sub.load_state_dict(state, strict=False)   # ❶ 變動只在 strict=False
            logger.info("Loaded pre-trained weights.")
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

    # ---------- 凍結策略 ---------- #
    if stage_idx == 0 and freeze_modules and use_pretrain and not force_no_freeze:
        cfg_path = os.path.join(model_save_dir, "freeze_config.json")
        if os.path.isfile(cfg_path):
            freeze_cfg = json.load(open(cfg_path, encoding="utf-8"))
            for name in ["nac_bbb", "fimodel", "nac_anfm"]:
                if freeze_cfg.get(name, False):
                    for p in getattr(model, name).parameters():
                        p.requires_grad = False
                    logger.info(f"Froze {name}")
        else:
            logger.warning("freeze_config.json not found – no freeze applied.")
    elif stage_idx == 1:
        for sub in (model.nac_bbb, model.fimodel, model.nac_anfm):
            for p in sub.parameters():
                p.requires_grad = True
        logger.info("Stage-1: all sub-modules unfrozen.")
    if force_no_freeze:
        logger.info("force_no_freeze = True → ensure all params trainable.")

    # 確保 final_mlp 一律可訓練
    for p in model.final_mlp.parameters(): p.requires_grad = True
    if not any(p.requires_grad for p in model.parameters()):
        logger.error("No trainable parameters – abort."); return None, None, None, None, None

    # ---------- Optimizer / Scheduler ---------- #
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    total_step = len(dataset.train) // batch_size + 1
    best_metric = float("-inf"); best_epoch = -1
    best_state = best_results = best_norms = best_mod = best_score = None
    patience_cnt = 0

    phases = ["train", "valid", "test"] if len(dataset.valid) else ["train", "test"]
    key_phase = "valid" if "valid" in phases else "test"

    # ==================== Epoch Loop ==================== #
    for epoch in range(1, n_epochs + 1):

        # ------- Training ------- #
        model.train(); epoch_loss = 0.0
        for step, (X, y) in enumerate(dataset.get_batch(batch_size, "train", shuffle=True), 1):
            X = torch.as_tensor(X, dtype=torch.long, device=device)
            probs, *_ = model(X, training=True)
            loss = model.elbo_loss(probs, y, num_samples=num_samples)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            if step % 10 == 0:
                logger.info(f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch}/{n_epochs} | "
                            f"Step {step}/{total_step} | Loss {loss.item():.4f}")

        logger.info(f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch} | "
                    f"Avg Loss {epoch_loss/total_step:.4f}")

        # ------- 參數梯度 Norm ------- #
        grad_norms = compute_param_grad_norm(model)

        # ------- Evaluation (預測指標) ------- #
        model.eval(); results: Dict[str, Dict[str, Any]] = {}
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

        # ------- 梯度貢獻 (只算 key phase) ------- #
        avg_mod_contrib, avg_param_contrib, avg_score_contrib = avg_grad_signals(
            model, dataset, batch_size, key_phase, device)

        # val_auc  = results.get("valid", {}).get("auc", float("nan"))
        test_auc = results.get("test",  {}).get("auc", float("nan"))

        logger.info(
            f"Stage {stage_idx} | Trial {trial_idx} | Epoch {epoch} | "
            f"{key_phase.capitalize()} AUC {results[key_phase]['auc']:.4f} | "
            f"Acc {results[key_phase]['acc']:.4f} | Logloss {results[key_phase]['logloss']:.4f} | \n"
            f"Test AUC {test_auc:.4f} | \n"
            f"ScoreGrad {avg_score_contrib:.3e} | "
            f"ModGrad z {avg_mod_contrib[0]:.2e}/comp {avg_mod_contrib[1]:.2e}/coop {avg_mod_contrib[2]:.2e} | "
            f"ParamGrad bbb {avg_param_contrib[0]:.2e}/fi {avg_param_contrib[1]:.2e}/anfm {avg_param_contrib[2]:.2e}"
        )

        # ------- LR Scheduler & early-stop ------- #
        scheduler.step(results[key_phase]["auc"])
        if results[key_phase]["auc"] > best_metric:
            best_metric, best_epoch = results[key_phase]["auc"], epoch
            best_state = model.state_dict()
            best_results = results | {"epoch": epoch}
            best_norms  = grad_norms
            best_mod    = avg_mod_contrib
            best_score  = avg_score_contrib
            patience_cnt = 0
            save_path = os.path.join(save_dir,
                f"nac_prob{prob_dim}_drop{dropout}_lr{learning_rate}_SGD_stage{stage_idx}_trial{trial_idx}.pth")
            torch.save(best_state, save_path)
            logger.info(f"*** New best {key_phase} AUC {best_metric:.4f} (epoch {epoch}) saved → {save_path}")
        else:
            patience_cnt += 1
            logger.info(f"No improvement → patience {patience_cnt}/{early_stop_patience}")
            if patience_cnt >= early_stop_patience:
                logger.info(f"Early stop @ epoch {epoch}. Best AUC {best_metric:.4f} (epoch {best_epoch})")
                break

    # ---------------- Return Best ---------------- #
    if best_state is None:
        logger.error("Training failed – no best state."); return None, None, None, None, None

    info = {
        "val_auc": best_metric,
        "results": best_results,
        "epoch":  best_epoch,
        "grad_norms": best_norms,
        "module_contrib": best_mod.tolist(),
        "score_contrib":  best_score,
        "prob_dim": prob_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "stage": stage_idx,
    }
    return best_metric, best_results, None, None, info


# ====================================================== #
#                          main                          #
# ====================================================== #
def main():
    # ---- 檢查資料/權重 ---- #
    for p in [path, ema_tensor_path, game_id_mapping_path]:
        if not os.path.exists(p):
            logger.error(f"Required file not found: {p}"); return
    for f in ["nac_bbb.pth", "fimodel.pth", "anfm.pth"]:
        if not os.path.exists(os.path.join(model_save_dir, f)):
            logger.error(f"Pre-trained model missing: {f}"); return

    stage0_aucs, stage1_aucs = [], []
    trial_best_paths = []
    
    for trial_idx in range(NUM_TRIALS):

        dataset = Data(path, team_size=team_size, seed=SEED+trial_idx)
        logger.info(f"[Trial {trial_idx}] n_individual={dataset.n_individual} | "
                    f"train/valid/test = {len(dataset.train)}/{len(dataset.valid)}/{len(dataset.test)}")
        logger.info("="*90)

        # ---------- Stage-0 ----------
        logger.info(f"\n=== Trial {trial_idx} | Stage-0 : fine-tune head ===")
        best_model_path, best_val_auc = None, float("-inf")
        metric, *_ = train_and_evaluate(
            dataset=dataset,
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
            force_no_freeze=force_no_freeze,
        )
        if metric is not None:
            best_val_auc = metric
            best_model_path = os.path.join(
                save_dir,
                f"nac_prob{prob_dim}_drop{dropout}_lr{learning_rate}_stage0_trial{trial_idx}.pth")
        else:
            logger.error("Stage-0 failed – abort trial"); continue


        # ---------- Stage-1 ----------
        logger.info(f"\n=== Trial {trial_idx} | Stage-1 : expand data & full fine-tune ===")
        dataset.expand_training_data()
        best_test_auc = float("-inf")
        metric, *_ = train_and_evaluate(
            dataset=dataset,
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
            force_no_freeze=force_no_freeze,
            prev_best_model_path=best_model_path,
        )
        if metric is not None:
            best_test_auc = metric

        # -- 收集統計 --
        stage0_aucs.append(best_val_auc)
        stage1_aucs.append(best_test_auc)
        trial_best_paths.append(best_model_path)
        logger.info(f"=== Trial {trial_idx} completed : "
                    f"Best Val AUC {best_val_auc:.4f} | Best Test AUC {best_test_auc:.4f} ===")

    # --------- 整體統計結果 ---------
    if stage0_aucs:
        s0_mu, s0_std = np.mean(stage0_aucs), np.std(stage0_aucs)
        s1_mu, s1_std = np.mean(stage1_aucs), np.std(stage1_aucs)
        print("\n========== Summary across trials ==========")
        print(f"Stage-0 Val AUC : {s0_mu:.4f} ± {s0_std:.4f}")
        print(f"Stage-1 Test AUC: {s1_mu:.4f} ± {s1_std:.4f}")
        print("Best model paths per trial:")
        for p in trial_best_paths:  print("  •", p)


if __name__ == "__main__":
    main()


