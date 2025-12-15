import torch, torch.nn as nn, torch.optim as optim
import numpy as np, random, logging, os
import sklearn.metrics as metrics
from datetime import datetime
from BaS import NAC_BBB
from data import Data

# ------------------ 基本設定 ------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device              = torch.device('cpu')
n_epochs            = 200
batch_size          = 32
path                = '../data/final_data/data_2013_2024.csv'
team_size           = 5
num_samples         = 100
early_stop_patience = 5
num_trials          = 3

# -------------- 日誌 --------------------------
log_dir = 'logs/BBB'; os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f"BaS_random_{current_time}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------ 損失與評估 ------------------
def elbo_loss(model, X, y, kl_w, ns):
    y_t = torch.tensor(y, dtype=torch.float, device=device)
    y_rep = y_t.unsqueeze(0).repeat(ns, 1)

    logits, _ = model(X, num_samples=ns)  # logits ∈ R
    ll = -nn.BCEWithLogitsLoss(reduction='sum')(logits, y_rep) / ns
    return -ll + kl_w * model.kl_divergence()


def evaluate(pred, label):
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    auc = metrics.roc_auc_score(label, pred)
    ll  = metrics.log_loss(label, np.clip(pred, 1e-3, 1-1e-3))
    acc = (label == (pred > .5)).mean()
    return auc, acc, ll

# ------------------ 訓練迴圈 -------------------
def train_and_evaluate(kl_w, lr, dataset, idx, trial):
    model = NAC_BBB(dataset.n_individual, team_size, device, 0, 1).to(device)
    opt   = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    sch   = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2, min_lr=1e-6)

    best_val, best_test, best_met, patience = -1e9, -1e9, {}, 0
    phases = ['train', 'valid', 'test'] if len(dataset.valid) > 0 else ['train', 'test']

    for ep in range(1, n_epochs + 1):
        # --------- train ---------
        model.train()
        tot_loss = 0
        for X, y in dataset.get_batch(batch_size, 'train'):
            X = torch.tensor(X, dtype=torch.long, device=device)
            loss = elbo_loss(model, X, y, kl_w, num_samples)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item()

        logger.info(f"[{idx}|{trial}] KL={kl_w:.5f} LR={lr:.5f} "
                    f"Ep {ep}/{n_epochs} TrainLoss {tot_loss:.4f}")

        # --------- eval ---------
        model.eval()
        metrics_dict = {}

        with torch.no_grad():
            for ph in phases:
                ps, labs = [], []

                for X, y in dataset.get_batch(batch_size, ph):
                    X = torch.tensor(X, dtype=torch.long, device=device)
                    logits, _ = model(X, num_samples=num_samples)
                    prob = torch.sigmoid(logits).mean(0)
                    ps.append(prob.cpu().numpy())
                    labs.append(y.flatten())

                # ✅ 防止空集合
                if len(ps) == 0:
                    logger.warning(f"[{idx}|{trial}] phase={ph} has no data, skipped.")
                    continue

                auc, acc, ll = evaluate(
                    np.concatenate(ps),
                    np.concatenate(labs)
                )

                metrics_dict[ph] = dict(auc=auc, acc=acc, logloss=ll)
                logger.info(f"    {ph.capitalize():5s} AUC {auc:.4f} "
                            f"Acc {acc:.4f} LogLoss {ll:.4f}")

        # --------- early stopping ---------
        if 'valid' in metrics_dict:
            sch.step(metrics_dict['valid']['auc'])
            if metrics_dict['valid']['auc'] > best_val:
                best_val, best_met = metrics_dict['valid']['auc'], metrics_dict
                patience = 0
            else:
                patience += 1
        else:
            sch.step(metrics_dict['test']['auc'])
            if metrics_dict['test']['auc'] > best_test:
                best_test, best_met = metrics_dict['test']['auc'], metrics_dict
                patience = 0
            else:
                patience += 1

        if patience >= early_stop_patience:
            break

    valid_auc = best_met['valid']['auc'] if 'valid' in best_met else None
    test_auc  = best_met['test']['auc']
    return valid_auc, test_auc

# ================= Main ========================
def main():
    dataset = Data(path, team_size=team_size, seed=SEED)
    logger.info("=== Random search over KL & LR ===")

    N_SAMPLES = 20
    rng = np.random.RandomState(SEED)
    kl_list = rng.uniform(0.005, 0.5, N_SAMPLES)
    lr_list = rng.uniform(0.001, 0.01, N_SAMPLES)

    best_valid_auc, best_cfg = -1e9, None

    for idx, (kl_w, lr) in enumerate(zip(kl_list, lr_list), 1):
        valid_aucs, test_aucs = [], []

        logger.info(f"\n-- Sample {idx}/{N_SAMPLES}: KL={kl_w:.5f} LR={lr:.5f} --")
        for t in range(num_trials):
            vauc, tauc = train_and_evaluate(kl_w, lr, dataset, idx, t)
            if vauc is not None:
                valid_aucs.append(vauc)
            test_aucs.append(tauc)

        mean_valid_auc = np.mean(valid_aucs) if valid_aucs else None
        mean_test_auc  = np.mean(test_aucs)

        if mean_valid_auc is not None:
            logger.info(f"Mean VALID AUC {mean_valid_auc:.4f}  Mean TEST AUC {mean_test_auc:.4f}")
            if mean_valid_auc > best_valid_auc:
                best_valid_auc = mean_valid_auc
                best_cfg = (kl_w, lr)
        else:
            logger.info(f"Mean TEST AUC {mean_test_auc:.4f}")

    logger.info("\n=== Search complete ===")
    if best_cfg:
        logger.info(f"best KL_weight = {best_cfg[0]:.5f}")
        logger.info(f"best learning_rate = {best_cfg[1]:.5f}")
        logger.info(f"VALID AUC = {best_valid_auc:.4f}")
    else:
        logger.info("no valid set → using test AUC only")

if __name__ == "__main__":
    main()
