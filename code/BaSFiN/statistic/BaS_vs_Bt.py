import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, roc_curve
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt


# 設置日誌
log_dir = 'logs/tests'
os.makedirs(log_dir, exist_ok=True)
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f'statistical_tests_{current_time}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# DeLong Test 實現（基於 statsmodels 或自定義）
def delong_test(y_true, y_pred1, y_pred2):
    """
    執行 DeLong Test，比較兩個模型的 AUC。
    """
    def auc_covariance(y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        n1 = np.sum(y_true == 1)
        n0 = np.sum(y_true == 0)
        if n1 == 0 or n0 == 0:
            raise ValueError("AUC covariance calculation requires both positive and negative samples")
        
        # 計算正類和負類的排序分數
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        
        # 計算 V10 和 V01
        v10 = np.mean([np.mean(pos_scores > neg_score) for neg_score in neg_scores])
        v01 = np.mean([np.mean(neg_scores > pos_score) for pos_score in pos_scores])
        
        # 計算方差
        var_v10 = np.var([np.mean(pos_scores > neg_score) for neg_score in neg_scores]) / n0
        var_v01 = np.var([np.mean(neg_scores > pos_score) for pos_score in pos_scores]) / n1
        
        # 協方差矩陣
        cov = np.array([[var_v10, 0], [0, var_v01]])
        return auc, cov

    auc1, cov1 = auc_covariance(y_true, y_pred1)
    auc2, cov2 = auc_covariance(y_true, y_pred2)
    
    # 計算 AUC 差異的方差
    diff = auc1 - auc2
    var_diff = cov1[0, 0] + cov2[0, 0]  # 假設模型間獨立，協方差為 0
    if var_diff <= 0:
        logger.warning("Variance of AUC difference is zero or negative, cannot compute z-score")
        return 0.0, 1.0
    
    # 計算 z 分數和 p 值
    z = diff / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))  # 雙尾檢驗
    return z, p_value

# McNemar Test
def mcnemar_test(y_true, y_pred1, y_pred2, threshold=0.5):
    """
    執行 McNemar Test，比較兩個模型的分類正確性。
    """
    pred1 = (y_pred1 >= threshold).astype(int)
    pred2 = (y_pred2 >= threshold).astype(int)
    
    # 構建列聯表
    table = np.zeros((2, 2))
    for t, p1, p2 in zip(y_true, pred1, pred2):
        if p1 == t and p2 == t:
            table[0, 0] += 1  # 兩個模型都正確
        elif p1 == t and p2 != t:
            table[0, 1] += 1  # 模型 1 正確，模型 2 錯誤
        elif p1 != t and p2 == t:
            table[1, 0] += 1  # 模型 1 錯誤，模型 2 正確
        else:
            table[1, 1] += 1  # 兩個模型都錯誤
    
    # 執行 McNemar Test
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue

def permutation_test(y_true, y_pred1, y_pred2, n_permutations=1000, metric='logloss'):
    """
    執行 Permutation Test，比較兩個模型的 Log Loss（或其他指標）。
    """
    def compute_metric(y_true, y_pred):
        if metric == 'logloss':
            y_pred = np.clip(y_pred, 0.001, 0.999)
            return log_loss(y_true, y_pred)
        elif metric == 'auc':
            return roc_auc_score(y_true, y_pred)
        elif metric == 'accuracy':
            return np.mean((y_pred >= 0.5).astype(int) == y_true)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    metric1 = compute_metric(y_true, y_pred1)
    metric2 = compute_metric(y_true, y_pred2)
    observed_diff = metric1 - metric2  # 注意：Log Loss 越小越好，因此差異方向相反
    
    diffs = []
    for _ in range(n_permutations):
        mask = np.random.rand(len(y_true)) < 0.5
        perm_pred1 = np.where(mask, y_pred1, y_pred2)
        perm_pred2 = np.where(mask, y_pred2, y_pred1)
        
        perm_metric1 = compute_metric(y_true, perm_pred1)
        perm_metric2 = compute_metric(y_true, perm_pred2)
        diffs.append(perm_metric1 - perm_metric2)
    
    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

def plot_roc_curves(y_true, y_pred_a, y_pred_b, auc_a, auc_b):
    fpr_a, tpr_a, _ = roc_curve(y_true, y_pred_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, y_pred_b)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_a, tpr_a, label=f'BaS (AUC = {auc_a:.4f})', color='blue')
    plt.plot(fpr_b, tpr_b, label=f'BT (AUC = {auc_b:.4f})', color='red')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curves for BaS and BT')
    plt.legend()
    plt.grid(True)
    plot_dir = "../plot"
    output_path = os.path.join(plot_dir, f'roc_curves_Bas_BT.png')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curves saved to {output_path}")


def main():
    # 讀取模型 A 和模型 B 的 CSV 文件
    model_a_path = '../output/BaS/test_avg_preds_labels_kl0.017433288221999882_2025-06-12_15-31-11.csv'
    model_b_path = '../output/BT/test_avg_preds_labels_step2_2025-06-12_16-41-52.csv'
    
    logger.info(f"Loading model A predictions from {model_a_path}")
    df_a = pd.read_csv(model_a_path)
    y_true_a = df_a['labels'].values
    y_pred_a = df_a['avg_preds'].values
    
    logger.info(f"Loading model B predictions from {model_b_path}")
    df_b = pd.read_csv(model_b_path)
    y_true_b = df_b['labels'].values
    y_pred_b = df_b['avg_preds'].values
    
    # 檢查標籤一致性
    if not np.array_equal(y_true_a, y_true_b):
        logger.error("Labels from model A and model B are not identical")
        raise ValueError("Labels must be identical for statistical tests")
    
    y_true = y_true_a
    logger.info(f"Number of samples: {len(y_true)}")
    
    # 計算並記錄 AUC、Accuracy 和 Log Loss
    auc_a = roc_auc_score(y_true, y_pred_a)
    auc_b = roc_auc_score(y_true, y_pred_b)
    acc_a = accuracy_score(y_true, (y_pred_a >= 0.5).astype(int))
    acc_b = accuracy_score(y_true, (y_pred_b >= 0.5).astype(int))
    logloss_a = log_loss(y_true, np.clip(y_pred_a, 0.001, 0.999))
    logloss_b = log_loss(y_true, np.clip(y_pred_b, 0.001, 0.999))
    
    logger.info(f"Model A: AUC = {auc_a:.4f}, Accuracy = {acc_a:.4f}, Log Loss = {logloss_a:.4f}")
    logger.info(f"Model B: AUC = {auc_b:.4f}, Accuracy = {acc_b:.4f}, Log Loss = {logloss_b:.4f}")
    
    logger.info("\n=== DeLong Test ===")
    try:
        z, p_value = delong_test(y_true, y_pred_a, y_pred_b)
        logger.info(f"DeLong Test: z-statistic = {z:.4f}, p-value = {p_value:.4e}")
    except Exception as e:
        logger.error(f"DeLong Test failed: {str(e)}")
    
    logger.info("\n=== McNemar Test ===")
    try:
        stat, p_value = mcnemar_test(y_true, y_pred_a, y_pred_b, threshold=0.5)
        logger.info(f"McNemar Test: statistic = {stat:.4f}, p-value = {p_value:.4e}")
    except Exception as e:
        logger.error(f"McNemar Test failed: {str(e)}")
    
    # 執行 Permutation Test (比較 Log Loss)
    logger.info("\n=== Permutation Test (Log Loss) ===")
    try:
        diff, p_value = permutation_test(y_true, y_pred_a, y_pred_b, n_permutations=1000, metric='logloss')
        logger.info(f"Permutation Test (Log Loss): observed difference = {diff:.4f}, p-value = {p_value:.4e}")
    except Exception as e:
        logger.error(f"Permutation Test (Log Loss) failed: {str(e)}")

    # 繪製 ROC 曲線
    logger.info("\n=== Plotting ROC Curves ===")
    try:
        plot_roc_curves(y_true, y_pred_a, y_pred_b, auc_a, auc_b)
    except Exception as e:
        logger.error(f"ROC curve plotting failed: {str(e)}")

if __name__ == "__main__":
    from scipy import stats
    main()