"""
Comprehensive Metabolomics ML Pipeline
Implements Stage 1 (Conventional Methods) and Stage 2 (LUPI with Knowledge Distillation)

Stage 1: Exhaustive search over feature selection methods, sampling strategies, and models.
Stage 2: Two-branch knowledge distillation using best Stage 1 models (no PCA).
  Branch 1 - Student trained in teacher's feature space
  Branch 2 - Student trained in union of teacher + student feature spaces
"""

import pandas as pd
import numpy as np
import random
import logging
import argparse
import os
from pathlib import Path
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests

# ML imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve,
    auc, average_precision_score, roc_curve
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import shap

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

# ── Global plot style ──────────────────────────────────────────────────────────
PALETTE = ['#2C7BB6', '#D7191C', '#1A9641', '#FDAE61', '#762A83', '#ABD9E9']

def set_journal_style():
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':         11,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'axes.labelsize':    11,
        'axes.labelweight':  'bold',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.facecolor':    'white',
        'figure.facecolor':  'white',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'legend.frameon':    False,
        'grid.color':        '#E0E0E0',
        'grid.linewidth':    0.6,
        'lines.linewidth':   1.8,
        'patch.linewidth':   0.8,
        'savefig.dpi':       300,
        'savefig.bbox':      'tight',
    })

set_journal_style()


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_file='pipeline_log.txt'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_figure(fig, filename, logger):
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved figure: {filename}")
    except Exception as e:
        logger.error(f"Error saving figure {filename}: {e}")
        plt.close(fig)


def make_dataset_output_dir(base_output_dir, dataset_name):
    folder = os.path.join(base_output_dir, dataset_name)
    os.makedirs(folder, exist_ok=True)
    return folder

def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    scores = []

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, n_pos, replace=True),
            rng.choice(neg_idx, n_neg, replace=True)
        ])

        yt = y_true[idx]
        ys = y_score[idx]

        try:
            val = metric_fn(yt, ys)
            if not np.isnan(val):
                scores.append(val)
        except:
            continue

    if not(scores):
        return 0.0, 0.0, 0.0

    scores = np.array(scores)
    alpha = (1 - ci) / 2

    return (
        float(np.mean(scores)),
        float(np.percentile(scores, alpha * 100)),
        float(np.percentile(scores, (1 - alpha) * 100))
    )

def compute_all_metrics_with_ci(y_true, y_proba, y_pred, n_boot=1000):
    y_true  = np.array(y_true, dtype=np.float64)
    y_proba = np.array(y_proba, dtype=np.float64)
    y_pred  = np.array(y_pred, dtype=np.float64)

    def _roc(yt, ys):  return roc_auc_score(yt, ys)
    def _ap(yt, ys):   return average_precision_score(yt, ys)
    def _prec(yt, ys): return precision_score(yt, ys, pos_label=1, zero_division=0)
    def _rec(yt, ys):  return recall_score(yt, ys, pos_label=1, zero_division=0)
    def _f1(yt, ys):   return f1_score(yt, ys, pos_label=1, zero_division=0)

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_val = auc(rec_arr, prec_arr)

    roc_mean, roc_lo, roc_hi   = bootstrap_ci(y_true, y_proba, _roc,  n_boot)
    ap_mean,  ap_lo,  ap_hi    = bootstrap_ci(y_true, y_proba, _ap,   n_boot)
    prec_mean,prec_lo,prec_hi  = bootstrap_ci(y_true, y_pred,  _prec, n_boot)
    rec_mean, rec_lo, rec_hi   = bootstrap_ci(y_true, y_pred,  _rec,  n_boot)
    f1_mean,  f1_lo,  f1_hi    = bootstrap_ci(y_true, y_pred,  _f1,   n_boot)

    return {
        'roc_auc':      roc_mean, 'roc_auc_lo':   roc_lo,  'roc_auc_hi':   roc_hi,
        'pr_auc':       pr_auc_val,
        'ap_score':     ap_mean,  'ap_score_lo':  ap_lo,   'ap_score_hi':  ap_hi,
        'precision':    prec_mean,'precision_lo': prec_lo, 'precision_hi': prec_hi,
        'recall':       rec_mean, 'recall_lo':    rec_lo,  'recall_hi':    rec_hi,
        'f1':           f1_mean,  'f1_lo':        f1_lo,   'f1_hi':        f1_hi,
        'confusion_matrix':       confusion_matrix(y_true, y_pred).tolist(),
        'classification_report':  classification_report(y_true, y_pred, output_dict=True),
    }


def youden_threshold(y_true, y_proba):
    """Find optimal classification threshold using Youden's J statistic."""
    y_true  = np.array(y_true,  dtype=np.float64)
    y_proba = np.array(y_proba, dtype=np.float64)
    thresholds = np.linspace(0, 1, 201)
    best_thr, best_j = 0.5, -np.inf
    for thr in thresholds:
        yp = (y_proba >= thr).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            j   = tpr - fpr
            if j > best_j:
                best_j = j
                best_thr = thr
        except Exception:
            pass
    return best_thr


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def random_oversample(X, y):
    class_counts = np.bincount(y.astype(int))
    max_count    = np.max(class_counts)
    X_res, y_res = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < max_count:
            extra = np.random.choice(idx, size=max_count - len(idx), replace=True)
            idx   = np.concatenate([idx, extra])
        X_res.append(X[idx])
        y_res.append(y[idx])
    return np.vstack(X_res), np.hstack(y_res)


def random_undersample(X, y):
    class_counts = np.bincount(y.astype(int))
    min_count    = np.min(class_counts)
    X_res, y_res = [], []
    for cls in np.unique(y):
        idx  = np.where(y == cls)[0]
        samp = np.random.choice(idx, size=min_count, replace=False)
        X_res.append(X[samp])
        y_res.append(y[samp])
    return np.vstack(X_res), np.hstack(y_res)


def smote_sample(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def find_elbow_point(values):
    n_points    = len(values)
    all_coords  = np.vstack([range(n_points), values]).T
    first_point = all_coords[0]
    last_point  = all_coords[-1]
    line_vec      = last_point - first_point
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_first = all_coords - first_point
    scalar_proj    = np.dot(vec_from_first, line_vec_norm)
    vec_proj       = np.outer(scalar_proj, line_vec_norm)
    vec_to_line    = vec_from_first - vec_proj
    return int(np.argmax(np.linalg.norm(vec_to_line, axis=1)))


def univariate_raw_pvalue(X_train, y_train, X_test, alpha=0.05):
    pvals = []
    for j in range(X_train.shape[1]):
        groups = [X_train[y_train == c, j] for c in np.unique(y_train)]
        _, p   = f_oneway(*groups)
        pvals.append(p)
    pvals = np.array(pvals)
    mask  = pvals < alpha
    if mask.sum() == 0:
        mask = pvals < np.percentile(pvals, 10)
    mask = mask.astype(bool)
    return X_train[:, mask], X_test[:, mask], mask


def univariate_bh_pvalue(X_train, y_train, X_test, alpha=0.05):
    pvals = []
    for j in range(X_train.shape[1]):
        groups = [X_train[y_train == c, j] for c in np.unique(y_train)]
        _, p   = f_oneway(*groups)
        pvals.append(p)
    pvals = np.array(pvals)
    reject, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    if reject.sum() == 0:
        reject = pvals < np.percentile(pvals, 10)
    reject = reject.astype(bool)
    return X_train[:, reject], X_test[:, reject], reject


def rf_rfecv_selection(X_train, y_train, X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rfecv = RFECV(
        estimator=rf,
        step=max(1, X_train.shape[1] // 20),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    rfecv.fit(X_train, y_train)
    mask = rfecv.support_.astype(bool)
    return X_train[:, mask], X_test[:, mask], mask, rfecv


def rf_sequential_topk(X_train, y_train, X_test, logger):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]

    cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    probe = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective='binary:logistic', eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1, tree_method='hist'
    )

    step_size = 1
    k_values  = list(range(step_size, len(order) + 1, step_size))
    if k_values[-1] != len(order):
        k_values.append(len(order))
    k_values = k_values[:200]

    aucs = []
    for k in k_values:
        idx_k  = order[:k]
        Xk     = X_train[:, idx_k]
        scores = []
        for tr, val in cv.split(Xk, y_train):
            probe.fit(Xk[tr], y_train[tr])
            scores.append(roc_auc_score(y_train[val], probe.predict_proba(Xk[val])[:, 1]))
        aucs.append(np.mean(scores))

    best_k       = k_values[int(np.argmax(aucs))]
    best_indices = order[:best_k]

    bool_mask = np.zeros(X_train.shape[1], dtype=bool)
    bool_mask[best_indices] = True

    logger.info(f"    Sequential top-k: k={best_k}, peak CV AUC={max(aucs):.4f}")
    return X_train[:, bool_mask], X_test[:, bool_mask], bool_mask, k_values, aucs


def apply_selector_to_data(X_raw_df, selector_info, fitted_scalers):
    """
    Apply a saved feature selector (from Stage 1) to new raw data.

    selector_info tuple: (scaler_type, selector_obj, selector_type)
      scaler_type:   'standard' | 'minmax'
      selector_type: 'mask' | 'variance_threshold'

    FIX: every return path explicitly casts to np.float64 to prevent
    bracket-string dtype errors such as '[7.258839E-2]' that arise when
    numpy slices a non-contiguous array produced by sklearn's transform().
    """
    scaler_type, selector_obj, selector_type = selector_info

    scaler = fitted_scalers[scaler_type]
    # Force a clean float64 C-contiguous array immediately after transform
    X_scaled = np.ascontiguousarray(scaler.transform(X_raw_df), dtype=np.float64)

    if selector_type == 'mask':
        mask = np.array(selector_obj, dtype=bool)
        if mask.shape[0] != X_scaled.shape[1]:
            raise ValueError(
                f"Mask length {mask.shape[0]} != number of features {X_scaled.shape[1]}. "
                f"Ensure teacher and student datasets have identical feature columns."
            )
        # np.ascontiguousarray ensures no strided/object-dtype artefacts
        return np.ascontiguousarray(X_scaled[:, mask], dtype=np.float64)

    elif selector_type == 'variance_threshold':
        return np.ascontiguousarray(selector_obj.transform(X_scaled), dtype=np.float64)

    else:
        raise ValueError(f"Unknown selector_type: {selector_type}")


def _extract_raw_selected(X_raw_df, selector_info):
    """
    Apply only the column-selection step of a selector to the *unscaled* raw DataFrame.

    Returns a float64 numpy array with the same column subset that
    apply_selector_to_data would select, but preserving original values.

    Used exclusively to supply X_raw to plot_shap_interpretability.
    """
    _, selector_obj, selector_type = selector_info
    X_raw_np = np.ascontiguousarray(X_raw_df.values, dtype=np.float64)

    if selector_type == 'mask':
        mask = np.array(selector_obj, dtype=bool)
        return np.ascontiguousarray(X_raw_np[:, mask], dtype=np.float64)

    elif selector_type == 'variance_threshold':
        # VarianceThreshold.transform works on any numeric array
        return np.ascontiguousarray(selector_obj.transform(X_raw_np), dtype=np.float64)

    else:
        raise ValueError(f"Unknown selector_type: {selector_type}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ml_pipeline_stage1(train_path, test_path, target_col, output_dir, logger):
    """
    Stage 1: Exhaustive ML pipeline.

    Returns
    -------
    results_df        : pd.DataFrame  — all experiment metrics
    trained_models    : dict  — key (fs_name, samp_name, model_name) -> model + preds
    feature_selectors : dict  — key fs_name -> (scaler_type, selector_obj, selector_type)
    fitted_scalers    : dict  — {'standard': StandardScaler, 'minmax': MinMaxScaler}
    """
    logger.info(f"Starting Stage 1: {train_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    drop_cols = [c for c in train_df.columns
                 if c.startswith(("SAMPLE_ID", "ORIG_ID", "Trimester", "Unnamed"))]
    train_df  = train_df.drop(columns=drop_cols, errors='ignore')
    test_df   = test_df.drop(columns=drop_cols,  errors='ignore')

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    dataset_name = Path(train_path).stem.replace('_train', '')
    ds_out       = make_dataset_output_dir(output_dir, dataset_name)

    # ── Scalers (fit only on training data) ───────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train).astype(np.float64)
    X_te_sc = scaler.transform(X_test).astype(np.float64)

    scaler2  = MinMaxScaler()
    X_tr_mm  = scaler2.fit_transform(X_train).astype(np.float64)
    X_te_mm  = scaler2.transform(X_test).astype(np.float64)

    fitted_scalers = {'standard': scaler, 'minmax': scaler2}

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    logger.info("Performing t-SNE analysis...")
    perplexity_values = [5, 10, 20, 30, 50]
    tsne_results = {}
    for perp in perplexity_values:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
        tsne_results[perp] = tsne.fit_transform(X_tr_sc)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    colors = [PALETTE[1] if y == 1 else PALETTE[0] for y in y_train]
    patch0 = mpatches.Patch(color=PALETTE[0], label='Control')
    patch1 = mpatches.Patch(color=PALETTE[1], label='Case')

    for idx, perp in enumerate(perplexity_values):
        emb = tsne_results[perp]
        axes[idx].scatter(emb[:, 0], emb[:, 1], c=colors, s=8, alpha=0.65, linewidths=0)
        axes[idx].set_title(f'Perplexity = {perp}')
        axes[idx].set_xlabel('Dimension 1')
        axes[idx].set_ylabel('Dimension 2')
        axes[idx].grid(True, lw=0.4)

    axes[5].legend(handles=[patch0, patch1], loc='center', fontsize=12)
    axes[5].axis('off')
    fig.suptitle(f'{dataset_name}: t-SNE at Varying Perplexity Values',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_tsne.png'), logger)

    # ── Feature Selection ─────────────────────────────────────────────────────
    logger.info("Performing feature selection...")

    feature_sets      = {}
    feature_selectors = {}
    feature_names_all = list(X_train.columns)

    # 1. Univariate raw p < 0.05
    logger.info("  Univariate raw p<0.05...")
    Xtr_raw, Xte_raw, mask_raw = univariate_raw_pvalue(
        X_tr_sc, y_train.values, X_te_sc)
    feature_sets['Univariate_raw_p05']      = (Xtr_raw, Xte_raw)
    feature_selectors['Univariate_raw_p05'] = ('standard', mask_raw, 'mask')
    logger.info(f"    Features selected: {mask_raw.sum()}")

    # 2. Univariate BH-adjusted p < 0.05
    logger.info("  Univariate BH p<0.05...")
    Xtr_bh, Xte_bh, mask_bh = univariate_bh_pvalue(
        X_tr_sc, y_train.values, X_te_sc)
    feature_sets['Univariate_BH_p05']      = (Xtr_bh, Xte_bh)
    feature_selectors['Univariate_BH_p05'] = ('standard', mask_bh, 'mask')
    logger.info(f"    Features selected: {mask_bh.sum()}")

    # 3. RFECV
    logger.info("  RFECV...")
    Xtr_rfecv, Xte_rfecv, rfecv_mask, rfecv_obj = rf_rfecv_selection(
        X_tr_mm, y_train.values, X_te_mm)
    feature_sets['RF_RFECV']      = (Xtr_rfecv, Xte_rfecv)
    feature_selectors['RF_RFECV'] = ('minmax', rfecv_mask, 'mask')
    logger.info(f"    Features selected: {rfecv_mask.sum()}")

    n_feat_arr  = rfecv_obj.cv_results_['n_features']
    mean_scores = rfecv_obj.cv_results_['mean_test_score']
    std_scores  = rfecv_obj.cv_results_['std_test_score']
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_feat_arr, mean_scores, color=PALETTE[0], lw=2)
    ax.fill_between(n_feat_arr, mean_scores - std_scores, mean_scores + std_scores,
                    alpha=0.2, color=PALETTE[0])
    ax.axvline(rfecv_obj.n_features_, color=PALETTE[1], ls='--', lw=1.5,
               label=f'Optimal: {rfecv_obj.n_features_} features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('CV ROC-AUC')
    ax.set_title(f'{dataset_name}: RFECV — CV ROC-AUC vs. Feature Count')
    ax.legend(); ax.grid(True, lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_rfecv_curve.png'), logger)

    # 4. Sequential top-k
    logger.info("  Sequential top-k...")
    Xtr_topk, Xte_topk, topk_mask, k_vals, k_aucs = rf_sequential_topk(
        X_tr_mm, y_train.values, X_te_mm, logger)
    feature_sets['RF_Sequential_TopK']      = (Xtr_topk, Xte_topk)
    feature_selectors['RF_Sequential_TopK'] = ('minmax', topk_mask, 'mask')
    logger.info(f"    Features selected: {topk_mask.sum()}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_vals, k_aucs, color=PALETTE[2], lw=2, marker='o', markersize=4)
    best_k_idx = int(np.argmax(k_aucs))
    ax.axvline(k_vals[best_k_idx], color=PALETTE[1], ls='--', lw=1.5,
               label=f'Optimal k = {k_vals[best_k_idx]}')
    ax.set_xlabel('Number of Top Features (k)')
    ax.set_ylabel('CV ROC-AUC')
    ax.set_title(f'{dataset_name}: Sequential Top-k Feature Selection')
    ax.legend(); ax.grid(True, lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_topk_curve.png'), logger)

    # 5. Variance threshold via elbow
    logger.info("  Variance threshold (elbow)...")
    variances  = np.var(X_tr_mm, axis=0)
    sorted_var = np.sort(variances)[::-1]
    elbow_idx  = find_elbow_point(sorted_var)
    elbow_thr  = float(sorted_var[elbow_idx])
    var_sel    = VarianceThreshold(threshold=elbow_thr)
    Xtr_var    = var_sel.fit_transform(X_tr_mm).astype(np.float64)
    Xte_var    = var_sel.transform(X_te_mm).astype(np.float64)
    feature_sets['Variance_Elbow']      = (Xtr_var, Xte_var)
    feature_selectors['Variance_Elbow'] = ('minmax', var_sel, 'variance_threshold')
    logger.info(f"    Threshold={elbow_thr:.6f}, Features retained: {Xtr_var.shape[1]}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(sorted_var)), sorted_var, color=PALETTE[0], lw=2)
    ax.scatter([elbow_idx], [elbow_thr], color=PALETTE[1], s=120, zorder=5,
               label=f'Elbow (thr={elbow_thr:.4f}, n={Xtr_var.shape[1]})')
    ax.axhline(elbow_thr, color=PALETTE[1], ls='--', lw=1.2)
    ax.set_xlabel('Feature Rank (Sorted by Variance)')
    ax.set_ylabel('Variance')
    ax.set_title(f'{dataset_name}: Elbow Detection for Variance Threshold')
    ax.legend(); ax.grid(True, lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_variance_elbow.png'), logger)

     # ── Save selected feature names per FS method to CSV ─────────────────
    logger.info("  Saving selected feature names per feature selector...")
    fs_feat_rows = []
    for fs_key, (scaler_type, selector_obj, selector_type) in feature_selectors.items():
        if selector_type == 'mask':
            mask = np.array(selector_obj, dtype=bool)
            selected = [feature_names_all[i] for i, m in enumerate(mask) if m]
        elif selector_type == 'variance_threshold':
            vt_mask = selector_obj.get_support()
            selected = [feature_names_all[i] for i, m in enumerate(vt_mask) if m]
        else:
            selected = []
        for feat in selected:
            fs_feat_rows.append({'FeatureSelector': fs_key, 'FeatureName': feat})
    fs_feat_df = pd.DataFrame(fs_feat_rows)
    fs_feat_csv = os.path.join(ds_out, f'{dataset_name}_selected_features.csv')
    fs_feat_df.to_csv(fs_feat_csv, index=False)
    logger.info(f"    Feature name CSV saved: {fs_feat_csv}")

    # Summary: feature count per method
    feat_count = fs_feat_df.groupby('FeatureSelector').size().reset_index(name='n_features')
    fig, ax = plt.subplots(figsize=(8, max(3, len(feat_count) * 0.55)))
    ax.barh(feat_count['FeatureSelector'], feat_count['n_features'],
            color=PALETTE[0], alpha=0.8)
    for i, row in enumerate(feat_count.itertuples()):
        ax.text(row.n_features + 0.3, i, str(row.n_features), va='center', fontsize=9)
    ax.set_xlabel('Number of Selected Features')
    ax.set_title(f'{dataset_name}: Features Selected per Method')
    ax.grid(axis='x', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_feature_counts.png'), logger)

    # 6. PCA 95% — Stage 1 only, intentionally excluded from feature_selectors
    logger.info("  PCA 95% variance...")
    pca = PCA(n_components=0.95, random_state=42)
    Xtr_pca = pca.fit_transform(X_tr_sc).astype(np.float64)
    Xte_pca = pca.transform(X_te_sc).astype(np.float64)
    feature_sets['PCA_95'] = (Xtr_pca, Xte_pca)
    # NOTE: PCA is intentionally NOT added to feature_selectors dict.

    # ── Sampling configurations ───────────────────────────────────────────────
    SAMPLING_CONFIGS = {
        'RandomOversample_NoWeight':  ('oversample',  None),
        'RandomUndersample_NoWeight': ('undersample', None),
        'SMOTE_NoWeight':             ('smote',       None),
        'NoSampling_Balanced':        ('none',        'balanced'),
        'NoSampling_NoWeight':        ('none',        None),
    }

    # ── Model definitions ─────────────────────────────────────────────────────
    def get_models(cw):
        pos_weight = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1.0)
        return {
            'XGBoost': {
                'model': XGBClassifier(
                    random_state=42, eval_metric='logloss',
                    use_label_encoder=False, tree_method='hist'),
                'params': {
                    'n_estimators':    [100, 200, 500],
                    'max_depth':       [3, 5, 7],
                    'learning_rate':   [0.01, 0.1],
                    'scale_pos_weight':[1] if cw is None else [pos_weight]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators':      [100, 200, 500],
                    'max_depth':         [10, 20, None],
                    'min_samples_split': [2, 5],
                    'class_weight':      [cw]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C':            [0.1, 1, 10],
                    'kernel':       ['rbf', 'linear'],
                    'class_weight': [cw]
                }
            },
            'GaussianNB': {
                'model': GaussianNB(),
                'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C':            [0.1, 1, 10],
                    'penalty':      ['l2'],
                    'class_weight': [cw]
                }
            }
        }

    # ── Experiment loop ───────────────────────────────────────────────────────
    logger.info("Running experiments...")
    results        = []
    trained_models = {}
    total          = len(feature_sets) * len(SAMPLING_CONFIGS) * 5
    exp_count      = 0

    for fs_name, (X_tr_fs, X_te_fs) in feature_sets.items():
        for samp_name, (samp_type, cw) in SAMPLING_CONFIGS.items():

            Xtr_s = X_tr_fs.copy()
            ytr_s = y_train.values.copy()

            if samp_type == 'oversample':
                Xtr_s, ytr_s = random_oversample(Xtr_s, ytr_s)
            elif samp_type == 'undersample':
                Xtr_s, ytr_s = random_undersample(Xtr_s, ytr_s)
            elif samp_type == 'smote':
                Xtr_s, ytr_s = smote_sample(Xtr_s, ytr_s)

            models = get_models(cw)
            for model_name, cfg in models.items():
                exp_count += 1
                logger.info(f"  [{exp_count}/{total}] {fs_name} | {samp_name} | {model_name}")
                try:
                    gs = GridSearchCV(
                        cfg['model'], cfg['params'],
                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                        scoring='f1', n_jobs=-1, verbose=0
                    )
                    gs.fit(Xtr_s, ytr_s)
                    best_model = gs.best_estimator_

                    y_pred  = best_model.predict(X_te_fs)
                    y_proba = best_model.predict_proba(X_te_fs)[:, 1].astype(np.float64)

                    m = compute_all_metrics_with_ci(y_test.values, y_proba, y_pred)

                    results.append({
                        'FeatureSelection': fs_name,
                        'Sampling':         samp_name,
                        'Model':            model_name,
                        'ROC_AUC':          m['roc_auc'],
                        'ROC_AUC_lo':       m['roc_auc_lo'],
                        'ROC_AUC_hi':       m['roc_auc_hi'],
                        'PR_AUC':           m['pr_auc'],
                        'AP_Score':         m['ap_score'],
                        'AP_Score_lo':      m['ap_score_lo'],
                        'AP_Score_hi':      m['ap_score_hi'],
                        'Precision':        m['precision'],
                        'Precision_lo':     m['precision_lo'],
                        'Precision_hi':     m['precision_hi'],
                        'Recall':           m['recall'],
                        'Recall_lo':        m['recall_lo'],
                        'Recall_hi':        m['recall_hi'],
                        'F1_Score':         m['f1'],
                        'F1_Score_lo':      m['f1_lo'],
                        'F1_Score_hi':      m['f1_hi'],
                        'BestParams':       str(gs.best_params_)
                    })

                    trained_models[(fs_name, samp_name, model_name)] = {
                        'model':   best_model,
                        'y_proba': y_proba,
                        'y_pred':  y_pred,
                    }

                except Exception as e:
                    logger.error(f"    Error: {e}")
                    nan_row = {
                        'FeatureSelection': fs_name, 'Sampling': samp_name,
                        'Model': model_name, 'BestParams': 'Error'
                    }
                    for k in ['ROC_AUC','ROC_AUC_lo','ROC_AUC_hi','PR_AUC',
                              'AP_Score','AP_Score_lo','AP_Score_hi',
                              'Precision','Precision_lo','Precision_hi',
                              'Recall','Recall_lo','Recall_hi',
                              'F1_Score','F1_Score_lo','F1_Score_hi']:
                        nan_row[k] = np.nan
                    results.append(nan_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(ds_out, f'{dataset_name}_stage1_results.csv'), index=False)
    logger.info(f"Results saved: {dataset_name}_stage1_results.csv")

    # ── Visualizations ────────────────────────────────────────────────────────
    logger.info("Generating visualizations...")
    valid = results_df.dropna(subset=['ROC_AUC'])
    metrics_cols  = ['ROC_AUC', 'Precision', 'Recall', 'F1_Score']
    metric_labels = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']

    fig, ax = plt.subplots(figsize=(9, 5))
    vp = ax.violinplot([valid[m].dropna().values for m in metrics_cols],
                       positions=range(len(metrics_cols)), widths=0.6,
                       showmedians=True, showextrema=False)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(PALETTE[i % len(PALETTE)])
        body.set_alpha(0.7)
    vp['cmedians'].set_color('black')
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name}: Performance Distribution Across All Configurations')
    ax.grid(axis='y', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_violin.png'), logger)

    model_summary = (valid.groupby('Model')
                     .agg(mean_auc=('ROC_AUC','mean'),
                          lo_auc=('ROC_AUC_lo','mean'),
                          hi_auc=('ROC_AUC_hi','mean'))
                     .reset_index()
                     .sort_values('mean_auc', ascending=True))
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in enumerate(model_summary.itertuples()):
        ax.errorbar(row.mean_auc, i,
                    xerr=[[row.mean_auc - row.lo_auc], [row.hi_auc - row.mean_auc]],
                    fmt='o', color=PALETTE[i % len(PALETTE)], markersize=8,
                    capsize=4, lw=1.5)
    ax.set_yticks(range(len(model_summary)))
    ax.set_yticklabels(model_summary['Model'])
    ax.set_xlabel('Mean ROC-AUC (95% CI)')
    ax.set_title(f'{dataset_name}: ROC-AUC by Model')
    ax.axvline(0.5, color='grey', ls='--', lw=1)
    ax.grid(axis='x', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_model_ci.png'), logger)

    pivot = (valid.groupby(['FeatureSelection', 'Sampling'])['ROC_AUC']
             .mean().unstack(fill_value=np.nan))
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f',
                cmap=sns.color_palette("Blues", as_cmap=True), ax=ax,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Mean ROC-AUC', 'shrink': 0.8})
    ax.set_title(f'{dataset_name}: Mean ROC-AUC — Feature Selection × Sampling Strategy')
    ax.set_xlabel('Sampling Strategy')
    ax.set_ylabel('Feature Selection Method')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_heatmap.png'), logger)

    top10 = valid.nlargest(10, 'ROC_AUC').copy()
    top10['Config'] = (top10['Model'] + ' / ' +
                       top10['FeatureSelection'] + ' / ' +
                       top10['Sampling'].str.split('_').str[0])
    top10 = top10.sort_values('ROC_AUC')
    fig, ax = plt.subplots(figsize=(10, 7))
    ys = range(len(top10))
    ax.barh(list(ys), top10['ROC_AUC'], color=PALETTE[0], alpha=0.75, height=0.6)
    ax.errorbar(top10['ROC_AUC'].values, list(ys),
                xerr=[top10['ROC_AUC'].values - top10['ROC_AUC_lo'].values,
                      top10['ROC_AUC_hi'].values - top10['ROC_AUC'].values],
                fmt='none', color='black', capsize=3, lw=1.2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(top10['Config'], fontsize=8)
    ax.set_xlabel('ROC-AUC (95% CI)')
    ax.set_title(f'{dataset_name}: Top 10 Configurations by ROC-AUC')
    for i, (v, hi) in enumerate(zip(top10['ROC_AUC'], top10['ROC_AUC_hi'])):
        ax.text(hi + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
    ax.set_xlim(0, min(1.05, top10['ROC_AUC_hi'].max() + 0.06))
    ax.grid(axis='x', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_top10_ci.png'), logger)

    samp_summary = (valid.groupby('Sampling')[metrics_cols]
                    .mean().reset_index().sort_values('ROC_AUC', ascending=False))
    x     = np.arange(len(samp_summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, lbl) in enumerate(zip(metrics_cols, metric_labels)):
        ax.bar(x + i * width, samp_summary[col], width, label=lbl,
               color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(samp_summary['Sampling'], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name}: Performance by Sampling Strategy')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_sampling_grouped.png'), logger)

    fs_summary = (valid.groupby('FeatureSelection')[metrics_cols]
                  .mean().reset_index().sort_values('ROC_AUC', ascending=False))
    x = np.arange(len(fs_summary))
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (col, lbl) in enumerate(zip(metrics_cols, metric_labels)):
        ax.bar(x + i * width, fs_summary[col], width, label=lbl,
               color=PALETTE[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(fs_summary['FeatureSelection'], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title(f'{dataset_name}: Performance by Feature Selection Method')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'{dataset_name}_stage1_featuresel_grouped.png'), logger)

    # ── Per-config detailed plots ──────────────────────────────────────────────
    logger.info("Generating per-config bar plots, ROC curves, and confusion matrices...")

    from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

    for key, data in trained_models.items():
        fs_name_k, samp_name_k, model_name_k = key

        row = valid[
            (valid['Model'] == model_name_k) &
            (valid['FeatureSelection'] == fs_name_k) &
            (valid['Sampling'] == samp_name_k)
        ]
        if row.empty:
            continue

        row = row.iloc[0]
        config_tag = f'{model_name_k}__{fs_name_k}__{samp_name_k}'
        config_title = f'{model_name_k} | {fs_name_k} | {samp_name_k}'

        # ── 1. Bar plot ────────────────────────────────────────────────────────
        metrics = {
            'ROC-AUC':   (row['ROC_AUC'],   row['ROC_AUC_lo'],   row['ROC_AUC_hi']),
            'Precision': (row['Precision'], row['Precision_lo'], row['Precision_hi']),
            'Recall':    (row['Recall'],    row['Recall_lo'],    row['Recall_hi']),
            'F1-Score':  (row['F1_Score'],  row['F1_Score_lo'],  row['F1_Score_hi']),
        }
        vals   = [v[0] for v in metrics.values()]
        errs_lo = [max(0, v[0] - v[1]) for v in metrics.values()]
        errs_hi = [max(0, v[2] - v[0]) for v in metrics.values()]

        fig, ax = plt.subplots(figsize=(7, 4))
        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, vals, yerr=[errs_lo, errs_hi],
                    color=PALETTE[:4], alpha=0.85, width=0.5,
                    capsize=5, error_kw={'lw': 1.5})
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(metrics.keys()), fontsize=11)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(f'{dataset_name}: {config_title}\nMetrics with 95% CI', fontsize=9)
        ax.grid(axis='y', lw=0.4)
        plt.tight_layout()
        save_figure(fig, os.path.join(ds_out, f'{dataset_name}_{config_tag}_bar.png'), logger)

        # ── 2. ROC curve ──────────────────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test.values, data['y_proba'])
        auc_val = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color=PALETTE[1],
                label=f'AUC = {auc_val:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{dataset_name}: {config_title}\nROC Curve', fontsize=9)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(lw=0.4)
        plt.tight_layout()
        save_figure(fig, os.path.join(ds_out, f'{dataset_name}_{config_tag}_roc.png'), logger)

        # ── 3. Confusion matrix ────────────────────────────────────────────────
        cm = confusion_matrix(y_test.values, data['y_pred'])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=['Control', 'PTB']).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'{dataset_name}: {config_title}\nConfusion Matrix', fontsize=9)
        plt.tight_layout()
        save_figure(fig, os.path.join(ds_out, f'{dataset_name}_{config_tag}_cm.png'), logger)

    best = valid.loc[valid['ROC_AUC'].idxmax()]
    logger.info(f"\nBest ({dataset_name}): {best['Model']} | "
                f"{best['FeatureSelection']} | {best['Sampling']}")
    logger.info(f"  ROC-AUC: {best['ROC_AUC']:.4f} "
                f"[{best['ROC_AUC_lo']:.4f}–{best['ROC_AUC_hi']:.4f}]")

    return results_df, trained_models, feature_selectors, fitted_scalers


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_best_stage1_models(stage1_results, logger):
    logger.info("Extracting best Stage 1 models (excluding PCA)...")
    best_per_dataset = {}

    for ds, data in stage1_results.items():
        results_df        = data['results_df']
        trained_models    = data['trained_models']
        feature_selectors = data['feature_selectors']
        fitted_scalers    = data['fitted_scalers']

        valid = results_df[
            (~results_df['FeatureSelection'].str.startswith('PCA')) &
            (results_df['ROC_AUC'].notna())
        ].copy()

        if valid.empty:
            logger.warning(f"  {ds}: No valid non-PCA configs — skipping")
            continue

        best_row   = valid.loc[valid['ROC_AUC'].idxmax()]
        fs_name    = best_row['FeatureSelection']
        samp_name  = best_row['Sampling']
        model_name = best_row['Model']
        key        = (fs_name, samp_name, model_name)

        if key not in trained_models:
            logger.warning(f"  {ds}: Trained model not found for {key} — skipping")
            continue

        if fs_name not in feature_selectors:
            logger.warning(f"  {ds}: Selector not found for '{fs_name}' — skipping")
            continue

        logger.info(f"  {ds}: {model_name} | {fs_name} | {samp_name} "
                    f"| ROC-AUC={best_row['ROC_AUC']:.4f}")

        best_per_dataset[ds] = {
            'model':                 trained_models[key]['model'],
            'feature_selector_info': feature_selectors[fs_name],
            'fitted_scalers':        fitted_scalers,
            'fs_name':               fs_name,
            'metrics_row':           best_row,
        }

    return best_per_dataset


def plot_shap_interpretability(model, X_input, X_raw, feature_names, output_dir, prefix, logger):
    """
    Comprehensive SHAP interpretability plots.

    Parameters
    ----------
    model         : fitted XGBRegressor
    X_input       : float64 numpy array (n_samples, n_features)
                    Scaled/selected values fed to the model — used to COMPUTE SHAP values.
    X_raw         : float64 numpy array (n_samples, n_features), same shape as X_input
                    Original unscaled values — used for DISPLAY in all plots so that
                    axis ticks show true metabolite concentrations.
    feature_names : list[str], length == n_features
    output_dir    : str
    prefix        : str
    logger        : logging.Logger
    """
    try:
        # ── Sanitise model input — guarantee clean float64 C-contiguous arrays ──
        X_arr     = np.ascontiguousarray(X_input, dtype=np.float64)
        X_raw_arr = np.ascontiguousarray(X_raw,   dtype=np.float64)

        if X_arr.shape != X_raw_arr.shape:
            raise ValueError(
                f"X_input shape {X_arr.shape} != X_raw shape {X_raw_arr.shape}. "
                "Both must have identical dimensions."
            )

        explainer    = shap.TreeExplainer(model)
        shap_values  = explainer.shap_values(X_arr)   # computed on scaled input
        def _parse_ev(ev):
            """Robustly convert SHAP expected_value to a plain Python float.

            XGBoost can return expected_value in several forms:
              - a plain float / numpy scalar
              - a string such as '0.083' or '[8.3647944E-2]'  (bracket notation)
              - a list or numpy array whose first element may itself be any of
                the above (including a bracket-string)
            """
            if isinstance(ev, str):
                return float(ev.strip('[]'))
            if isinstance(ev, (list, np.ndarray)):
                # Recurse so bracket-string elements inside arrays are handled too
                return _parse_ev(ev[0])
            return float(ev)

        expected_val = _parse_ev(explainer.expected_value)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.array(shap_values, dtype=np.float64)

        top_idx   = np.argsort(np.abs(shap_values).mean(0))[::-1][:20]
        top_names = [feature_names[i] for i in top_idx]
        shap_top  = shap_values[:, top_idx]
        X_raw_top = X_raw_arr[:, top_idx]   # raw values for top features

        # ── 1. Beeswarm — axis values are raw metabolite concentrations ───────
        plt.figure(figsize=(9, 7))
        shap.summary_plot(shap_top, X_raw_top, feature_names=top_names,
                          plot_type='dot', show=False, max_display=20)
        plt.title('SHAP Beeswarm — Top 20 Features (raw feature values)',
                  fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_figure(plt.gcf(),
                    os.path.join(output_dir, f'{prefix}_shap_beeswarm.png'), logger)

        # ── 2. Bar — mean |SHAP| ─────────────────────────────────────────────
        mean_abs = np.abs(shap_values).mean(0)
        order    = np.argsort(mean_abs)[::-1][:20]
        fig, ax  = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(order)), mean_abs[order], color=PALETTE[0], alpha=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title('SHAP Feature Importance (Mean |SHAP|)')
        ax.grid(axis='x', lw=0.4)
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, f'{prefix}_shap_bar.png'), logger)

        # ── 3. Dependence plots (top 3) — x-axis shows raw values ────────────
        for feat in [feature_names[i] for i in order[:3]]:
            fidx = feature_names.index(feat)
            fig, ax = plt.subplots(figsize=(7, 5))
            # Pass X_raw_arr so the x-axis of the dependence plot reflects
            # true metabolite concentrations, not scaled values.
            shap.dependence_plot(fidx, shap_values, X_raw_arr,
                                 feature_names=feature_names,
                                 ax=ax, show=False, dot_size=20, alpha=0.6)
            ax.set_title(f'SHAP Dependence: {feat} (raw values)')
            ax.set_xlabel(f'{feat} (raw)')
            ax.grid(lw=0.4)
            plt.tight_layout()
            safe = feat.replace('/', '_').replace(' ', '_')[:40]
            save_figure(fig,
                        os.path.join(output_dir, f'{prefix}_shap_dep_{safe}.png'), logger)

        # ── 4. Decision plot (sample of up to 200) ───────────────────────────
        try:
            rng_idx = np.random.RandomState(42).choice(
                len(X_arr), size=min(200, len(X_arr)), replace=False)
            shap.decision_plot(
                expected_val,
                shap_values[rng_idx][:, order[:15]],
                feature_names=[feature_names[i] for i in order[:15]],
                show=False, link='identity')
            plt.title('SHAP Decision Plot — Top 15 Features')
            plt.tight_layout()
            save_figure(plt.gcf(),
                        os.path.join(output_dir, f'{prefix}_shap_decision.png'), logger)
        except Exception as e:
            logger.warning(f"  Decision plot skipped: {e}")

        # ── 5. SHAP heatmap ───────────────────────────────────────────────────
        try:
            rng_idx   = np.random.RandomState(42).choice(
                len(X_arr), size=min(200, len(X_arr)), replace=False)
            shap_heat = shap_values[rng_idx][:, order[:15]]
            feat_heat = [feature_names[i] for i in order[:15]]
            fig, ax   = plt.subplots(figsize=(12, 6))
            vmax      = np.abs(shap_heat).max()
            im = ax.imshow(shap_heat.T, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
            ax.set_yticks(range(len(feat_heat)))
            ax.set_yticklabels(feat_heat, fontsize=8)
            ax.set_xlabel('Sample Index')
            ax.set_title('SHAP Value Heatmap — Top 15 Features')
            plt.colorbar(im, ax=ax, label='SHAP value', fraction=0.03)
            plt.tight_layout()
            save_figure(fig,
                        os.path.join(output_dir, f'{prefix}_shap_heatmap.png'), logger)
        except Exception as e:
            logger.warning(f"  Heatmap skipped: {e}")

        # ── 6. Waterfall — highest / lowest confidence ────────────────────────
        # data= uses raw values so displayed feature values are interpretable
        try:
            raw_preds = np.clip(
                np.array(model.predict(X_arr), dtype=np.float64), 0, 1)
            for label, sample_i in [
                ('highest_conf', int(np.argmax(raw_preds))),
                ('lowest_conf',  int(np.argmin(raw_preds)))
            ]:
                exp_obj = shap.Explanation(
                    values=shap_values[sample_i].astype(np.float64),
                    base_values=expected_val,
                    data=X_raw_arr[sample_i],   # raw values shown on waterfall axis
                    feature_names=feature_names
                )
                shap.waterfall_plot(exp_obj, max_display=15, show=False)
                plt.title(f'SHAP Waterfall — {label.replace("_", " ").title()} (raw values)')
                plt.tight_layout()
                save_figure(plt.gcf(),
                            os.path.join(output_dir,
                                         f'{prefix}_shap_waterfall_{label}.png'),
                            logger)
        except Exception as e:
            logger.warning(f"  Waterfall plot skipped: {e}")

    except Exception as e:
        logger.error(f"  SHAP interpretability failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: KNOWLEDGE DISTILLATION
# ══════════════════════════════════════════════════════════════════════════════

def knowledge_distillation_stage2(teacher_ds, student_ds,
                                   teacher_info, student_info,
                                   target_col, output_dir, logger):
    """
    Two-branch knowledge distillation using best Stage 1 models.

    Branch 1: Student regressor trained on teacher's feature space.
    Branch 2: Student regressor trained on union of teacher + student feature spaces.
    """
    logger.info(f"\nKD: {teacher_ds} -> {student_ds}")

    # ── Load raw data ─────────────────────────────────────────────────────────
    t_train = pd.read_csv(f'{teacher_ds}_train.csv')
    t_test  = pd.read_csv(f'{teacher_ds}_test.csv')
    s_train = pd.read_csv(f'{student_ds}_train.csv')
    s_test  = pd.read_csv(f'{student_ds}_test.csv')

    drop_cols = [c for c in t_train.columns
                 if c.startswith(("SAMPLE_ID", "ORIG_ID", "Trimester", "Unnamed"))]
    for df in [t_train, t_test, s_train, s_test]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

    t_feat_cols = [c for c in t_train.columns if c != target_col]
    s_feat_cols = [c for c in s_train.columns if c != target_col]
    assert t_feat_cols == s_feat_cols, (
        f"Feature column mismatch between {teacher_ds} and {student_ds}. "
        f"Teacher has {len(t_feat_cols)} features, student has {len(s_feat_cols)}."
    )

    Xtt_raw = t_train[t_feat_cols];  ytt = t_train[target_col]
    Xte_raw = t_test[t_feat_cols];   yte = t_test[target_col]
    Xst_raw = s_train[s_feat_cols];  yst = s_train[target_col]
    Xse_raw = s_test[s_feat_cols];   yse = s_test[target_col]

    # ── Project through teacher's selector (scaled) ───────────────────────────
    t_sel     = teacher_info['feature_selector_info']
    t_scalers = teacher_info['fitted_scalers']

    Xtt_t = apply_selector_to_data(Xtt_raw, t_sel, t_scalers)
    Xte_t = apply_selector_to_data(Xte_raw, t_sel, t_scalers)
    Xst_t = apply_selector_to_data(Xst_raw, t_sel, t_scalers)
    Xse_t = apply_selector_to_data(Xse_raw, t_sel, t_scalers)

    # ── Project through student's own selector (scaled) ───────────────────────
    s_sel     = student_info['feature_selector_info']
    s_scalers = student_info['fitted_scalers']

    Xst_s = apply_selector_to_data(Xst_raw, s_sel, s_scalers)
    Xse_s = apply_selector_to_data(Xse_raw, s_sel, s_scalers)

    logger.info(f"  Teacher train shape: {Xtt_t.shape}")
    logger.info(f"  Student train — teacher FS: {Xst_t.shape}")
    logger.info(f"  Student train — student FS: {Xst_s.shape}")

    assert Xst_t.shape[0] == Xst_s.shape[0], (
        f"Row count mismatch: Xst_t has {Xst_t.shape[0]} rows, "
        f"Xst_s has {Xst_s.shape[0]} rows."
    )

    # ── Raw (unscaled) arrays for SHAP display ────────────────────────────────
    # These mirror the column selection of the scaled arrays but preserve
    # original metabolite concentrations so SHAP plot axes are interpretable.
    Xst_raw_b1 = _extract_raw_selected(Xst_raw, t_sel)   # Branch 1: teacher cols only
    # Branch 2: union of teacher-selected cols + student-selected cols (unscaled)
    Xst_raw_s  = _extract_raw_selected(Xst_raw, s_sel)
    Xst_raw_b2 = np.hstack([Xst_raw_b1, Xst_raw_s])

    logger.info(f"  Raw arrays for SHAP — B1: {Xst_raw_b1.shape}, B2: {Xst_raw_b2.shape}")

    teacher_model = teacher_info['model']

    # ── Teacher soft probabilities on student training data ───────────────────
    teacher_probs_st = np.array(
        teacher_model.predict_proba(Xst_t)[:, 1], dtype=np.float64)

    teacher_auc = roc_auc_score(
        yte, teacher_model.predict_proba(Xte_t)[:, 1])
    logger.info(f"  Teacher ROC-AUC (own test set): {teacher_auc:.4f}")

    # ── Tune alpha via CV on student training data ────────────────────────────
    logger.info("  Tuning alpha...")
    alpha_values = [0.3, 0.5, 0.7]
    best_alpha, best_alpha_score = 0.5, -np.inf
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for alpha in alpha_values:
        cv_scores = []
        for tr_i, va_i in cv.split(Xst_t, yst):
            soft_cv = ((1 - alpha) * yst.iloc[tr_i].values.astype(np.float64)
                       + alpha * teacher_probs_st[tr_i])
            reg = XGBRegressor(
                objective='reg:squarederror', n_estimators=200,
                max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
            reg.fit(Xst_t[tr_i], soft_cv)
            preds = np.clip(
                np.array(reg.predict(Xst_t[va_i]), dtype=np.float64), 0, 1)
            try:
                cv_scores.append(roc_auc_score(yst.iloc[va_i], preds))
            except Exception:
                cv_scores.append(0.5)
        mean_s = float(np.mean(cv_scores))
        logger.info(f"    alpha={alpha}: CV AUC={mean_s:.4f}")
        if mean_s > best_alpha_score:
            best_alpha_score = mean_s
            best_alpha = alpha

    logger.info(f"  Best alpha={best_alpha}")
    soft_labels = ((1 - best_alpha) * yst.values.astype(np.float64)
                   + best_alpha * teacher_probs_st)

    # ── Student regressor hyperparameter grid ─────────────────────────────────
    reg_param_grid = {
        'n_estimators':    [300, 500, 700],
        'max_depth':       [4, 6, 8],
        'learning_rate':   [0.01, 0.05, 0.1],
        'subsample':       [0.7, 0.8, 0.9],
        'colsample_bytree':[0.7, 0.8, 0.9]
    }

    def train_student_regressor(X_train_br, y_soft, X_test_br, y_test_br, branch_label):
        X_train_br = np.ascontiguousarray(X_train_br, dtype=np.float64)
        X_test_br  = np.ascontiguousarray(X_test_br,  dtype=np.float64)
        y_soft     = np.array(y_soft, dtype=np.float64)

        logger.info(f"  [{branch_label}] n_train={X_train_br.shape[0]}, "
                    f"n_feat={X_train_br.shape[1]}")

        reg_base = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        reg_gs   = GridSearchCV(
            reg_base, reg_param_grid,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
        )
        reg_gs.fit(X_train_br, y_soft)
        student_reg = reg_gs.best_estimator_
        logger.info(f"  [{branch_label}] Best params: {reg_gs.best_params_}")

        y_proba  = np.clip(
            np.array(student_reg.predict(X_test_br), dtype=np.float64), 0, 1)
        best_thr = youden_threshold(y_test_br.values, y_proba)
        y_pred   = (y_proba >= best_thr).astype(int)

        metrics = compute_all_metrics_with_ci(y_test_br.values, y_proba, y_pred)
        metrics['threshold']   = best_thr
        metrics['best_params'] = reg_gs.best_params_

        logger.info(f"  [{branch_label}] ROC-AUC={metrics['roc_auc']:.4f} "
                    f"[{metrics['roc_auc_lo']:.4f}–{metrics['roc_auc_hi']:.4f}]  "
                    f"F1={metrics['f1']:.4f}  Recall={metrics['recall']:.4f}")
        return student_reg, metrics

    # ── Branch 1: Teacher feature space only ──────────────────────────────────
    logger.info("  Branch 1: Teacher feature space")
    student_b1, metrics_b1 = train_student_regressor(
        Xst_t, soft_labels, Xse_t, yse, 'Branch1-TeacherFS')

    # ── Branch 2: Union feature space ─────────────────────────────────────────
    logger.info("  Branch 2: Union feature space")
    Xst_union_raw = np.hstack([Xst_t, Xst_s])
    Xse_union_raw = np.hstack([Xse_t, Xse_s])

    union_scaler = StandardScaler()
    Xst_union    = union_scaler.fit_transform(Xst_union_raw).astype(np.float64)
    Xse_union    = union_scaler.transform(Xse_union_raw).astype(np.float64)

    student_b2, metrics_b2 = train_student_regressor(
        Xst_union, soft_labels, Xse_union, yse, 'Branch2-UnionFS')

    # ── Per-KD-pair output folder ─────────────────────────────────────────────
    kd_label = f'{teacher_ds}_to_{student_ds}'
    ds_out   = make_dataset_output_dir(output_dir, f'KD_{kd_label}')

    # ── SHAP for Branch 1 ─────────────────────────────────────────────────────
    # X_input = scaled teacher-selected values (what the model was trained on)
    # X_raw   = unscaled teacher-selected values (for interpretable plot axes)
    try:
        _, t_selector_obj, t_selector_type = t_sel
        if t_selector_type == 'mask':
            t_mask = np.array(t_selector_obj, dtype=bool)
            b1_feat_names = [s_feat_cols[i] for i, m in enumerate(t_mask) if m]
        elif t_selector_type == 'variance_threshold':
            t_mask = t_selector_obj.get_support()
            b1_feat_names = [s_feat_cols[i] for i, m in enumerate(t_mask) if m]
        else:
            b1_feat_names = [f'T_{i}' for i in range(Xst_t.shape[1])]
        plot_shap_interpretability(
            student_b1,
            X_input=Xst_t,
            X_raw=Xst_raw_b1,
            feature_names=b1_feat_names,
            output_dir=ds_out,
            prefix=f'branch1_{student_ds}',
            logger=logger)
    except Exception as e:
        logger.error(f"  Branch 1 SHAP failed: {e}")

    # ── SHAP for Branch 2 ─────────────────────────────────────────────────────
    # X_input = union-rescaled values (what the model was trained on)
    # X_raw   = unscaled union values (teacher cols + student cols concatenated)
    try:
        # Extract actual feature names for union space (teacher cols + student cols)
        _, s_selector_obj, s_selector_type = s_sel
        if s_selector_type == 'mask':
            s_mask = np.array(s_selector_obj, dtype=bool)
            s_feat_names_selected = [s_feat_cols[i] for i, m in enumerate(s_mask) if m]
        elif s_selector_type == 'variance_threshold':
            s_mask = s_selector_obj.get_support()
            s_feat_names_selected = [s_feat_cols[i] for i, m in enumerate(s_mask) if m]
        else:
            s_feat_names_selected = [f'S_{i}' for i in range(Xst_s.shape[1])]

        # Prefix duplicates that appear in both teacher and student selections
        t_set = set(b1_feat_names)
        s_feat_names_prefixed = [
            f'S_{n}' if n in t_set else n
            for n in s_feat_names_selected
        ]
        b2_feat_names = b1_feat_names + s_feat_names_prefixed
        plot_shap_interpretability(
            student_b2,
            X_input=Xst_union,
            X_raw=Xst_raw_b2,
            feature_names=b2_feat_names,
            output_dir=ds_out,
            prefix=f'branch2_{student_ds}',
            logger=logger)
    except Exception as e:
        logger.error(f"  Branch 2 SHAP failed: {e}")

    # ── ROC + PR comparison: Base vs B1 vs B2 ──────────────────────────────
    student_base_model  = student_info['model']
    student_base_probs  = np.array(
        student_base_model.predict_proba(Xse_s)[:, 1], dtype=np.float64)
    b1_proba_test = np.clip(
        np.array(student_b1.predict(Xse_t),     dtype=np.float64), 0, 1)
    b2_proba_test = np.clip(
        np.array(student_b2.predict(Xse_union), dtype=np.float64), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for probs, label, color, ls in [
        (student_base_probs, f'Base ({student_ds})',     PALETTE[0], '--'),
        (b1_proba_test,    'Student B1 (Teacher FS)',     PALETTE[1], '-'),
        (b2_proba_test,    'Student B2 (Union FS)',        PALETTE[2], '-'),
    ]:
        fpr_, tpr_, _  = roc_curve(yse, probs)
        auc_           = roc_auc_score(yse, probs)
        prec_, rec_, _ = precision_recall_curve(yse, probs)
        ap_            = average_precision_score(yse, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=2, ls=ls,
                     label=f'{label}  AUC={auc_:.3f}')
        axes[1].plot(rec_,  prec_, color=color, lw=2, ls=ls,
                     label=f'{label}  AP={ap_:.3f}')

    axes[0].plot([0,1],[0,1],'k--',lw=0.8,alpha=0.4)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend(fontsize=8); axes[0].grid(lw=0.4)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(fontsize=8); axes[1].grid(lw=0.4)
    fig.suptitle(f'Knowledge Distillation: {teacher_ds} → {student_ds}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'KD_{kd_label}_roc_pr.png'), logger)

    # ── Confusion matrices ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in [
        (axes[0], metrics_b1, 'Branch 1 (Teacher FS)'),
        (axes[1], metrics_b2, 'Branch 2 (Union FS)'),
    ]:
        cm = np.array(m['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred 0','Pred 1'],
                    yticklabels=['True 0','True 1'],
                    linewidths=0.5, cbar=False)
        ax.set_title(title)
    fig.suptitle(f'Confusion Matrices — KD {teacher_ds} → {student_ds}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'KD_{kd_label}_confusion.png'), logger)

    # ── Branch comparison bar chart ───────────────────────────────────────────
    metric_keys      = ['roc_auc', 'precision', 'recall', 'f1']
    metric_labels_kd = ['ROC-AUC', 'Precision', 'Recall', 'F1']
    x     = np.arange(len(metric_keys))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    b1_vals = [metrics_b1[k] for k in metric_keys]
    b2_vals = [metrics_b2[k] for k in metric_keys]
    b1_errs = [
        [max(0,metrics_b1[k] - metrics_b1.get(f'{k}_lo', metrics_b1[k])) for k in metric_keys],
        [max(0,metrics_b1.get(f'{k}_hi', metrics_b1[k]) - metrics_b1[k]) for k in metric_keys]
    ]
    b2_errs = [
        [max(0,metrics_b2[k] - metrics_b2.get(f'{k}_lo', metrics_b2[k])) for k in metric_keys],
        [max(0,metrics_b2.get(f'{k}_hi', metrics_b2[k]) - metrics_b2[k]) for k in metric_keys]
    ]
    ax.bar(x - width/2, b1_vals, width, label='Branch 1 (Teacher FS)',
           color=PALETTE[1], alpha=0.85, yerr=b1_errs, capsize=4,
           error_kw={'lw': 1.2})
    ax.bar(x + width/2, b2_vals, width, label='Branch 2 (Union FS)',
           color=PALETTE[2], alpha=0.85, yerr=b2_errs, capsize=4,
           error_kw={'lw': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels_kd)
    ax.set_ylabel('Score (95% CI)')
    ax.set_title(f'KD {teacher_ds} → {student_ds}: Branch Comparison')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', lw=0.4)
    plt.tight_layout()
    save_figure(fig, os.path.join(ds_out, f'KD_{kd_label}_branch_comparison.png'), logger)

    return {
        'branch1':       {'model': student_b1, 'metrics': metrics_b1,
                          'feature_space': 'teacher'},
        'branch2':       {'model': student_b2, 'metrics': metrics_b2,
                          'feature_space': 'union', 'union_scaler': union_scaler},
        'alpha':         best_alpha,
        'teacher_model': teacher_model,
        'teacher_auc':   teacher_auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_complete_pipeline(output_dir='output', log_file='pipeline_log.txt',
                          target_col='PTB_NEW', datasets=None):
    if datasets is None:
        datasets = ['T1', 'T2', 'T3', 'T1_T2', 'T2_T3', 'T1_T3']

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(log_file)
    logger.info("=" * 80)
    logger.info(f"Metabolomics ML Pipeline | target={target_col} | datasets={datasets}")
    logger.info("=" * 80)

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: CONVENTIONAL METHODS")
    logger.info("=" * 80)

    stage1_results = {}
    for ds in datasets:
        tr = f'{ds}_train.csv'
        te = f'{ds}_test.csv'
        if not os.path.exists(tr) or not os.path.exists(te):
            logger.warning(f"Skipping {ds}: files not found")
            continue
        try:
            results_df, trained_models, feature_selectors, fitted_scalers = \
                analyze_ml_pipeline_stage1(tr, te, target_col, output_dir, logger)
            stage1_results[ds] = {
                'results_df':        results_df,
                'trained_models':    trained_models,
                'feature_selectors': feature_selectors,
                'fitted_scalers':    fitted_scalers,
            }
        except Exception as e:
            logger.error(f"Stage 1 error ({ds}): {e}")

    if not stage1_results:
        logger.error("No Stage 1 results — aborting.")
        return {}

    # ── Extract best models per dataset ───────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: KNOWLEDGE DISTILLATION")
    logger.info("=" * 80)

    best_per_dataset = extract_best_stage1_models(stage1_results, logger)

    if len(best_per_dataset) < 2:
        logger.error("Need at least 2 datasets with valid models for KD — aborting Stage 2.")
        return {'stage1': stage1_results}

    # ── Select teacher: dataset with highest ROC-AUC ──────────────────────────
    best_teacher_ds = max(
        best_per_dataset,
        key=lambda ds: best_per_dataset[ds]['metrics_row']['ROC_AUC']
    )
    teacher_info = best_per_dataset[best_teacher_ds]
    logger.info(f"\nTeacher: {best_teacher_ds}  "
                f"ROC-AUC={teacher_info['metrics_row']['ROC_AUC']:.4f}  "
                f"FS={teacher_info['fs_name']}")

    # ── Run KD for each remaining dataset as student ──────────────────────────
    stage2_kd = {}
    for student_ds, student_info in best_per_dataset.items():
        if student_ds == best_teacher_ds:
            continue
        try:
            kd_res = knowledge_distillation_stage2(
                best_teacher_ds, student_ds,
                teacher_info, student_info,
                target_col, output_dir, logger
            )
            stage2_kd[f'{best_teacher_ds}_to_{student_ds}'] = kd_res
        except Exception as e:
            logger.error(f"KD error {best_teacher_ds}->{student_ds}: {e}")

    # ── Save Stage 2 summary CSV ──────────────────────────────────────────────
    kd_rows = []
    for name, kd in stage2_kd.items():
        for branch_key, branch_label in [('branch1', 'Teacher_FS'),
                                          ('branch2', 'Union_FS')]:
            m = kd[branch_key]['metrics']
            kd_rows.append({
                'KD_Pair':              name,
                'Branch':               branch_label,
                'Alpha':                kd['alpha'],
                'Teacher_ROC_AUC':      kd['teacher_auc'],
                'Student_ROC_AUC':      m['roc_auc'],
                'Student_ROC_AUC_lo':   m['roc_auc_lo'],
                'Student_ROC_AUC_hi':   m['roc_auc_hi'],
                'Student_PR_AUC':       m['pr_auc'],
                'Student_AP':           m['ap_score'],
                'Student_AP_lo':        m['ap_score_lo'],
                'Student_AP_hi':        m['ap_score_hi'],
                'Student_Recall':       m['recall'],
                'Student_Recall_lo':    m['recall_lo'],
                'Student_Recall_hi':    m['recall_hi'],
                'Student_Precision':    m['precision'],
                'Student_Precision_lo': m['precision_lo'],
                'Student_Precision_hi': m['precision_hi'],
                'Student_F1':           m['f1'],
                'Student_F1_lo':        m['f1_lo'],
                'Student_F1_hi':        m['f1_hi'],
                'Threshold':            m['threshold'],
                'Best_Params':          str(m['best_params']),
            })

    if kd_rows:
        kd_df = pd.DataFrame(kd_rows)
        kd_df.to_csv(os.path.join(output_dir, 'stage2_kd_results.csv'), index=False)
        logger.info("Stage 2 KD results saved.")

        fig, ax = plt.subplots(figsize=(10, max(4, len(kd_df) * 0.55)))
        branch_colors = {'Teacher_FS': PALETTE[1], 'Union_FS': PALETTE[2]}
        offsets       = {'Teacher_FS': -0.15, 'Union_FS': 0.15}
        pairs         = kd_df['KD_Pair'].unique()
        y_ticks       = {pair: i for i, pair in enumerate(pairs)}
        legend_added  = set()

        for _, row in kd_df.iterrows():
            y     = y_ticks[row['KD_Pair']] + offsets[row['Branch']]
            lbl   = row['Branch'] if row['Branch'] not in legend_added else ''
            legend_added.add(row['Branch'])
            ax.errorbar(
                row['Student_ROC_AUC'], y,
                xerr=[[row['Student_ROC_AUC'] - row['Student_ROC_AUC_lo']],
                      [row['Student_ROC_AUC_hi'] - row['Student_ROC_AUC']]],
                fmt='o', color=branch_colors[row['Branch']],
                markersize=8, capsize=4, lw=1.5, label=lbl
            )

        if len(kd_df) > 0:
            teacher_auc_ref = kd_df['Teacher_ROC_AUC'].iloc[0]
            ax.axvline(teacher_auc_ref, color='grey', lw=1, ls=':',
                       label=f'Teacher AUC={teacher_auc_ref:.3f}')

        ax.set_yticks(list(y_ticks.values()))
        ax.set_yticklabels(list(y_ticks.keys()), fontsize=9)
        ax.set_xlabel('Student ROC-AUC (95% CI)')
        ax.set_title('Knowledge Distillation: Student Performance by Branch and Dataset Pair')
        ax.legend(title='Branch', loc='lower right')
        ax.grid(axis='x', lw=0.4)
        plt.tight_layout()
        save_figure(fig, os.path.join(output_dir, 'stage2_kd_overview.png'), logger)

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Complete")
    logger.info("=" * 80)

    return {
        'stage1':       stage1_results,
        'stage2_kd':    stage2_kd,
        'best_teacher': {
            'dataset': best_teacher_ds,
            'info':    teacher_info,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_col', type=str, default='PTB_NEW')
    parser.add_argument('--datasets',   type=str, nargs='+',
                        default=['T1', 'T2', 'T3', 'T1_T2', 'T2_T3', 'T1_T3'])
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--log_file',   type=str, default='pipeline_log.txt')
    args = parser.parse_args()

    run_complete_pipeline(
        output_dir=args.output_dir,
        log_file=args.log_file,
        target_col=args.target_col,
        datasets=args.datasets
    )
