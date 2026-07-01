# -*- coding: utf-8 -*-
"""
Machine-learning pipeline for binary diagnostic classification.

The script performs data preprocessing, feature selection, model training,
hyperparameter tuning, internal validation, model comparison, and SHAP-based
model interpretation.

Expected input:
    Dataset.xlsx by default, or another Excel file supplied with --input.
    If the input file is unavailable, a simulated de-identified dataset is generated.

Example:
    python ml_pipeline.py --input Dataset.xlsx
"""

import os
import argparse
import warnings
import math
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, brier_score_loss, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import shap

warnings.filterwarnings("ignore")

# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
RANDOM_SPLIT_SEED = 94       # 70:30 training/internal validation split
RANDOM_ANALYSIS_SEED = 42    # LASSO, CV, model tuning, bootstrap
N_BOOT = 1000                # bootstrap repetitions for 95% CIs
INNER_CV_SPLITS = 5          # tuning CV inside training cohort
OUTER_CV_SPLITS = 5          # out-of-fold performance estimation inside training cohort
LASSO_CV_SPLITS = 10
FINAL_MODEL_FEATURES = [
    'feature_01', 'feature_02', 'feature_03',
    'feature_04', 'feature_05', 'feature_06',
    'feature_07', 'feature_08', 'feature_09'
]
ALL_PREDICTOR_FEATURES = FINAL_MODEL_FEATURES + [
    'feature_10', 'feature_11', 'feature_12', 'feature_13'
]



def format_ci(point, low, high, digits=3):
    return f"{point:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def safe_div(num, den):
    return np.nan if den == 0 else num / den


def calculate_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': safe_div(tn, tn + fp),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'Brier Score': brier_score_loss(y_true, y_prob),
        'PPV': precision_score(y_true, y_pred, zero_division=0),
        'NPV': safe_div(tn, tn + fn),
        'FPR': safe_div(fp, fp + tn),
        'FNR': safe_div(fn, fn + tp),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
    }


def bootstrap_metric_ci(y_true, y_prob, threshold, n_boot=N_BOOT, seed=RANDOM_ANALYSIS_SEED):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(seed)
    metric_names = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 Score',
                    'Brier Score', 'PPV', 'NPV', 'FPR', 'FNR']
    boot_rows = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        row = calculate_binary_metrics(y_true[idx], y_prob[idx], threshold)
        boot_rows.append([row[m] for m in metric_names])
    boot_df = pd.DataFrame(boot_rows, columns=metric_names)
    return boot_df.quantile(0.025), boot_df.quantile(0.975)


def youden_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[np.argmax(tpr - fpr)])


# ==============================================================================
# DeLong test implementation for correlated ROC AUCs
# ==============================================================================
def compute_midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    t2 = np.empty(n, dtype=float)
    t2[j] = t
    return t2


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))[0, 0]
    z = float(np.ravel(z)[0])
    normal_cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return float(2 * (1 - normal_cdf))


def delong_roc_test(y_true, pred_one, pred_two):
    y_true = np.asarray(y_true).astype(int)
    pred_one = np.asarray(pred_one)
    pred_two = np.asarray(pred_two)
    order = np.argsort(-y_true)  # positives first
    label_1_count = int(np.sum(y_true))
    predictions_sorted = np.vstack((pred_one, pred_two))[:, order]
    aucs, cov = fast_delong(predictions_sorted, label_1_count)
    return calc_pvalue(aucs, cov), float(aucs[0] - aucs[1])


def paired_bootstrap_auc_diff(y_true, pred_one, pred_two, n_boot=N_BOOT, seed=RANDOM_ANALYSIS_SEED):
    y_true = np.asarray(y_true).astype(int)
    pred_one = np.asarray(pred_one)
    pred_two = np.asarray(pred_two)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], pred_one[idx]) - roc_auc_score(y_true[idx], pred_two[idx]))
    diffs = np.asarray(diffs)
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def generate_simulated_dataset(n_samples=220, seed=RANDOM_ANALYSIS_SEED):
    """Generate a de-identified binary-classification dataset for code testing."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n_samples, len(ALL_PREDICTOR_FEATURES)))
    coefs = np.array([0.9, -0.8, 0.7, 0.6, -0.5, 0.5, -0.4, 0.35, 0.3])
    linear_score = x[:, :len(FINAL_MODEL_FEATURES)].dot(coefs) + rng.normal(scale=0.8, size=n_samples)
    prob = 1 / (1 + np.exp(-linear_score))
    y = rng.binomial(1, prob)

    simulated_df = pd.DataFrame(x, columns=ALL_PREDICTOR_FEATURES)
    missing_mask = rng.rand(*simulated_df.shape) < 0.02
    simulated_df = simulated_df.mask(missing_mask)
    simulated_df.insert(0, 'record_id', [f'R{i + 1:04d}' for i in range(n_samples)])
    simulated_df['Group'] = y
    return simulated_df


# ==============================================================================
# STEP 1: DATA LOADING
# ==============================================================================
parser = argparse.ArgumentParser(description='Run the machine-learning analysis pipeline.')
parser.add_argument('--input', default='Dataset.xlsx', help='Path to the Excel dataset. Default: Dataset.xlsx')
parser.add_argument('--summary-output', default='analysis_summary.xlsx', help='Path for the summary workbook.')
args = parser.parse_args()

input_file = args.input
summary_output = args.summary_output
if os.path.exists(input_file):
    df = pd.read_excel(input_file)
    df.columns = [col.replace('#', '') for col in df.columns]
    data_source = f'Loaded input file: {input_file}'
else:
    df = generate_simulated_dataset()
    data_source = 'Input file not found; simulated de-identified dataset was generated.'

target = 'Group'
exclude_vars_for_model = [
    'sample_id', 'record_id', 'label', target
]
features_pool = [col for col in df.columns if col not in exclude_vars_for_model]

# ==============================================================================
# STEP 2: TRAINING/HELD-OUT INTERNAL VALIDATION SPLIT
# ============================================================================== 
df_train, df_val = train_test_split(
    df, test_size=0.3, random_state=RANDOM_SPLIT_SEED, stratify=df[target]
)
y_train = df_train[target].reset_index(drop=True).astype(int)
y_val = df_val[target].reset_index(drop=True).astype(int)

# ==============================================================================
# STEP 3: TRAINING-ONLY LASSO AND FEATURE-SET ASSESSMENT
# ==============================================================================
print("Running training-only LASSO feature selection...")

# LASSO uses training-cohort medians and training-fitted standardization only.
lasso_imputer = SimpleImputer(strategy='median')
lasso_scaler = StandardScaler()
X_train_pool_imputed = pd.DataFrame(
    lasso_imputer.fit_transform(df_train[features_pool]),
    columns=features_pool,
    index=df_train.index
)
X_train_pool_scaled = pd.DataFrame(
    lasso_scaler.fit_transform(X_train_pool_imputed),
    columns=features_pool,
    index=df_train.index
)

lasso_cv = LassoCV(cv=LASSO_CV_SPLITS, random_state=RANDOM_ANALYSIS_SEED, max_iter=100000, n_alphas=100)
lasso_cv.fit(X_train_pool_scaled, y_train)
alphas = lasso_cv.alphas_
mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
# The one-standard-error rule is defined using the standard error of the
# mean cross-validation error, not the fold-to-fold standard deviation.
mse_sd = np.std(lasso_cv.mse_path_, axis=1, ddof=1)
mse_se = mse_sd / np.sqrt(LASSO_CV_SPLITS)
idx_min = int(np.argmin(mse_mean))
alpha_min = float(alphas[idx_min])
one_se_threshold = mse_mean[idx_min] + mse_se[idx_min]
eligible_1se_indices = np.where(mse_mean <= one_se_threshold)[0]

# sklearn orders alpha values from largest to smallest. The first eligible
# alpha is therefore the strongest penalty and yields the most parsimonious
# model within one standard error of the minimum cross-validation error.
idx_1se = int(eligible_1se_indices[0])
alpha_1se = float(alphas[idx_1se])

lasso_min_model = Lasso(alpha=alpha_min, random_state=RANDOM_ANALYSIS_SEED, max_iter=100000).fit(X_train_pool_scaled, y_train)
selected_min_features = [features_pool[i] for i, coef in enumerate(lasso_min_model.coef_) if coef != 0]


# LASSO, correlation analysis, and VIF are used for feature-set assessment.
# The final modeling feature set reflects the features retained after this assessment.
lasso_candidate_features = selected_min_features
missing_model_features = [f for f in FINAL_MODEL_FEATURES if f not in df_train.columns]
if missing_model_features:
    raise ValueError(f"Missing final model features in the input data: {missing_model_features}")
final_model_features = FINAL_MODEL_FEATURES

# Correlation summaries are calculated in memory only; no figures are exported.
if len(lasso_candidate_features) > 1:
    df_corr_candidates = X_train_pool_scaled[lasso_candidate_features].corr(method='spearman')
else:
    df_corr_candidates = pd.DataFrame()

X_train_final_for_corr = X_train_pool_scaled[final_model_features]
df_corr_final = X_train_final_for_corr.corr(method='spearman')

X_vif = add_constant(X_train_final_for_corr)
df_vif_report = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
df_vif_report = df_vif_report[df_vif_report['Feature'] != 'const']

# ==============================================================================
# STEP 4: MODEL SPECIFICATIONS AND GRID SEARCH
# ==============================================================================
print("Tuning models with prespecified grid search.")

X_train_final_raw = df_train[final_model_features].reset_index(drop=True)
X_val_final_raw = df_val[final_model_features].reset_index(drop=True)

# Every model is wrapped in a pipeline so imputation and scaling are fitted only within
# the corresponding training data during final fitting, inner CV, and outer CV.
def make_pipeline(classifier):
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])

model_specs = {
    'Logistic Regression': {
        'estimator': make_pipeline(LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__C': [0.01, 0.1, 1, 10]}
    },
    'Random Forest': {
        'estimator': make_pipeline(RandomForestClassifier(
            criterion='gini',
            bootstrap=True,
            max_features='sqrt',
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 5, 10]}
    },
    'SVM': {
        'estimator': make_pipeline(SVC(
            kernel='rbf',
            probability=True,
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__C': [0.1, 1, 10, 100], 'clf__gamma': ['scale', 'auto', 0.01, 0.1]}
    },
    'Single-hidden-layer MLP': {
        'estimator': make_pipeline(MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__hidden_layer_sizes': [(100,)], 'clf__alpha': [0.0001, 0.001]}
    },
    'Two-hidden-layer MLP': {
        'estimator': make_pipeline(MLPClassifier(
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__hidden_layer_sizes': [(100, 50)], 'clf__alpha': [0.0001, 0.001]}
    },
    'XGBoost': {
        'estimator': make_pipeline(XGBClassifier(
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric='logloss',
            random_state=RANDOM_ANALYSIS_SEED
        )),
        'grid': {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]}
    },
    'LightGBM': {
        'estimator': make_pipeline(LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=-1,
            random_state=RANDOM_ANALYSIS_SEED,
            verbose=-1
        )),
        'grid': {'clf__n_estimators': [50, 100], 'clf__learning_rate': [0.01, 0.1]}
    }
}

inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_ANALYSIS_SEED)
outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_ANALYSIS_SEED)

best_models = {}
hyperparams_rows = []

for name, spec in model_specs.items():
    grid_search = GridSearchCV(
        estimator=clone(spec['estimator']),
        param_grid=spec['grid'],
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train_final_raw, y_train)
    best_models[name] = grid_search.best_estimator_
    hyperparams_rows.append({
        'Classifier_Algorithm': name,
        'Hyperparameter_Search_Strategy': 'Prespecified exhaustive grid search',
        'Optimization_Metric': 'ROC AUC',
        'Inner_CV_Folds': INNER_CV_SPLITS,
        'Random_State': RANDOM_ANALYSIS_SEED,
        'Evaluated_Hyperparameter_Grid_Space': str(spec['grid']),
        'Optimized_Final_Parameters': str(grid_search.best_params_),
        'Best_Inner_CV_AUC': round(float(grid_search.best_score_), 4)
    })

df_hyperparams_table = pd.DataFrame(hyperparams_rows)

# ==============================================================================
# STEP 5: NESTED OUT-OF-FOLD TRAINING PERFORMANCE
# ==============================================================================
print("Estimating training performance with nested out-of-fold prediction...")
train_report_rows = []
train_confusion_rows = []
saved_train_oof_probs = {}


for name, spec in model_specs.items():
    nested_estimator = GridSearchCV(
        estimator=clone(spec['estimator']),
        param_grid=spec['grid'],
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True
    )
    oof_probs = cross_val_predict(
        nested_estimator, X_train_final_raw, y_train, cv=outer_cv,
        method='predict_proba', n_jobs=-1
    )[:, 1]
    saved_train_oof_probs[name] = oof_probs
    threshold = youden_threshold(y_train, oof_probs)
    metrics = calculate_binary_metrics(y_train, oof_probs, threshold)
    low, high = bootstrap_metric_ci(y_train, oof_probs, threshold)

    train_report_rows.append({
        'Algorithm': name,
        'Evaluation_Design': f'Nested {OUTER_CV_SPLITS}-fold OOF; inner {INNER_CV_SPLITS}-fold grid search',
        'Threshold_Source': 'Youden index from out-of-fold training probabilities',
        'Best Threshold': round(threshold, 3),
        **{m: format_ci(metrics[m], low[m], high[m]) for m in ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 Score', 'Brier Score', 'PPV', 'NPV', 'FPR', 'FNR']}
    })
    train_confusion_rows.append({
        'Algorithm': name, 'Threshold': round(threshold, 3),
        'TN': metrics['TN'], 'FP': metrics['FP'], 'FN': metrics['FN'], 'TP': metrics['TP'],
        'FPR': metrics['FPR'], 'FNR': metrics['FNR'], 'PPV': metrics['PPV'], 'NPV': metrics['NPV']
    })

df_train_eval = pd.DataFrame(train_report_rows)
df_train_confusion = pd.DataFrame(train_confusion_rows)

# ==============================================================================
# STEP 6: HELD-OUT INTERNAL VALIDATION PERFORMANCE
# ==============================================================================
print("Evaluating final optimized models in the held-out internal validation cohort...")

val_report_rows = []
val_confusion_rows = []
saved_val_probs = {}


for name, model in best_models.items():

    # --------------------------------------------------------------------------
    # Single source of truth for validation prediction
    # --------------------------------------------------------------------------
    v_probs = model.predict_proba(X_val_final_raw)[:, 1]
    saved_val_probs[name] = v_probs

    # Threshold is determined from the training cohort only to avoid validation leakage
    train_probs_for_threshold = model.predict_proba(X_train_final_raw)[:, 1]
    threshold = youden_threshold(y_train, train_probs_for_threshold)

    # All validation metrics are calculated using the same validation probability vector
    metrics = calculate_binary_metrics(y_val, v_probs, threshold)
    low, high = bootstrap_metric_ci(y_val, v_probs, threshold)

    val_report_rows.append({
        'Algorithm': name,
        'Evaluation_Design': 'Held-out internal validation cohort',
        'Threshold_Source': 'Youden index from training cohort probabilities only',
        'Best Threshold': round(threshold, 3),
        **{
            m: format_ci(metrics[m], low[m], high[m])
            for m in [
                'AUC', 'Accuracy', 'Sensitivity', 'Specificity',
                'F1 Score', 'Brier Score', 'PPV', 'NPV', 'FPR', 'FNR'
            ]
        }
    })

    val_confusion_rows.append({
        'Algorithm': name,
        'Threshold': round(threshold, 3),
        'TN': metrics['TN'],
        'FP': metrics['FP'],
        'FN': metrics['FN'],
        'TP': metrics['TP'],
        'FPR': metrics['FPR'],
        'FNR': metrics['FNR'],
        'PPV': metrics['PPV'],
        'NPV': metrics['NPV']
    })


df_val_eval = pd.DataFrame(val_report_rows)
df_val_confusion = pd.DataFrame(val_confusion_rows)

# Consistency check: all validation probabilities must match the validation cohort size
for model_name, probs in saved_val_probs.items():
    assert len(probs) == len(y_val), f"Validation probability length mismatch for {model_name}"

print("Validation predictions and evaluation metrics were generated from the same prediction source.")

# ==============================================================================
# STEP 7: DECISION-CURVE SUMMARY
# ==============================================================================
print("Calculating decision-curve summary without exporting figures...")

dca_grid = np.linspace(0.01, 0.99, 99)
y_val_arr = np.asarray(y_val)
n_samples = len(y_val_arr)
prevalence = np.mean(y_val_arr)
all_benefit = prevalence - (1 - prevalence) * (dca_grid / (1 - dca_grid))

dca_summary_rows = []
for name, probs in saved_val_probs.items():
    model_benefit = []
    for t in dca_grid:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val_arr, preds, labels=[0, 1]).ravel()
        nb = (tp / n_samples) - (fp / n_samples) * (t / (1 - t))
        model_benefit.append(nb)
    model_benefit = np.asarray(model_benefit)
    dca_summary_rows.append({
        'Algorithm': name,
        'Threshold_Range_Evaluated': '0.01-0.99',
        'Thresholds_With_Net_Benefit_Above_Treat_All_and_Treat_None':
            ', '.join([f"{t:.2f}" for t, nb, ab in zip(dca_grid, model_benefit, all_benefit) if nb > max(ab, 0)])
    })

df_dca_summary = pd.DataFrame(dca_summary_rows)

# ==============================================================================
# STEP 8: PAIRWISE MODEL COMPARISONS IN TRAINING AND VALIDATION COHORTS
# ==============================================================================
print("Running pairwise AUC comparisons in training and validation cohorts...")

# --------------------------------------------------------------------------
# Safety checks: ensure all probability vectors match the corresponding labels
# --------------------------------------------------------------------------
for model_name, probs in saved_train_oof_probs.items():
    assert len(probs) == len(y_train), f"Training probability length mismatch for {model_name}"

for model_name, probs in saved_val_probs.items():
    assert len(probs) == len(y_val), f"Validation probability length mismatch for {model_name}"


def build_pairwise_auc_comparison(y_true, prob_dict, cohort_name):
    """
    Pairwise AUC comparison using the same probability vectors that were used
    for ROC plotting and performance evaluation.
    """
    algos_local = list(prob_dict.keys())
    rows = []

    for i in range(len(algos_local)):
        for j in range(i + 1, len(algos_local)):

            model_1 = algos_local[i]
            model_2 = algos_local[j]

            prob_1 = np.asarray(prob_dict[model_1])
            prob_2 = np.asarray(prob_dict[model_2])
            y_true_arr = np.asarray(y_true).astype(int)

            auc_1 = roc_auc_score(y_true_arr, prob_1)
            auc_2 = roc_auc_score(y_true_arr, prob_2)

            try:
                p_delong, auc_diff = delong_roc_test(y_true_arr, prob_1, prob_2)
            except Exception:
                p_delong = np.nan
                auc_diff = auc_1 - auc_2

            boot_mean, boot_low, boot_high = paired_bootstrap_auc_diff(
                y_true_arr, prob_1, prob_2
            )

            rows.append({
                'Cohort': cohort_name,
                'Model_1': model_1,
                'Model_2': model_2,
                'AUC_Model_1': round(auc_1, 3),
                'AUC_Model_2': round(auc_2, 3),
                'AUC_Difference_Model_1_minus_Model_2': round(auc_diff, 3),
                'DeLong_P_value': p_delong,
                'Bootstrap_Mean_AUC_Difference': round(boot_mean, 3),
                'Bootstrap_95CI_Lower': round(boot_low, 3),
                'Bootstrap_95CI_Upper': round(boot_high, 3)
            })

    return pd.DataFrame(rows)


# Training cohort: use nested out-of-fold probabilities
df_train_pairwise_auc_comparison = build_pairwise_auc_comparison(
    y_train,
    saved_train_oof_probs,
    'Training cohort: nested out-of-fold predictions'
)

# Validation cohort: use the SAME saved validation probabilities as ROC Figure 5
df_val_pairwise_auc_comparison = build_pairwise_auc_comparison(
    y_val,
    saved_val_probs,
    'Held-out internal validation cohort'
)

df_pairwise_auc_comparison = pd.concat(
    [df_train_pairwise_auc_comparison, df_val_pairwise_auc_comparison],
    ignore_index=True
)


# ==============================================================================
# Compact SVM vs Logistic Regression comparison
# ==============================================================================
def build_svm_lr_comparison(y_true, prob_dict, cohort_name):
    if 'SVM' not in prob_dict or 'Logistic Regression' not in prob_dict:
        return pd.DataFrame()

    y_true_arr = np.asarray(y_true).astype(int)
    svm_prob = np.asarray(prob_dict['SVM'])
    lr_prob = np.asarray(prob_dict['Logistic Regression'])

    svm_auc = roc_auc_score(y_true_arr, svm_prob)
    lr_auc = roc_auc_score(y_true_arr, lr_prob)

    try:
        p_delong, auc_diff = delong_roc_test(y_true_arr, svm_prob, lr_prob)
    except Exception:
        p_delong = np.nan
        auc_diff = svm_auc - lr_auc

    boot_mean, boot_low, boot_high = paired_bootstrap_auc_diff(
        y_true_arr,
        svm_prob,
        lr_prob
    )

    return pd.DataFrame([{
        'Cohort': cohort_name,
        'Comparison': 'SVM vs Logistic Regression',
        'SVM_AUC': round(svm_auc, 4),
        'LR_AUC': round(lr_auc, 4),
        'AUC_Difference_SVM_minus_LR': round(auc_diff, 4),
        'DeLong_P_value': p_delong,
        'Paired_Bootstrap_AUC_Difference_95CI': f"{boot_mean:.3f} ({boot_low:.3f}-{boot_high:.3f})",
        'Interpretation_Guide': 'If the CI includes 0 or DeLong P >= 0.05, state that SVM and LR showed comparable discrimination and avoid claiming statistical superiority.'
    }])


df_svm_lr_comparison = pd.concat([
    build_svm_lr_comparison(
        y_train,
        saved_train_oof_probs,
        'Training cohort: nested out-of-fold predictions'
    ),
    build_svm_lr_comparison(
        y_val,
        saved_val_probs,
        'Held-out internal validation cohort'
    )
], ignore_index=True)

print("Pairwise DeLong comparisons completed using the same probability sources as ROC analysis.")

# ==============================================================================
# STEP 9: SHAP INTERPRETATION FOR PRIMARY SVM MODEL
# ============================================================================== 
print("Calculating SHAP values for SVM. Interpret as predictive contribution, not causality.")
primary_model_name = 'SVM'
primary_model = best_models[primary_model_name]

# SHAP on pipeline input: the explainer calls the complete preprocessing + classifier pipeline.
X_train_shap = X_train_final_raw.copy()
X_val_shap = X_val_final_raw.copy()
background = shap.kmeans(pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_train_shap), columns=final_model_features), 10)

# Define prediction function that accepts the imputed/scaled-compatible array from KernelExplainer.
def svm_predict_proba_from_array(x):
    x_df = pd.DataFrame(x, columns=final_model_features)
    return primary_model.predict_proba(x_df)

X_val_shap_imputed = pd.DataFrame(
    primary_model.named_steps['imputer'].transform(X_val_shap),
    columns=final_model_features
)
explainer = shap.KernelExplainer(svm_predict_proba_from_array, background)
shap_values = explainer.shap_values(X_val_shap_imputed, nsamples=150)

if isinstance(shap_values, list):
    shap_values_target = shap_values[1]
elif isinstance(shap_values, np.ndarray):
    shap_values_target = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
else:
    shap_values_target = shap_values.values[:, :, 1] if (hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3) else shap_values

# Summarize SHAP values in memory only; no SHAP figures are exported.
shap_importance = np.mean(np.abs(shap_values_target), axis=0)
df_shap_importance = pd.DataFrame({
    'Feature': final_model_features,
    'Mean_Absolute_SHAP_Value': shap_importance
}).sort_values('Mean_Absolute_SHAP_Value', ascending=False).reset_index(drop=True)

# ==============================================================================
# STEP 10: SUMMARY EXPORT
# ==============================================================================
summary_info = pd.DataFrame([{
    'Data_Source': data_source,
    'Training_Size': len(df_train),
    'Validation_Size': len(df_val),
    'Outcome_Column': target,
    'Summary_Workbook': summary_output
}])

with pd.ExcelWriter(summary_output, engine='openpyxl') as writer:
    summary_info.to_excel(writer, sheet_name='Run_Info', index=False)
    df_hyperparams_table.to_excel(writer, sheet_name='Hyperparameters', index=False)
    df_train_eval.to_excel(writer, sheet_name='Training_Performance', index=False)
    df_val_eval.to_excel(writer, sheet_name='Validation_Performance', index=False)
    df_svm_lr_comparison.to_excel(writer, sheet_name='SVM_vs_LR', index=False)
    df_shap_importance.to_excel(writer, sheet_name='SHAP_Summary', index=False)

# ==============================================================================
# STEP 11: TERMINAL SUMMARY
# ==============================================================================
print("=" * 80)
print("Machine-learning pipeline completed successfully.")
print(data_source)
print(f"Summary workbook exported to: {os.path.abspath(summary_output)}")
print("=" * 80)
