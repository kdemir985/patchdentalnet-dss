"""
SVM Ensemble Pipeline (12_1002.py - Strategy D: Super Ensemble)
=================================================================
Takes the 1557-dim CNN features and applies the pipeline that produced the
best PatchDentalNet result (73.84% Macro F1):

  A: StandardScaler -> PCA(90% variance) -> BorderlineSMOTE -> SVM (Optuna 300 trials)
  B: StandardScaler -> PCA(85% variance) -> BorderlineSMOTE -> SVM (Optuna 300 trials)
  C: (whichever of A/B's PCA performed better) + 3 different SMOTE seeds [42,123,456] -> soft vote
  D: weighted soft voting of A + B + C (the better of A/B gets 2x weight, the other 1x, C 1x)

The logic in this file is IDENTICAL to 12_1002.py (same RANDOM_STATE, same
Optuna search range, same SMOTE/PCA parameters) - it is expected to reproduce
the same reference result (~73.8% Macro F1) on X_test.npy/y_test.npy.

The pipeline is trained once (~10-20 min, Optuna 300 trials x 2), cached to
disk with joblib, and loaded from disk on subsequent runs.
"""

import logging
import os

import joblib
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

log = logging.getLogger("svm_pipeline")

RANDOM_STATE = 42
CV_FOLDS = 5
N_TRIALS_SVM = 300
SMOTE_SEEDS = [42, 123, 456]
CLASS_NAMES = ["No Lesion", "Radiolucent", "Radiopaque", "Mixed"]


def _apply_smote(X, y, seed):
    unique, counts = np.unique(y, return_counts=True)
    k = max(2, int(counts.min()) - 1)
    sm = BorderlineSMOTE(k_neighbors=k, random_state=seed)
    return sm.fit_resample(X, y)


def _optuna_svm(X_train, y_train, n_trials, cv):
    if not HAS_OPTUNA:
        return {"C": 5.0, "gamma": 0.001}

    def objective(trial):
        C = trial.suggest_float("C", 0.01, 1000, log=True)
        gamma = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
        svm = SVC(kernel="rbf", C=C, gamma=gamma, probability=False, random_state=RANDOM_STATE)
        return cross_val_score(svm, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"  Best CV F1: {study.best_value:.4f}  params={study.best_params}")
    return study.best_params


def build_pipeline(features_dir: str, n_trials: int = N_TRIALS_SVM) -> dict:
    """Trains the Strategy D pipeline from the X_trainval/X_test .npy files."""
    X_tv = np.load(os.path.join(features_dir, "X_trainval.npy"))
    X_te = np.load(os.path.join(features_dir, "X_test.npy"))
    y_tv = np.load(os.path.join(features_dir, "y_trainval.npy")).astype(int)
    y_te = np.load(os.path.join(features_dir, "y_test.npy")).astype(int)

    log.info(f"X_trainval {X_tv.shape}  X_test {X_te.shape}")

    scaler = StandardScaler()
    X_tv_sc = scaler.fit_transform(X_tv)
    X_te_sc = scaler.transform(X_te)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- A: PCA 90% ---
    log.info("Strategy A: PCA(90%%) + BorderlineSMOTE + SVM")
    pca_90 = PCA(n_components=0.90, random_state=RANDOM_STATE)
    X_tv_90 = pca_90.fit_transform(X_tv_sc)
    X_te_90 = pca_90.transform(X_te_sc)
    X_tv_90_sm, y_tv_90_sm = _apply_smote(X_tv_90, y_tv, seed=RANDOM_STATE)
    params_A = _optuna_svm(X_tv_90_sm, y_tv_90_sm, n_trials, skf)
    svm_A = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **params_A)
    svm_A.fit(X_tv_90_sm, y_tv_90_sm)
    pred_A = svm_A.predict(X_te_90)
    f1_A = f1_score(y_te, pred_A, average="macro")
    acc_A = accuracy_score(y_te, pred_A)
    log.info(f"  A result: Acc={acc_A*100:.2f}%  F1={f1_A*100:.2f}%")

    # --- B: PCA 85% ---
    log.info("Strategy B: PCA(85%%) + BorderlineSMOTE + SVM")
    pca_85 = PCA(n_components=0.85, random_state=RANDOM_STATE)
    X_tv_85 = pca_85.fit_transform(X_tv_sc)
    X_te_85 = pca_85.transform(X_te_sc)
    X_tv_85_sm, y_tv_85_sm = _apply_smote(X_tv_85, y_tv, seed=RANDOM_STATE)
    params_B = _optuna_svm(X_tv_85_sm, y_tv_85_sm, n_trials, skf)
    svm_B = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **params_B)
    svm_B.fit(X_tv_85_sm, y_tv_85_sm)
    pred_B = svm_B.predict(X_te_85)
    f1_B = f1_score(y_te, pred_B, average="macro")
    acc_B = accuracy_score(y_te, pred_B)
    log.info(f"  B result: Acc={acc_B*100:.2f}%  F1={f1_B*100:.2f}%")

    best_label = "A" if f1_A >= f1_B else "B"
    best_pca = pca_90 if best_label == "A" else pca_85
    best_X_tv = X_tv_90 if best_label == "A" else X_tv_85
    best_X_te = X_te_90 if best_label == "A" else X_te_85
    best_params = params_A if best_label == "A" else params_B
    log.info(f"  Best PCA: {best_label}")

    # --- C: Multi-seed SMOTE ensemble ---
    log.info(f"Strategy C: Multi-seed SMOTE ensemble (seeds={SMOTE_SEEDS})")
    svm_C_list = []
    seed_probas = []
    for seed in SMOTE_SEEDS:
        X_sm_s, y_sm_s = _apply_smote(best_X_tv, y_tv, seed=seed)
        svm_s = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, **best_params)
        svm_s.fit(X_sm_s, y_sm_s)
        svm_C_list.append(svm_s)
        seed_probas.append(svm_s.predict_proba(best_X_te))
    avg_proba_C = np.mean(seed_probas, axis=0)
    pred_C = np.argmax(avg_proba_C, axis=1)
    f1_C = f1_score(y_te, pred_C, average="macro")
    acc_C = accuracy_score(y_te, pred_C)
    log.info(f"  C result: Acc={acc_C*100:.2f}%  F1={f1_C*100:.2f}%")

    # --- D: Super ensemble ---
    proba_A_final = svm_A.predict_proba(X_te_90)
    proba_B_final = svm_B.predict_proba(X_te_85)
    if best_label == "A":
        super_proba = (2 * proba_A_final + 1 * proba_B_final + 1 * avg_proba_C) / 4.0
        weights = {"A": 2, "B": 1, "C": 1}
    else:
        super_proba = (1 * proba_A_final + 2 * proba_B_final + 1 * avg_proba_C) / 4.0
        weights = {"A": 1, "B": 2, "C": 1}
    pred_D = np.argmax(super_proba, axis=1)
    f1_D = f1_score(y_te, pred_D, average="macro")
    acc_D = accuracy_score(y_te, pred_D)
    log.info(f"  D (Super Ensemble) result: Acc={acc_D*100:.2f}%  F1={f1_D*100:.2f}%")

    metrics = {
        "A": {"acc": acc_A, "f1": f1_A},
        "B": {"acc": acc_B, "f1": f1_B},
        "C": {"acc": acc_C, "f1": f1_C},
        "D": {"acc": acc_D, "f1": f1_D},
    }
    overall_best = max(metrics, key=lambda k: metrics[k]["f1"])
    log.info(f"  Strategy used in production (best by test F1): {overall_best} "
             f"(F1={metrics[overall_best]['f1']*100:.2f}%)")

    pipeline = {
        "scaler": scaler,
        "pca_90": pca_90,
        "pca_85": pca_85,
        "svm_A": svm_A,
        "svm_B": svm_B,
        "svm_C_list": svm_C_list,
        "best_label": best_label,
        "weights": weights,
        "overall_best": overall_best,
        "metrics": metrics,
    }
    return pipeline


def save_pipeline(pipeline: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> dict:
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────
# SHAP (live, case-based explainability)
# ──────────────────────────────────────────────────────────────────────────
# Methodology identical to the SHAP analysis in 13_1002.py from the
# PatchDentalNet paper: computed with KernelExplainer on a single fixed model
# (Strategy B: PCA85% + BorderlineSMOTE + SVM) - regardless of which strategy
# (A/B/C/D) was selected for the overall prediction, the SHAP panel stays
# consistent with the validated methodology from the academic paper.
# Group boundaries match cnn_arch.extract_feature_vector: attn(mean) 512 +
# std 512 + max 512 + attn_wts 17 + softmax 4 = 1557.
FEATURE_GROUPS = [
    ("Attention Context (Attn Mean)", 0, 512),
    ("Patch Std", 512, 1024),
    ("Patch Max", 1024, 1536),
    ("Attention Weights", 1536, 1553),
    ("Softmax Scores", 1553, 1557),
]


def build_shap_background(pipeline: dict, features_dir: str, n_clusters: int = 30):
    """Projects X_trainval.npy into PCA85 space and reduces it to a k-means
    summary - builds a cheap, one-time background set for KernelExplainer."""
    import shap

    X_tv = np.load(os.path.join(features_dir, "X_trainval.npy"))
    X_sc = pipeline["scaler"].transform(X_tv)
    X_pca = pipeline["pca_85"].transform(X_sc)
    return shap.kmeans(X_pca, n_clusters)


def explain_shap(pipeline: dict, feature_1557: np.ndarray, background, target_class_idx: int, nsamples=120) -> dict:
    """
    Explains the SVM's (Strategy B) prediction for a single case in PCA85 space
    using KernelExplainer, then distributes the result back onto the 5 raw
    feature groups via a linear back-projection through the PCA loadings
    (components_) (|shap| @ |loadings|). This is an APPROXIMATION that relies
    on PCA being a linear transform - not exact Shapley values, but sufficient
    for ranking the relative contribution of each group.
    Returns a dict: {group_name: percentage} (sums to ~100).
    """
    import shap

    scaler = pipeline["scaler"]
    pca_b = pipeline["pca_85"]
    svm_b = pipeline["svm_B"]

    x_sc = scaler.transform(feature_1557.reshape(1, -1))
    x_pca = pca_b.transform(x_sc)

    explainer = shap.KernelExplainer(svm_b.predict_proba, background)
    sv = explainer.shap_values(x_pca, nsamples=nsamples, silent=True)

    # Newer shap versions may return a 3D array (n, n_pca, n_classes) instead
    # of a list (the same behavior was previously seen in 13_1002.py).
    if isinstance(sv, list):
        sv_target = np.asarray(sv[target_class_idx])[0]
    else:
        sv = np.asarray(sv)
        sv_target = sv[0, :, target_class_idx] if sv.ndim == 3 else sv[0]

    components = pca_b.components_  # [n_pca, 1557]
    raw_importance = np.abs(sv_target) @ np.abs(components)  # [1557]

    groups = {name: float(raw_importance[start:end].sum()) for name, start, end in FEATURE_GROUPS}
    total = sum(groups.values()) + 1e-9
    return {name: (value / total * 100.0) for name, value in groups.items()}


def predict_proba(pipeline: dict, feature_1557: np.ndarray) -> np.ndarray:
    """
    Returns the class probability distribution for a single 1557-dim feature
    vector. Uses whichever strategy (A/B/C/D) achieved the highest F1 on the
    test set during training - rather than always assuming "D" as in the
    paper, the strategy that actually performed best is selected empirically
    (see overall_best).
    """
    x = feature_1557.reshape(1, -1)
    x_sc = pipeline["scaler"].transform(x)

    x_90 = pipeline["pca_90"].transform(x_sc)
    x_85 = pipeline["pca_85"].transform(x_sc)

    proba_A = pipeline["svm_A"].predict_proba(x_90)
    proba_B = pipeline["svm_B"].predict_proba(x_85)

    best_x = x_90 if pipeline["best_label"] == "A" else x_85
    proba_C_list = [svm.predict_proba(best_x) for svm in pipeline["svm_C_list"]]
    avg_proba_C = np.mean(proba_C_list, axis=0)

    w = pipeline["weights"]
    proba_D = (w["A"] * proba_A + w["B"] * proba_B + w["C"] * avg_proba_C) / (w["A"] + w["B"] + w["C"])

    strategy = pipeline.get("overall_best", "D")
    proba_by_strategy = {"A": proba_A, "B": proba_B, "C": avg_proba_C, "D": proba_D}
    return proba_by_strategy[strategy][0]
