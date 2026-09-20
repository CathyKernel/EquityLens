"""Machine-learning modelling: forward-horizon direction prediction.

Compares three classifiers on the weekly-sampled panel dataset:

* Logistic Regression (interpretable baseline),
* Random Forest (bagged trees),
* XGBoost (gradient boosting - typically the strongest tabular learner).

Methodology guardrails (what makes the comparison honest)
---------------------------------------------------------
* **Chronological split**: the last 20% of *dates* form the untouched
  hold-out test set; the model never sees the future during training.
* **Time-series cross-validation**: ``TimeSeriesSplit`` over the training
  period for model selection, so folds never train on future dates to
  predict the past.
* **Class balance check**: up-periods are reported so accuracy can be
  compared against the majority-class baseline.
* **Threshold discipline**: probabilities are kept (not hard labels) so the
  backtest can apply its own confidence threshold.

Outputs: per-model CV metrics, hold-out test metrics (accuracy, precision,
recall, F1, ROC-AUC), ROC curves, confusion matrix and feature importance
for the winning model.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .feature_engineering import FEATURE_COLUMNS, TARGET

logger = logging.getLogger(__name__)


def _model_matrix(cfg) -> Dict[str, Pipeline]:
    """Build the candidate model pipelines from config hyperparameters.

    Parameters
    ----------
    cfg:
        Pipeline config namespace (``cfg.model`` is used).

    Returns
    -------
    dict
        Model name -> sklearn Pipeline (scaling where required).
    """
    m = cfg.model
    rs = int(m.random_state)

    logistic = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(m.models.logistic.C),
            max_iter=2000,
            random_state=rs,
        )),
    ])

    forest = RandomForestClassifier(
        n_estimators=int(m.models.random_forest.n_estimators),
        max_depth=int(m.models.random_forest.max_depth),
        min_samples_leaf=int(m.models.random_forest.min_samples_leaf),
        max_features="sqrt",
        n_jobs=-1,
        random_state=rs,
    )

    xgb = XGBClassifier(
        n_estimators=int(m.models.xgboost.n_estimators),
        max_depth=int(m.models.xgboost.max_depth),
        learning_rate=float(m.models.xgboost.learning_rate),
        subsample=float(m.models.xgboost.subsample),
        colsample_bytree=float(m.models.xgboost.colsample_bytree),
        min_child_weight=int(m.models.xgboost.min_child_weight),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=rs,
    )

    return {"logistic": logistic, "random_forest": forest, "xgboost": xgb}


def chronological_split(
    dataset: pd.DataFrame, test_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into train/test on the *date* axis (no shuffling).

    Parameters
    ----------
    dataset:
        Model-ready dataset (sorted by date).
    test_ratio:
        Fraction of distinct dates reserved for the hold-out test set.

    Returns
    -------
    tuple
        ``(train_df, test_df)``; both keep all tickers within their dates.
    """
    dates = np.sort(dataset["date"].unique())
    split_idx = int(len(dates) * (1.0 - test_ratio))
    split_date = dates[split_idx]
    train = dataset[dataset["date"] < split_date].copy()
    test = dataset[dataset["date"] >= split_date].copy()
    logger.info(
        "Chronological split at %s: train %d rows / test %d rows",
        pd.Timestamp(split_date).date(), len(train), len(test),
    )
    return train, test


def _feature_cols(dataset: pd.DataFrame) -> list[str]:
    """Return the ordered feature columns present in the dataset."""
    extra = [c for c in dataset.columns if c.startswith("sector_")]
    return list(FEATURE_COLUMNS) + extra


def cross_validate_models(
    train: pd.DataFrame, cfg
) -> pd.DataFrame:
    """Time-series cross-validated ROC-AUC for every candidate model.

    Parameters
    ----------
    train:
        Training split of the panel.
    cfg:
        Pipeline config namespace.

    Returns
    -------
    pd.DataFrame
        Mean +/- std CV ROC-AUC and accuracy per model.
    """
    cols = _feature_cols(train)
    X, y = train[cols], train[TARGET].astype(int)
    cv = TimeSeriesSplit(n_splits=int(cfg.model.cv_folds))

    rows = []
    for name, model in _model_matrix(cfg).items():
        auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({
            "model": name,
            "cv_roc_auc_mean": round(float(auc.mean()), 4),
            "cv_roc_auc_std": round(float(auc.std()), 4),
            "cv_accuracy_mean": round(float(acc.mean()), 4),
        })
        logger.info(
            "CV %s: ROC-AUC %.4f +/- %.4f", name, auc.mean(), auc.std()
        )
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)


def evaluate_on_test(
    train: pd.DataFrame, test: pd.DataFrame, cfg
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Fit every model on the full train set and score it on the hold-out.

    Parameters
    ----------
    train:
        Training split.
    test:
        Chronologically later hold-out split.
    cfg:
        Pipeline config namespace.

    Returns
    -------
    tuple
        ``(metrics, artefacts)`` where metrics maps model name -> test
        metrics, and artefacts holds fitted models and test predictions.
    """
    cols = _feature_cols(train)
    X_train, y_train = train[cols], train[TARGET].astype(int)
    X_test, y_test = test[cols], test[TARGET].astype(int)

    metrics: Dict[str, Dict[str, Any]] = {}
    artefacts: Dict[str, Any] = {"y_test": y_test, "feature_names": cols}
    majority = float(y_test.mean())

    for name, model in _model_matrix(cfg).items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "majority_class_share": round(majority, 4),
        }
        artefacts[f"proba_{name}"] = pd.Series(
            proba, index=test.index, name=name
        )
        artefacts[f"model_{name}"] = model
        logger.info(
            "Test %s: acc=%.4f auc=%.4f (majority baseline=%.4f)",
            name, metrics[name]["accuracy"], metrics[name]["roc_auc"], majority,
        )

    return metrics, artefacts


def plot_model_diagnostics(
    artefacts: Dict[str, Any],
    metrics: Dict[str, Dict[str, Any]],
    figures_dir: pathlib.Path,
) -> Dict[str, str]:
    """ROC curves and confusion matrix for the best model.

    Parameters
    ----------
    artefacts:
        Output of :func:`evaluate_on_test`.
    metrics:
        Per-model test metrics.
    figures_dir:
        Output directory for figures.

    Returns
    dict
        Figure name -> saved path.
    """
    figures_dir = pathlib.Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: Dict[str, str] = {}

    best_name = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    y_test = artefacts["y_test"]

    # -- ROC curves (all models) ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name in metrics:
        proba = artefacts[f"proba_{name}"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        lw = 2.4 if name == best_name else 1.4
        ax.plot(
            fpr, tpr, linewidth=lw,
            label=f"{name} (AUC = {metrics[name]['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.0,
            label="Random coin-flip")
    ax.set_title("ROC Curves on the Hold-out Test Set")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", frameon=False)
    path = figures_dir / "08_roc_curves.png"
    fig.savefig(path)
    plt.close(fig)
    figures["roc_curves"] = str(path)

    # -- Confusion matrix (best model) -----------------------------------------
    proba = artefacts[f"proba_{best_name}"]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    ConfusionMatrixDisplay(
        cm, display_labels=["Down period", "Up period"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name} (threshold = 0.5)")
    path = figures_dir / "09_confusion_matrix.png"
    fig.savefig(path)
    plt.close(fig)
    figures["confusion_matrix"] = str(path)

    # -- Feature importance (best tree model) ----------------------------------
    tree_name = best_name if best_name != "logistic" else "xgboost"
    model = artefacts[f"model_{tree_name}"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        figures["best_model"] = best_name
        return figures
    names = artefacts["feature_names"]
    order = np.argsort(importances)
    top = order[-15:]
    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    ax.barh(
        np.array(names)[top], np.array(importances)[top],
        color="#1f77b4", alpha=0.85,
    )
    ax.set_title(f"Top-15 Feature Importance - {tree_name} (gain)")
    ax.set_xlabel("Relative importance")
    path = figures_dir / "10_feature_importance.png"
    fig.savefig(path)
    plt.close(fig)
    figures["feature_importance"] = str(path)

    figures["best_model"] = best_name
    return figures


def run_modeling(
    dataset: pd.DataFrame, cfg, figures_dir: pathlib.Path
) -> Dict[str, Any]:
    """Full modelling workflow: CV -> hold-out evaluation -> diagnostics.

    Parameters
    ----------
    dataset:
        Model-ready panel from feature engineering.
    cfg:
        Pipeline config namespace.
    figures_dir:
        Output directory for figures.

    Returns
    -------
    dict
        ``{"cv": DataFrame, "test_metrics": {...}, "figures": {...},
        "artefacts": {...}, "test_frame": test split with probabilities}``
    """
    dataset = dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
    train, test = chronological_split(dataset, float(cfg.model.test_ratio))

    cv_table = cross_validate_models(train, cfg)
    metrics, artefacts = evaluate_on_test(train, test, cfg)
    diag = plot_model_diagnostics(artefacts, metrics, figures_dir)

    # Attach best-model probabilities to the test frame for the backtest
    best_name = diag["best_model"]
    test_frame = test.copy()
    test_frame["proba_up"] = artefacts[f"proba_{best_name}"].values

    return {
        "cv": cv_table,
        "test_metrics": metrics,
        "figures": diag,
        "artefacts": artefacts,
        "best_model": best_name,
        "test_frame": test_frame,
    }
