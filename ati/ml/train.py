"""Train baseline ML models on labeled historical setups (walk-forward friendly)."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ati.config import MODELS_DIR, ensure_dirs
from ati.indicators.technical import enrich
from ati.ml.inference import FEATURE_ORDER


@dataclass
class TrainReport:
    samples: int
    auc: float
    path: str
    anomaly_rate: float


def _label_paths(df: pd.DataFrame, horizon: int = 10, rr: float = 2.0) -> pd.DataFrame:
    """Label whether ATR-based target is hit before stop over horizon (path dependent)."""
    d = enrich(df)
    rows = []
    for i in range(60, len(d) - horizon):
        row = d.iloc[i]
        if pd.isna(row.get("atr_14")) or row["atr_14"] <= 0:
            continue
        entry = float(row["Close"])
        stop = entry - 2.0 * float(row["atr_14"])
        target = entry + rr * (entry - stop)
        future = d.iloc[i + 1 : i + 1 + horizon]
        hit_target = False
        hit_stop = False
        for _, bar in future.iterrows():
            if bar["Low"] <= stop:
                hit_stop = True
                break
            if bar["High"] >= target:
                hit_target = True
                break
        y = 1 if hit_target and not hit_stop else 0
        fwd = float(d["Close"].iloc[i + horizon] / entry - 1)
        feats = {
            "ret_5": float(row.get("ret_5", 0) or 0),
            "ret_21": float(row.get("ret_21", 0) or 0),
            "ret_63": float(row.get("ret_63", 0) or 0),
            "rsi_14": float(row.get("rsi_14", 50) or 50),
            "adx": float(row.get("adx", 15) or 15),
            "vol_surge": float(row.get("vol_surge", 1) or 1),
            "rvol_20": float(row.get("rvol_20", 0.2) or 0.2),
            "bb_width": float(row.get("bb_width", 0.1) or 0.1),
            "nifty_ret_21": 0.0,
            "signal_score": float(min(1.5, max(0.0, (row.get("ret_63", 0) or 0) / 0.15))),
            "risk_pct": (entry - stop) / entry,
            "reward_pct": (target - entry) / entry,
            "rr": rr,
            "y": y,
            "fwd_ret": fwd,
        }
        rows.append(feats)
    return pd.DataFrame(rows)


def train_from_frames(frames: dict[str, pd.DataFrame], nifty: pd.DataFrame | None = None) -> TrainReport:
    ensure_dirs()
    parts = []
    for sym, df in frames.items():
        lab = _label_paths(df)
        if lab.empty:
            continue
        if nifty is not None and len(nifty) > 22:
            # approximate align by index position not calendar — good enough for baseline
            nret = float(nifty["Close"].iloc[-1] / nifty["Close"].iloc[-22] - 1)
            lab["nifty_ret_21"] = nret
        parts.append(lab)
    if not parts:
        raise RuntimeError("No training samples")
    data = pd.concat(parts, ignore_index=True)
    data[FEATURE_ORDER] = data[FEATURE_ORDER].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=FEATURE_ORDER + ["y", "fwd_ret"])
    if data.empty:
        raise RuntimeError("No clean training samples after NaN filter")
    x = data[FEATURE_ORDER].values.astype(float)
    y = data["y"].values.astype(int)
    y_reg = data["fwd_ret"].values.astype(float)

    # Prefer GBM; fall back gracefully
    clf: Pipeline | GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
        from xgboost import XGBRegressor

        clf_model = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            verbosity=0,
        )
        reg_model = XGBRegressor(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            verbosity=0,
        )
    except Exception:  # noqa: BLE001
        clf_model = GradientBoostingClassifier(random_state=42)
        reg_model = GradientBoostingRegressor(random_state=42)

    # Time-series CV AUC
    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for train_idx, test_idx in tscv.split(x):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf_model.fit(x[train_idx], y[train_idx])
        proba = clf_model.predict_proba(x[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    auc = float(np.mean(aucs)) if aucs else 0.5

    clf_model.fit(x, y)
    reg_model.fit(x, y_reg)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(x)
    anomaly_rate = float((iso.predict(x) == -1).mean())

    # Also fit calibrated logistic baseline for interpretability dump
    logit = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    if len(np.unique(y)) > 1:
        logit.fit(x, y)

    path = MODELS_DIR / "signal_model.joblib"
    joblib.dump(
        {
            "classifier": clf_model,
            "regressor": reg_model,
            "isolation_forest": iso,
            "logistic": logit,
            "feature_order": FEATURE_ORDER,
            "cv_auc": auc,
        },
        path,
    )
    return TrainReport(samples=len(data), auc=auc, path=str(path), anomaly_rate=anomaly_rate)
