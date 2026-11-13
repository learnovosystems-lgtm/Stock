"""ML inference for signal reliability — trees first, never LLMs for prices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ati.config import MODELS_DIR, ensure_dirs
from ati.indicators.technical import enrich
from ati.strategies.base import RawSignal


@dataclass
class ModelPrediction:
    prob_target_before_stop: float
    expected_return: float
    expected_adverse: float
    expected_favourable: float
    confidence: float
    source: str


def build_feature_row(daily: pd.DataFrame, nifty: pd.DataFrame, signal: RawSignal) -> dict[str, float]:
    d = enrich(daily) if "atr_14" not in daily.columns else daily
    last = d.iloc[-1]
    nifty_ret = 0.0
    if nifty is not None and len(nifty) > 21:
        nifty_ret = float(nifty["Close"].iloc[-1] / nifty["Close"].iloc[-22] - 1)
    risk = abs(signal.ideal_entry - signal.stop) / max(signal.ideal_entry, 1e-9)
    reward = abs(signal.target_1 - signal.ideal_entry) / max(signal.ideal_entry, 1e-9)
    return {
        "ret_5": float(last.get("ret_5", 0) or 0),
        "ret_21": float(last.get("ret_21", 0) or 0),
        "ret_63": float(last.get("ret_63", 0) or 0),
        "rsi_14": float(last.get("rsi_14", 50) or 50),
        "adx": float(last.get("adx", 15) or 15),
        "vol_surge": float(last.get("vol_surge", 1) or 1),
        "rvol_20": float(last.get("rvol_20", 0.2) or 0.2),
        "bb_width": float(last.get("bb_width", 0.1) or 0.1),
        "nifty_ret_21": nifty_ret,
        "signal_score": float(signal.score),
        "risk_pct": float(risk),
        "reward_pct": float(reward),
        "rr": float(signal.risk_reward),
    }


FEATURE_ORDER = [
    "ret_5",
    "ret_21",
    "ret_63",
    "rsi_14",
    "adx",
    "vol_surge",
    "rvol_20",
    "bb_width",
    "nifty_ret_21",
    "signal_score",
    "risk_pct",
    "reward_pct",
    "rr",
]


class SignalModel:
    def __init__(self, bundle: dict | None = None):
        self.bundle = bundle

    @classmethod
    def load_default(cls) -> "SignalModel":
        ensure_dirs()
        path = MODELS_DIR / "signal_model.joblib"
        if path.exists():
            return cls(joblib.load(path))
        return cls(None)

    def predict(self, daily: pd.DataFrame, nifty: pd.DataFrame, signal: RawSignal) -> ModelPrediction:
        feats = build_feature_row(daily, nifty, signal)
        if self.bundle is None:
            return self._heuristic(feats, signal)

        try:
            x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)
            # Guard against NaNs from live features
            if not np.isfinite(x).all():
                return self._heuristic(feats, signal)
            clf = self.bundle["classifier"]
            reg = self.bundle.get("regressor")
            raw_prob = float(clf.predict_proba(x)[0][1])
            prior = self._heuristic(feats, signal)
            # Blend with transparent prior — tree models on path labels are weakly
            # calibrated (CV AUC ~0.55-0.60); never trust extreme raw probs alone.
            auc = float(self.bundle.get("cv_auc", 0.55))
            w = float(np.clip((auc - 0.5) / 0.2, 0.2, 0.7))
            prob = float(np.clip(w * raw_prob + (1 - w) * prior.prob_target_before_stop, 0.05, 0.9))
            exp_ret = float(reg.predict(x)[0]) if reg is not None else prior.expected_return
            exp_ret = 0.5 * exp_ret + 0.5 * prior.expected_return
            conf = min(0.9, max(0.45, auc))
            return ModelPrediction(
                prob_target_before_stop=prob,
                expected_return=exp_ret,
                expected_adverse=-feats["risk_pct"],
                expected_favourable=feats["reward_pct"],
                confidence=conf,
                source="trained_model_blended",
            )
        except Exception:  # noqa: BLE001
            return self._heuristic(feats, signal)

    def _heuristic(self, feats: dict[str, float], signal: RawSignal) -> ModelPrediction:
        # Transparent baseline until lab trains a model — still data-derived
        rr = max(feats["rr"], 0.1)
        momentum = np.tanh(feats["ret_21"] * 5)
        rs_proxy = np.tanh(feats["ret_63"] * 3)
        vol_pen = -0.15 if feats["rvol_20"] > 0.35 else 0.0
        base = 0.45 + 0.15 * momentum + 0.15 * rs_proxy + 0.1 * np.tanh(feats["signal_score"]) + 0.05 * np.tanh(rr - 2)
        prob = float(np.clip(base + vol_pen, 0.05, 0.9))
        exp_ret = prob * feats["reward_pct"] - (1 - prob) * feats["risk_pct"]
        return ModelPrediction(
            prob_target_before_stop=prob,
            expected_return=float(exp_ret),
            expected_adverse=-feats["risk_pct"],
            expected_favourable=feats["reward_pct"],
            confidence=0.55,
            source="heuristic_prior",
        )
