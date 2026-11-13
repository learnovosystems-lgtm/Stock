"""Hidden Markov regime classifier on NIFTY returns + realized vol + VIX."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ati.models.opportunity import MarketRegime


@dataclass
class HMMRegimeResult:
    regime: MarketRegime
    state: int
    confidence: float
    state_labels: dict[int, str]


def fit_predict_regime(
    nifty: pd.DataFrame,
    vix: pd.DataFrame | None = None,
    n_states: int = 3,
) -> HMMRegimeResult:
    """Gaussian-mixture HMM proxy (stable without hmmlearn dependency).

    States are labeled post-hoc by mean return / volatility.
    """
    close = nifty["Close"].astype(float)
    ret = np.log(close / close.shift(1))
    vol = ret.rolling(20).std() * np.sqrt(252)
    feat = pd.DataFrame({"ret": ret, "vol": vol}).dropna()
    if vix is not None and not vix.empty:
        aligned = feat.join(vix["Close"].rename("vix"), how="left").ffill().dropna()
    else:
        aligned = feat.copy()
        aligned["vix"] = 15.0
    if len(aligned) < 80:
        return HMMRegimeResult(MarketRegime.UNCERTAIN, -1, 0.0, {})

    x = aligned[["ret", "vol", "vix"]].values
    # Standardize
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    xz = (x - mu) / sd
    gmm = GaussianMixture(n_components=n_states, covariance_type="full", random_state=42)
    gmm.fit(xz)
    states = gmm.predict(xz)
    proba = gmm.predict_proba(xz)[-1]
    last_state = int(states[-1])

    # Label states by average return and vol
    labels: dict[int, str] = {}
    for s in range(n_states):
        mask = states == s
        mean_ret = float(aligned["ret"].values[mask].mean()) if mask.any() else 0.0
        mean_vol = float(aligned["vol"].values[mask].mean()) if mask.any() else 0.0
        if mean_vol > np.median(aligned["vol"]) * 1.25:
            labels[s] = "HIGH_VOL"
        elif mean_ret > 0:
            labels[s] = "BULL"
        else:
            labels[s] = "BEAR"

    # If a state is neither clearly bull nor bear and low vol → sideways
    for s in range(n_states):
        mask = states == s
        mean_ret = float(aligned["ret"].values[mask].mean()) if mask.any() else 0.0
        mean_vol = float(aligned["vol"].values[mask].mean()) if mask.any() else 0.0
        if abs(mean_ret) < aligned["ret"].std() * 0.15 and mean_vol <= np.median(aligned["vol"]):
            labels[s] = "SIDEWAYS"

    mapping = {
        "BULL": MarketRegime.BULL,
        "BEAR": MarketRegime.BEAR,
        "SIDEWAYS": MarketRegime.SIDEWAYS,
        "HIGH_VOL": MarketRegime.HIGH_VOL,
    }
    regime = mapping.get(labels.get(last_state, "SIDEWAYS"), MarketRegime.UNCERTAIN)
    return HMMRegimeResult(regime, last_state, float(proba[last_state]), labels)
