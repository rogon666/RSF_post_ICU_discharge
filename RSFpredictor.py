from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib


@dataclass
class RSFBundle:
    estimator: Any
    feature_names: list[str]
    horizon_days: int
    age_group: str
    risk_cutoffs: Dict[str, float]  # {"q25":..., "q50":..., "q75":...}

    @staticmethod
    def load(path: str | Path) -> "RSFBundle":
        obj = joblib.load(path)
        return RSFBundle(
            estimator=obj["estimator"],
            feature_names=list(obj["feature_names"]),
            horizon_days=int(obj["horizon_days"]),
            age_group=str(obj["age_group"]),
            risk_cutoffs=dict(obj["risk_cutoffs"]),
        )

    def predict_one(self, x: Dict[str, Any]) -> Dict[str, Any]:
        # Build aligned 1-row dataframe in the exact training order
        X = pd.DataFrame([x])
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")

        X = X[self.feature_names]

        # survival function at horizon
        sf = self.estimator.predict_survival_function(X, return_array=True)  # shape (1, n_times)
        # map to closest time index
        times = np.asarray(self.estimator.event_times_)
        idx = int(np.argmin(np.abs(times - self.horizon_days)))
        surv = float(sf[0, idx])
        risk = float(1.0 - surv)

        # 4 risk categories based on stored cutoffs
        q25, q50, q75 = self.risk_cutoffs["q25"], self.risk_cutoffs["q50"], self.risk_cutoffs["q75"]
        if risk <= q25:
            cat = "low"
        elif risk <= q50:
            cat = "medium-low"
        elif risk <= q75:
            cat = "medium-high"
        else:
            cat = "high"

        return {
            "horizon_days": self.horizon_days,
            "age_group": self.age_group,
            "survival_probability": surv,
            "mortality_risk": risk,
            "risk_category": cat,
        }


def age_group_from_age(age: float) -> str:
    if age < 25:
        return "age0_24"
    if age < 65:
        return "age25_64"
    return "age65plus"


class RSFPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)

        self.models: Dict[Tuple[int, str], RSFBundle] = {}
        # adjust filenames if yours differ
        mapping = {
            (30, "age0_24"): "rsf_30d_age0_24.joblib",
            (30, "age25_64"): "rsf_30d_age25_64.joblib",
            (30, "age65plus"): "rsf_30d_age65plus.joblib",
            (60, "age0_24"): "rsf_60d_age0_24.joblib",
            (60, "age25_64"): "rsf_60d_age25_64.joblib",
            (60, "age65plus"): "rsf_60d_age65plus.joblib",
        }
        for key, fname in mapping.items():
            self.models[key] = RSFBundle.load(model_dir / fname)

    def predict(self, x: Dict[str, Any], horizon_days: int) -> Dict[str, Any]:
        if "age" not in x:
            raise ValueError("Input must include 'age' to select age-stratum model.")
        group = age_group_from_age(float(x["age"]))
        bundle = self.models[(int(horizon_days), group)]
        return bundle.predict_one(x)
