# modules/dynamic/detector.py
"""
Dynamic (behavioural) malware detector for IIoT network traffic.

Model
  A Random Forest over the flow features from features.py. Chosen because:
    - it handles mixed-scale, heavy-tailed network features without scaling,
    - class_weight='balanced_subsample' copes with IoT-23's extreme class
      imbalance (millions of scan flows vs a handful of C&C flows),
    - it provides feature importances, which the dashboard uses as the
      "feature importance" explanation promised for dynamic detections
      (Grad-CAM covers the static CNN; this covers the behavioural side).

Two levels of output
  1. Per flow:   behaviour class + class probabilities + P(malicious)
  2. Per capture (one run of one sample, or one Digital Twin session):
       capture_score      — mean P(malicious) of the top 5% most suspicious
                            flows (min 1). Robust to one noisy flow but still
                            sensitive when only a minority of traffic is bad.
       behaviour_profile  — counts of each malicious behaviour detected
       dominant_behaviour — most frequent malicious behaviour
       explanation        — features where flagged flows deviate most from
                            the benign baseline learned at training time
       mitre              — ATT&CK for ICS techniques for the behaviours seen
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, classification_report,
)

from .features import extract_flow_features, feature_names, FEATURE_DESCRIPTIONS

BENIGN = 'Benign'
MODEL_FORMAT_VERSION = 1


@dataclass
class DynamicReport:
    n_flows: int
    n_malicious_flows: int
    malicious_fraction: float
    capture_score: float
    verdict: str
    behaviour_profile: dict
    dominant_behaviour: Optional[str]
    explanation: list = field(default_factory=list)
    mitre: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'n_flows': self.n_flows,
            'n_malicious_flows': self.n_malicious_flows,
            'malicious_fraction': self.malicious_fraction,
            'capture_score': self.capture_score,
            'verdict': self.verdict,
            'behaviour_profile': self.behaviour_profile,
            'dominant_behaviour': self.dominant_behaviour,
            'explanation': self.explanation,
            'mitre': self.mitre,
        }


def load_behaviour_mitre(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def merge_mitre(behaviours: list[str], db: dict) -> dict:
    tactics, techniques, seen = [], [], set()
    for b in behaviours:
        entry = db.get(b, {})
        for t in entry.get('tactics', []):
            if t not in tactics:
                tactics.append(t)
        for tech in entry.get('techniques', []):
            if tech['id'] not in seen:
                seen.add(tech['id'])
                techniques.append(tech)
    return {'tactics': tactics, 'techniques': techniques}


class DynamicDetector:
    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = 25,
                 random_state: int = 42, n_jobs: int = -1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced_subsample',
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names = feature_names()
        self.classes_: list[str] = []
        self.benign_mean: Optional[np.ndarray] = None
        self.benign_std: Optional[np.ndarray] = None
        self.malicious_threshold = 0.5

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, flows: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DynamicDetector':
        """Fit on parsed conn.log rows. Labels come from `y` or flows['behaviour']."""
        if y is None:
            if 'behaviour' not in flows.columns:
                raise ValueError("Training data needs a 'behaviour' column (labelled conn.log)")
            y = flows['behaviour']
        X = extract_flow_features(flows)
        y = pd.Series(y).astype(str).values
        if len(set(y)) < 2:
            raise ValueError("Need at least two behaviour classes to train")
        self.model.fit(X.values, y)
        self.classes_ = [str(c) for c in self.model.classes_]

        benign_rows = X.values[y == BENIGN]
        ref = benign_rows if len(benign_rows) else X.values
        self.benign_mean = ref.mean(axis=0)
        self.benign_std = ref.std(axis=0) + 1e-6
        return self

    # ── inference ───────────────────────────────────────────────────────────
    def _check_fitted(self):
        if not self.classes_:
            raise RuntimeError("DynamicDetector is not trained — run scripts/train_dynamic.py")

    def predict_proba_df(self, flows: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X = extract_flow_features(flows)
        proba = self.model.predict_proba(X.values)
        out = pd.DataFrame(proba, columns=self.classes_, index=flows.index)
        out['malicious_prob'] = 1.0 - out[BENIGN] if BENIGN in out else 1.0
        out['behaviour_pred'] = np.array(self.classes_)[proba.argmax(axis=1)]
        return out

    def analyse_capture(self, flows: pd.DataFrame, mitre_db: Optional[dict] = None,
                        top_frac: float = 0.05) -> DynamicReport:
        """Capture-level verdict. See module docstring for the scoring rule."""
        pred = self.predict_proba_df(flows)
        mp = pred['malicious_prob'].values
        k = max(1, math.ceil(top_frac * len(mp)))
        capture_score = float(np.sort(mp)[-k:].mean()) if len(mp) else 0.0

        flagged = pred['malicious_prob'] >= self.malicious_threshold
        mal_behaviours = pred.loc[flagged, 'behaviour_pred']
        mal_behaviours = mal_behaviours[mal_behaviours != BENIGN]
        profile = {str(k): int(v) for k, v in mal_behaviours.value_counts().items()}
        dominant = max(profile, key=profile.get) if profile else None

        verdict = ('MALICIOUS' if capture_score >= 0.8 else
                   'SUSPICIOUS' if capture_score >= self.malicious_threshold else 'BENIGN')

        explanation = self.explain(flows.loc[flagged]) if flagged.any() else []
        mitre = merge_mitre(list(profile), mitre_db or {})

        return DynamicReport(
            n_flows=int(len(mp)),
            n_malicious_flows=int(flagged.sum()),
            malicious_fraction=float(flagged.mean()) if len(mp) else 0.0,
            capture_score=capture_score,
            verdict=verdict,
            behaviour_profile=profile,
            dominant_behaviour=dominant,
            explanation=explanation,
            mitre=mitre,
        )

    # ── explainability ──────────────────────────────────────────────────────
    def feature_importance(self, top: Optional[int] = None) -> list[dict]:
        self._check_fitted()
        imp = self.model.feature_importances_
        order = np.argsort(imp)[::-1]
        rows = [{'feature': self.feature_names[i], 'importance': float(imp[i]),
                 'description': FEATURE_DESCRIPTIONS.get(self.feature_names[i], '')}
                for i in order]
        return rows[:top] if top else rows

    def explain(self, flagged_flows: pd.DataFrame, top: int = 5) -> list[dict]:
        """
        Local explanation for a set of flagged flows: which features deviate
        most from the benign baseline, weighted by global importance.
        """
        self._check_fitted()
        if flagged_flows is None or len(flagged_flows) == 0:
            return []
        X = extract_flow_features(flagged_flows).values
        mean = X.mean(axis=0)
        z = (mean - self.benign_mean) / self.benign_std
        score = np.abs(z) * self.model.feature_importances_
        order = np.argsort(score)[::-1][:top]
        out = []
        for i in order:
            if score[i] <= 0:
                continue
            name = self.feature_names[i]
            out.append({
                'feature': name,
                'description': FEATURE_DESCRIPTIONS.get(name, ''),
                'flagged_mean': float(mean[i]),
                'benign_mean': float(self.benign_mean[i]),
                'z_score': float(z[i]),
                'direction': 'higher' if z[i] > 0 else 'lower',
            })
        return out

    # ── evaluation ──────────────────────────────────────────────────────────
    def evaluate(self, flows: pd.DataFrame, y: Optional[pd.Series] = None) -> dict:
        """Multi-class behaviour metrics + binary malicious-vs-benign metrics."""
        self._check_fitted()
        y_true = (pd.Series(y) if y is not None else flows['behaviour']).astype(str).values
        pred = self.predict_proba_df(flows)
        y_pred = pred['behaviour_pred'].values
        labels = self.classes_

        p_m, r_m, f_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', labels=labels, zero_division=0)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        pc = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0)
        per_class = {labels[i]: {'precision': float(pc[0][i]), 'recall': float(pc[1][i]),
                                 'f1': float(pc[2][i]), 'support': int(pc[3][i])}
                     for i in range(len(labels))}

        is_mal_true = (y_true != BENIGN).astype(int)
        is_mal_pred = (pred['malicious_prob'].values >= self.malicious_threshold).astype(int)
        tp = int(((is_mal_true == 1) & (is_mal_pred == 1)).sum())
        fp = int(((is_mal_true == 0) & (is_mal_pred == 1)).sum())
        tn = int(((is_mal_true == 0) & (is_mal_pred == 0)).sum())
        fn = int(((is_mal_true == 1) & (is_mal_pred == 0)).sum())
        try:
            bin_auc = float(roc_auc_score(is_mal_true, pred['malicious_prob'].values))
        except ValueError:
            bin_auc = None

        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(p_m), 'recall_macro': float(r_m), 'f1_macro': float(f_m),
            'precision_weighted': float(p_w), 'recall_weighted': float(r_w), 'f1_weighted': float(f_w),
            'per_class': per_class,
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            'labels': labels,
            'classification_report': classification_report(
                y_true, y_pred, labels=labels, zero_division=0),
            'binary': {
                'detection_rate': tp / (tp + fn) if (tp + fn) else None,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) else None,
                'precision': tp / (tp + fp) if (tp + fp) else None,
                'roc_auc': bin_auc,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            },
            'num_test_flows': int(len(y_true)),
        }

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path, metadata: Optional[dict] = None) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'format_version': MODEL_FORMAT_VERSION,
            'model': self.model,
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'benign_mean': self.benign_mean,
            'benign_std': self.benign_std,
            'malicious_threshold': self.malicious_threshold,
            'metadata': metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path) -> 'DynamicDetector':
        """
        Load a model saved by `save`. joblib uses pickle, so only load files
        produced by this project's own training script.
        """
        import joblib
        blob = joblib.load(Path(path))
        if blob.get('format_version') != MODEL_FORMAT_VERSION:
            raise ValueError("Unsupported dynamic model format — retrain with scripts/train_dynamic.py")
        if blob['feature_names'] != feature_names():
            raise ValueError("Feature set changed since this model was trained — retrain it")
        det = cls()
        det.model = blob['model']
        det.classes_ = blob['classes']
        det.benign_mean = blob['benign_mean']
        det.benign_std = blob['benign_std']
        det.malicious_threshold = blob.get('malicious_threshold', 0.5)
        det.metadata = blob.get('metadata', {})
        return det
