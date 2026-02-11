"""
Deterministic agents for the SECOM excursion triage pipeline.

Each agent reads/writes the CaseFile blackboard. No LLM calls here —
these are pure data-processing stages.
"""

import os
import zipfile
import hashlib
import datetime as dt
import urllib.request
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance

from .casefile import CaseFile, EvidenceItem, Finding, Recommendation


# ────────────────────────── Base ──────────────────────────


class BaseAgent:
    name: str = "BaseAgent"

    def log(self, case: CaseFile, msg: str, **kwargs):
        case.audit_log.append(
            {
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "agent": self.name,
                "msg": msg,
                "meta": kwargs,
            }
        )


# ────────────────────────── 1. Ingestion ──────────────────────────


class IngestionAgent(BaseAgent):
    """Downloads and parses the SECOM dataset from UCI."""

    name = "IngestionAgent"

    SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"

    def run(self, case: CaseFile) -> CaseFile:
        out_zip = "secom.zip"
        out_dir = "secom_data"
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(out_zip):
            self.log(case, "Downloading SECOM zip", url=self.SECOM_URL)
            urllib.request.urlretrieve(self.SECOM_URL, out_zip)

        self.log(case, "Extracting zip", zip_path=out_zip)
        with zipfile.ZipFile(out_zip, "r") as z:
            z.extractall(out_dir)

        data_path = os.path.join(out_dir, "secom.data")
        labels_path = os.path.join(out_dir, "secom_labels.data")
        if not (os.path.exists(data_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(
                f"Expected secom.data and secom_labels.data in {out_dir}. "
                f"Found: {os.listdir(out_dir)}"
            )

        X = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")
        y_raw = pd.read_csv(labels_path, sep=r"\s+", header=None, engine="python")

        y = y_raw.iloc[:, 0].astype(int)
        y_bin = (y == 1).astype(int)  # UCI: -1 = pass, +1 = fail

        ts = None
        if y_raw.shape[1] > 1:
            ts_parsed = pd.to_datetime(y_raw.iloc[:, 1], errors="coerce")
            if ts_parsed.notna().sum() > 0:
                ts = ts_parsed
        if ts is None:
            ts = pd.date_range("2008-01-01", periods=len(y_bin), freq="h")

        df = X.copy()
        df.columns = [f"x{i:03d}" for i in range(df.shape[1])]
        df["y_fail"] = y_bin.values
        df["timestamp"] = ts.values
        df = df.sort_values("timestamp").reset_index(drop=True)

        case.artifacts["raw_df_shape"] = df.shape
        case.artifacts["raw_fail_rate"] = float(df["y_fail"].mean())
        case.artifacts["df_raw"] = df
        self.log(
            case, "Loaded dataset",
            rows=df.shape[0], cols=df.shape[1],
            fail_rate=case.artifacts["raw_fail_rate"],
        )
        return case


# ────────────────────────── 2. Data Quality ──────────────────────────


class DataQualityAgent(BaseAgent):
    """Drops high-missing columns, builds preprocessing pipeline."""

    name = "DataQualityAgent"

    def __init__(self, missing_threshold: float = 0.30):
        self.missing_threshold = missing_threshold

    def run(self, case: CaseFile) -> CaseFile:
        df = case.artifacts["df_raw"].copy()
        X = df.drop(columns=["y_fail", "timestamp"])
        y = df["y_fail"].astype(int)

        missing_frac = X.isna().mean().sort_values(ascending=False)
        high_missing_cols = missing_frac[missing_frac > self.missing_threshold].index.tolist()

        pre_columns = [c for c in X.columns if c not in high_missing_cols]
        X2 = X[pre_columns]

        preprocess = Pipeline(steps=[
            ("var", VarianceThreshold(threshold=0.0)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ])

        case.artifacts["X_df"] = X2
        case.artifacts["y"] = y
        case.artifacts["preprocess"] = preprocess
        case.artifacts["dq_high_missing_dropped"] = high_missing_cols
        case.artifacts["dq_top_missing"] = missing_frac.head(10).to_dict()
        case.artifacts["dq_missing_summary"] = {
            "mean_missing_frac": float(X.isna().mean().mean()),
            "max_missing_frac": float(missing_frac.max()),
            "dropped_cols_count": len(high_missing_cols),
        }
        self.log(case, "Prepared preprocessing", dropped_high_missing=len(high_missing_cols))
        return case


# ────────────────────────── 3. Detection ──────────────────────────


class DetectionAgent(BaseAgent):
    """Trains logistic-regression risk model + IsolationForest anomaly baseline."""

    name = "DetectionAgent"

    def __init__(self, train_frac: float = 0.75, anomaly_contamination: float = 0.05):
        self.train_frac = train_frac
        self.anomaly_contamination = anomaly_contamination

    def run(self, case: CaseFile) -> CaseFile:
        df = case.artifacts["df_raw"].copy()
        X_df = case.artifacts["X_df"]
        y = case.artifacts["y"]
        preprocess = case.artifacts["preprocess"]

        split_idx = int(self.train_frac * len(df))
        X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        model = Pipeline(steps=[("pre", preprocess), ("clf", clf)])
        model.fit(X_train, y_train)

        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= 0.5).astype(int)
        auc = (
            roc_auc_score(y_test, proba_test)
            if len(np.unique(y_test)) > 1
            else float("nan")
        )

        # Anomaly baseline: fit on PASS-only training data
        X_train_pass = X_train[y_train == 0]
        X_train_pass_t = preprocess.fit_transform(X_train_pass)
        iso = IsolationForest(
            n_estimators=300,
            contamination=self.anomaly_contamination,
            random_state=7,
        )
        iso.fit(X_train_pass_t)
        X_test_t = preprocess.transform(X_test)
        anomaly_score = -iso.decision_function(X_test_t)

        case.artifacts.update({
            "model": model,
            "detection_auc": float(auc),
            "detection_report": classification_report(y_test, pred_test, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
            "proba_test": proba_test,
            "pred_test": pred_test,
            "y_test": y_test.values,
            "ts_test": df["timestamp"].iloc[split_idx:].values.astype("datetime64[ns]"),
            "anomaly_score": anomaly_score,
        })

        case.findings.append(Finding(
            claim=f"Risk model scored the most recent {1-self.train_frac:.0%} window; AUC={auc:.3f}.",
            confidence=0.8 if not np.isnan(auc) else 0.6,
            evidence=[EvidenceItem("metrics", {"auc": auc, "cm": case.artifacts["confusion_matrix"]})],
            assumptions=["Time ordering may be synthetic if timestamps are not parseable."],
        ))
        self.log(case, "Trained risk + anomaly models", auc=case.artifacts["detection_auc"])
        return case


# ────────────────────────── 4. Blast Radius ──────────────────────────


class BlastRadiusAgent(BaseAgent):
    """Synthesizes tool/lot IDs (SECOM has no MES genealogy) and finds suspect tools."""

    name = "BlastRadiusAgent"

    def run(self, case: CaseFile) -> CaseFile:
        df = case.artifacts["df_raw"].copy()
        X_df = case.artifacts["X_df"].copy()

        def stable_bucket(row: pd.Series, n_buckets: int, salt: str) -> int:
            vals = row.iloc[:20].fillna(0).astype(float).round(3).values.tobytes()
            h = hashlib.sha256(salt.encode("utf-8") + vals).hexdigest()
            return int(h[:8], 16) % n_buckets

        df["tool_id"] = [stable_bucket(X_df.iloc[i], 12, "tool") for i in range(len(df))]
        df["lot_id"] = [stable_bucket(X_df.iloc[i], 80, "lot") for i in range(len(df))]

        split_idx = int(0.75 * len(df))
        test_df = df.iloc[split_idx:].copy()
        test_df["risk"] = case.artifacts["proba_test"]
        test_df["is_pred_fail"] = case.artifacts["pred_test"]

        tool_stats = (
            test_df.groupby("tool_id")["is_pred_fail"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        suspect_tools = tool_stats.head(3).reset_index().to_dict(orient="records")

        case.artifacts["suspect_tools"] = suspect_tools
        case.artifacts["blast_radius_note"] = (
            "tool_id/lot_id are synthetic for SECOM demo; "
            "replace with MES genealogy in production."
        )

        case.findings.append(Finding(
            claim="Top suspect 'tools' (synthetic IDs) by elevated predicted fail rate in scoring window.",
            confidence=0.6,
            evidence=[EvidenceItem("suspect_tools", {"top3": suspect_tools})],
            assumptions=["SECOM has no MES genealogy; IDs are deterministic buckets for demo only."],
        ))
        self.log(case, "Computed suspect tools (synthetic)", top3=suspect_tools)
        return case


# ────────────────────────── 5. Hypothesis ──────────────────────────


class HypothesisAgent(BaseAgent):
    """Computes permutation-importance drivers for fail risk."""

    name = "HypothesisAgent"

    def run(self, case: CaseFile) -> CaseFile:
        df = case.artifacts["df_raw"]
        X_df = case.artifacts["X_df"]
        y = case.artifacts["y"]
        preprocess = case.artifacts["preprocess"]

        split_idx = int(0.75 * len(df))
        X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = Pipeline(steps=[
            ("pre", preprocess),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ])
        model.fit(X_train, y_train)

        r = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=7, scoring="roc_auc",
        )
        importances = pd.Series(r.importances_mean, index=X_df.columns).sort_values(ascending=False)
        top10 = importances.head(10)

        case.artifacts["top_drivers"] = top10.to_dict()
        case.hypotheses.append(Finding(
            claim="Top variables associated with fail risk (global permutation importance).",
            confidence=0.7,
            evidence=[EvidenceItem("top_drivers", {"top10": top10.to_dict()})],
            assumptions=["Feature names are anonymized (x000..); map to real sensor tags in production."],
        ))
        self.log(case, "Computed top drivers", top10=list(top10.index))
        return case


# ────────────────────────── 6. Action ──────────────────────────


class ActionAgent(BaseAgent):
    """Generates human-in-the-loop recommendations (never autonomous actions)."""

    name = "ActionAgent"

    def run(self, case: CaseFile) -> CaseFile:
        top_drivers = list(case.artifacts.get("top_drivers", {}).keys())[:5]
        suspect_tools = case.artifacts.get("suspect_tools", [])
        tool_ids = [d.get("tool_id") for d in suspect_tools]

        case.recommendations = [
            Recommendation(
                action=(
                    f"Containment (demo): HOLD lots associated with suspect "
                    f"tool_ids={tool_ids} until engineering review."
                ),
                risk="May increase cycle time / WIP aging; reduces escape risk.",
                approvals_required=["Process Engineering", "Production Control"],
                evidence=[EvidenceItem("suspect_tools", {"top3": suspect_tools})],
            ),
            Recommendation(
                action=(
                    f"Diagnostics: check drift/limit violations and recent changes "
                    f"for top signals {top_drivers} on suspect tools."
                ),
                risk="Engineering time; may require tool downtime for checks.",
                approvals_required=["Process Engineering", "Equipment Engineering"],
                evidence=[EvidenceItem("top_drivers", {"top5": top_drivers})],
            ),
            Recommendation(
                action=(
                    "Validation: run a short qual (monitor) after corrective action; "
                    "confirm risk returns to baseline."
                ),
                risk="Consumes capacity; increases metrology load.",
                approvals_required=["Process Engineering"],
                evidence=[],
            ),
        ]
        self.log(case, "Generated recommendations", n=len(case.recommendations))
        return case


# ────────────────────────── 7. Audit ──────────────────────────


class AuditAgent(BaseAgent):
    """Checks all findings/hypotheses have evidence links."""

    name = "AuditAgent"

    def run(self, case: CaseFile) -> CaseFile:
        problems: List[str] = []
        for f in case.findings + case.hypotheses:
            if not f.evidence:
                problems.append(f.claim)

        case.artifacts["audit_missing_evidence_claims"] = problems
        self.log(case, "Audit complete", missing_evidence=len(problems))
        return case


# ────────────────────────── 8. Communications ──────────────────────────


class CommsAgent(BaseAgent):
    """Generates a human-readable shift summary."""

    name = "CommsAgent"

    def run(self, case: CaseFile) -> CaseFile:
        fail_rate = case.artifacts.get("raw_fail_rate", 0)
        auc = case.artifacts.get("detection_auc", 0)
        suspect_tools = case.artifacts.get("suspect_tools", [])
        top_drivers = list(case.artifacts.get("top_drivers", {}).keys())[:5]

        summary = (
            f"INCIDENT SUMMARY — {case.case_id}\n"
            f"Scope: {case.scope}\n"
            f"Dataset: SECOM (UCI). Overall fail rate={fail_rate:.3%}.\n"
            f"Detection: risk model AUC={auc:.3f} on most recent window.\n"
            f"Suspect tools: {suspect_tools} (synthetic IDs for demo).\n"
            f"Top drivers: {top_drivers} (anonymized features x000..).\n"
            f"Guardrail: No autonomous hold/release/tool-down actions; "
            f"recommendations require human approval."
        )
        case.artifacts["shift_summary"] = summary
        self.log(case, "Generated shift summary")
        return case
