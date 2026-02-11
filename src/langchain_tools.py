"""
LangChain tools that wrap CaseFile accessors.

Each tool reads from the CaseFile blackboard — the LLM agents can ONLY
see data through these functions (tool-grounding).
"""

import json
import threading
from typing import Any, Dict, Optional

from langchain_core.tools import tool

from .casefile import CaseFile


# ────────────────────────── Thread-safe CaseStore ──────────────────────────


class CaseStore:
    """Thread-safe in-memory case store for concurrent Gradio sessions."""

    def __init__(self):
        self._store: Dict[str, CaseFile] = {}
        self._lock = threading.Lock()

    def put(self, case: CaseFile):
        with self._lock:
            self._store[case.case_id] = case

    def get(self, case_id: str) -> CaseFile:
        with self._lock:
            if case_id not in self._store:
                raise KeyError(f"Case '{case_id}' not found. Run agents first.")
            return self._store[case_id]

    def list_ids(self):
        with self._lock:
            return list(self._store.keys())


# ────────────────────────── Singleton store (set by UI at startup) ──────────────────────────

_CASE_STORE: Optional[CaseStore] = None


def set_global_store(store: CaseStore):
    global _CASE_STORE
    _CASE_STORE = store


def _get_case(case_id: str) -> CaseFile:
    if _CASE_STORE is None:
        raise RuntimeError("CaseStore not initialized. Call set_global_store() first.")
    return _CASE_STORE.get(case_id)


# ────────────────────────── LangChain @tool functions ──────────────────────────


@tool
def get_summary(case_id: str) -> str:
    """Get the incident summary, scope, and known limitations from the CaseFile."""
    case = _get_case(case_id)
    return json.dumps({
        "case_id": case.case_id,
        "scope": case.scope,
        "summary": case.artifacts.get("shift_summary", ""),
        "limitations": [
            "SECOM has no true MES genealogy; suspect tool/lot IDs are synthetic (demo).",
            "Features are anonymized x000..; real deployment must map to sensor tags.",
        ],
    }, default=str)


@tool
def get_metrics(case_id: str) -> str:
    """Get model performance metrics: AUC, confusion matrix, overall fail rate."""
    case = _get_case(case_id)
    return json.dumps({
        "auc": case.artifacts.get("detection_auc"),
        "confusion_matrix": case.artifacts.get("confusion_matrix"),
        "fail_rate_overall": case.artifacts.get("raw_fail_rate"),
        "detection_report": case.artifacts.get("detection_report"),
    }, default=str)


@tool
def get_top_drivers(case_id: str) -> str:
    """Get top risk-driver features (permutation importance) from the CaseFile."""
    case = _get_case(case_id)
    return json.dumps({
        "top_drivers": case.artifacts.get("top_drivers", {}),
        "note": "Feature names are anonymized (x000..); map to real sensor tags in production.",
    }, default=str)


@tool
def get_suspect_tools(case_id: str) -> str:
    """Get suspect tools and blast-radius containment information."""
    case = _get_case(case_id)
    return json.dumps({
        "suspect_tools": case.artifacts.get("suspect_tools", []),
        "note": case.artifacts.get("blast_radius_note", ""),
    }, default=str)


@tool
def get_recommendations(case_id: str) -> str:
    """Get human-in-the-loop recommendations (actions, risks, required approvals)."""
    case = _get_case(case_id)
    recs = []
    for r in case.recommendations:
        recs.append({
            "action": r.action,
            "risk": r.risk,
            "approvals_required": r.approvals_required,
        })
    return json.dumps({"recommendations": recs}, default=str)


@tool
def get_audit_trail(case_id: str, n: int = 8) -> str:
    """Get audit trail (last N log entries) and any findings missing evidence links."""
    case = _get_case(case_id)
    return json.dumps({
        "missing_evidence_claims": case.artifacts.get("audit_missing_evidence_claims", []),
        "audit_log_tail": case.audit_log[-n:] if case.audit_log else [],
    }, default=str)


@tool
def get_severity(case_id: str) -> str:
    """Get severity assessment: heuristic severity level and predicted fail rate in scoring window."""
    import numpy as np
    case = _get_case(case_id)
    pred = np.array(case.artifacts.get("pred_test", []))
    if pred.size == 0:
        return json.dumps({"severity": "UNKNOWN", "pred_fail_rate_window": None})
    window_rate = float(pred.mean())
    if window_rate >= 0.20:
        sev = "HIGH"
    elif window_rate >= 0.10:
        sev = "MEDIUM"
    else:
        sev = "LOW"
    return json.dumps({"severity": sev, "pred_fail_rate_window": window_rate})


# ────────────────────────── Tool registry ──────────────────────────

ALL_TOOLS = [
    get_summary,
    get_metrics,
    get_top_drivers,
    get_suspect_tools,
    get_recommendations,
    get_audit_trail,
    get_severity,
]
