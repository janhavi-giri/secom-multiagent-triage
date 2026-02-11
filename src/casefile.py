"""
CaseFile: shared blackboard contract for the multi-agent triage pipeline.

All agents read/write through this structure, ensuring auditability
and a single source of truth.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class EvidenceItem:
    name: str
    details: Dict[str, Any]


@dataclass
class Finding:
    claim: str
    confidence: float
    evidence: List[EvidenceItem] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)


@dataclass
class Recommendation:
    action: str
    risk: str
    approvals_required: List[str]
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class CaseFile:
    case_id: str
    scope: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    hypotheses: List[Finding] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    # ── Serialization (excludes non-JSON-safe artifacts like models / DataFrames) ──

    _SKIP_ARTIFACT_KEYS = frozenset([
        "df_raw", "X_df", "y", "preprocess", "model",
        "proba_test", "pred_test", "y_test", "ts_test", "anomaly_score",
    ])

    def to_serializable_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (drops heavy/non-serializable artifacts)."""
        d = asdict(self)
        d["artifacts"] = {
            k: v for k, v in d["artifacts"].items()
            if k not in self._SKIP_ARTIFACT_KEYS
        }
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_serializable_dict(), default=str, **kwargs)
