"""
Deterministic Q&A router — answers from CaseFile artifacts only (no LLM).
"""

import json
from typing import Optional

import numpy as np

from .casefile import CaseFile


def _fmt_pct(x):
    try:
        return f"{100 * float(x):.2f}%"
    except Exception:
        return "N/A"


def severity(case: CaseFile) -> dict:
    pred = np.array(case.artifacts.get("pred_test", []))
    if pred.size == 0:
        return {"severity": "UNKNOWN", "pred_fail_rate_window": None}
    window_rate = float(pred.mean())
    if window_rate >= 0.20:
        sev = "HIGH"
    elif window_rate >= 0.10:
        sev = "MEDIUM"
    else:
        sev = "LOW"
    return {"severity": sev, "pred_fail_rate_window": window_rate}


def header(case: CaseFile) -> str:
    fr = case.artifacts.get("raw_fail_rate")
    auc = case.artifacts.get("detection_auc")
    rows, cols = case.artifacts.get("raw_df_shape", ("?", "?"))
    return (
        f"Case **{case.case_id}** | Rows/Cols: **{rows}/{cols}** | "
        f"Overall fail rate: **{_fmt_pct(fr)}** | AUC (test window): **{auc:.3f}**"
    )


def answer(case: Optional[CaseFile], question: str) -> str:
    if case is None:
        return "No case has been run yet. Click **Run Agents** first."

    q = (question or "").strip().lower()
    if not q:
        return (
            "Try: summary | severity | blast radius | suspect tools | "
            "top drivers | recommendations | evidence | performance."
        )

    sev = severity(case)
    head = header(case)

    if any(k in q for k in ["summary", "incident", "what happened", "overview"]):
        return (
            f"{head}\n\n### Incident Summary\n"
            f"{case.artifacts.get('shift_summary', 'Summary not available.')}"
        )

    if any(k in q for k in ["severity", "how bad", "impact", "urgent"]):
        return (
            f"{head}\n\n### Severity\n"
            f"- Severity (heuristic): **{sev['severity']}**\n"
            f"- Predicted fail rate in scoring window: "
            f"**{_fmt_pct(sev['pred_fail_rate_window'])}**\n"
            f"- Guardrail: demo heuristic; not a fab-certified limit."
        )

    if any(k in q for k in ["blast", "radius", "hold", "containment"]):
        st = case.artifacts.get("suspect_tools", [])
        note = case.artifacts.get("blast_radius_note", "")
        tool_ids = [d.get("tool_id") for d in st]
        return (
            f"{head}\n\n### Blast Radius / Containment (Demo)\n"
            f"- Suspect tool_ids (synthetic): **{tool_ids}**\n"
            f"- Recommended: HOLD lots associated with these tool_ids "
            f"pending engineering review.\n\n"
            f"Evidence:\n```json\n{json.dumps(st, indent=2)}\n```\n"
            f"Note: {note}"
        )

    if any(k in q for k in ["suspect tool", "tools"]):
        st = case.artifacts.get("suspect_tools", [])
        return (
            f"{head}\n\n### Suspect Tools (Demo)\n"
            f"```json\n{json.dumps(st, indent=2)}\n```"
        )

    if any(k in q for k in ["driver", "drivers", "hypothesis", "root cause", "likely cause"]):
        td = case.artifacts.get("top_drivers", {})
        if not td:
            return "No top drivers available. Run agents first."
        top10 = list(td.items())[:10]
        lines = "\n".join([f"- **{k}**: {v:.6f}" for k, v in top10])
        return (
            f"{head}\n\n### Top Drivers (Permutation Importance, ΔROC-AUC)\n{lines}\n\n"
            "Assumption: feature names are anonymized (x000..); "
            "map to real sensor tags in production."
        )

    if any(k in q for k in ["action", "recommend", "next step", "what should we do"]):
        recs = case.recommendations or []
        if not recs:
            return "No recommendations available. Run agents first."
        out = [f"{head}\n\n### Recommended Actions (Human-in-the-loop)\n"]
        for i, r in enumerate(recs, 1):
            out.append(
                f"**{i}) {r.action}**\n"
                f"- Risk: {r.risk}\n"
                f"- Approvals: {', '.join(r.approvals_required)}\n"
            )
        return "\n".join(out)

    if any(k in q for k in ["evidence", "audit", "trace", "log", "why"]):
        missing = case.artifacts.get("audit_missing_evidence_claims", [])
        tail = case.audit_log[-10:] if case.audit_log else []
        return (
            f"{head}\n\n### Evidence & Audit Trail\n"
            f"- Findings missing evidence links: **{len(missing)}**\n"
            f"{('```json\\n' + json.dumps(missing, indent=2) + '\\n```') if missing else 'None'}\n\n"
            f"Audit log (tail):\n```json\n{json.dumps(tail, indent=2)}\n```"
        )

    if any(k in q for k in ["auc", "performance", "metrics", "confusion"]):
        auc = case.artifacts.get("detection_auc")
        cm = case.artifacts.get("confusion_matrix")
        return (
            f"{head}\n\n### Model Performance\n"
            f"- AUC: **{float(auc):.4f}**\n"
            f"- Confusion matrix [[TN, FP],[FN, TP]]: **{cm}**\n"
            f"- Note: Threshold=0.5; tune for fab risk tolerance."
        )

    return (
        "I can answer only from the case artifacts.\n"
        "Try: summary | severity | blast radius | suspect tools | "
        "top drivers | recommendations | evidence | performance."
    )
