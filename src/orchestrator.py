"""
Orchestrator: runs the deterministic agent pipeline in sequence.
"""

from .casefile import CaseFile
from .agents import (
    IngestionAgent,
    DataQualityAgent,
    DetectionAgent,
    BlastRadiusAgent,
    HypothesisAgent,
    ActionAgent,
    AuditAgent,
    CommsAgent,
)


def default_agents():
    """Factory — returns a fresh list each call (avoids shared-state bugs)."""
    return [
        IngestionAgent(),
        DataQualityAgent(missing_threshold=0.30),
        DetectionAgent(train_frac=0.75, anomaly_contamination=0.05),
        BlastRadiusAgent(),
        HypothesisAgent(),
        ActionAgent(),
        AuditAgent(),
        CommsAgent(),
    ]


def run_pipeline(case_id: str = "SECOM-DEMO-001", agents=None) -> CaseFile:
    if agents is None:
        agents = default_agents()

    case = CaseFile(
        case_id=case_id,
        scope={
            "workflow": "excursion_triage",
            "dataset": "SECOM (UCI ML Repository)",
            "window": "most recent 25% of samples (time-ordered)",
            "note": "tool_id/lot_id are synthetic for public demo only",
        },
    )

    for agent in agents:
        case = agent.run(case)
    return case
