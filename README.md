# SECOM Multi‑Agent Excursion Triage (Public Demo)

A **public, reproducible** demo of **data‑driven decision making** in a semiconductor manufacturing **excursion triage** workflow, built with **LangChain** and **LangGraph**.

## Two Modes

| Mode | Framework | LLM Required? | Entry Point |
|------|-----------|---------------|-------------|
| **A) Deterministic Ops Console** | Pure Python agents + keyword Q&A | No | `python -m src.ui_gradio` |
| **B) LangGraph Multi‑Agent** | ReAct agents + Supervisor routing | Yes (OpenAI) | `python -m src.ui_gradio_llm` |

> **Guardrail:** No autonomous irreversible actions (hold/release/tool-down). All outputs are recommendations requiring human approval.

---

## Architecture

### Deterministic Pipeline (runs in both modes)

```
IngestionAgent → DataQualityAgent → DetectionAgent → BlastRadiusAgent
    → HypothesisAgent → ActionAgent → AuditAgent → CommsAgent
```

Shared state is a **CaseFile** (blackboard pattern): `scope`, `artifacts`, `findings`, `hypotheses`, `recommendations`, `audit_log`.

### LangGraph Multi-Agent (Mode B)

```
┌──────────────┐
│  Supervisor   │  ← keyword routing (which specialists to invoke)
└──────┬───────┘
       │ (conditional edges)
  ┌────┴────┬────────────┬──────────┐
  ▼         ▼            ▼          ▼
OpsAgent  HypAgent  ActionAgent  EvidAgent   ← each is a ReAct agent
  │         │            │          │           (LangGraph create_react_agent)
  └────┬────┴────────────┴──────────┘
       ▼
┌──────────────┐
│  Synthesizer  │  ← merges specialist outputs into final answer
└──────────────┘
```

Each specialist uses the **ReAct pattern** (Reason + Act): the LLM reasons about which CaseFile tool to call, calls it, observes the result, and repeats until it has enough information to answer.

**Tool-grounding:** LLM agents can ONLY access data through defined CaseFile tool functions — they cannot invent metrics or data.

---

## Dataset Reference

- **SECOM dataset**: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. DOI: **10.24432/C54305**
- https://archive.ics.uci.edu/ml/datasets/SECOM

### Limitations (explicit)

- SECOM does **not** provide real fab genealogy (tool/chamber/lot from MES). This demo **synthesizes deterministic `tool_id`/`lot_id` buckets** for blast-radius illustration only.
- Feature names are anonymized (`x000..`) — map to real sensor tags in production.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### A) Deterministic Ops Console (no LLM)
```bash
python -m src.ui_gradio
```

### B) LangGraph Multi-Agent (ReAct + Supervisor)
```bash
export OPENAI_API_KEY="your-key"
python -m src.ui_gradio_llm
```

---

## Key Frameworks & Libraries

| Library | Role |
|---------|------|
| **LangChain** | Tool definitions (`@tool`), LLM abstraction (`ChatOpenAI`) |
| **LangGraph** | Agent graph (`StateGraph`), ReAct agents (`create_react_agent`), conditional routing |
| **scikit-learn** | Logistic regression, IsolationForest, permutation importance |
| **Gradio** | Interactive web UI |
| **Pydantic** | Data validation |

---

## Repo Layout

```
src/
├── casefile.py          # CaseFile + evidence contracts (with serialization)
├── agents.py            # Deterministic agents (ingestion → comms)
├── orchestrator.py      # Runs the deterministic pipeline
├── qa.py                # Keyword-based Q&A router (no LLM)
├── langchain_tools.py   # LangChain @tool wrappers for CaseFile accessors
├── langgraph_agents.py  # LangGraph ReAct agents + Supervisor + Synthesizer
├── ui_gradio.py         # Deterministic Gradio UI
└── ui_gradio_llm.py     # LangGraph multi-agent Gradio UI
```

---

## Safety & Governance

- LLM agents are constrained to **tool outputs only** (CaseFile accessors). If a fact isn't in the CaseFile, the agent must say it's unavailable.
- Recommendations always list required approvals and do **not** execute holds, tool-downs, or recipe changes.
- The Supervisor node routes questions to only the relevant specialists (not all four every time).

---

## Citation

> McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C54305.
