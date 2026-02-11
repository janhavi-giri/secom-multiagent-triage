"""
LangGraph multi-agent orchestration for SECOM excursion triage.

Architecture:
  ┌──────────────┐
  │  Supervisor   │  ← routes question to relevant specialist(s)
  └──────┬───────┘
         │ (conditional edges)
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
  OpsAgent  HypAgent  ActionAgent  EvidAgent   ← each is a ReAct agent
    │         │          │          │              with CaseFile tools
    └────┬────┴──────────┴──────────┘
         ▼
  ┌──────────────┐
  │  Synthesizer  │  ← merges specialist outputs into final answer
  └──────────────┘

Each specialist uses LangGraph's prebuilt `create_react_agent` which implements
the ReAct (Reason + Act) loop: the LLM reasons about what tool to call, calls it,
observes the result, and repeats until it has enough information to answer.
"""

import json
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

from .langchain_tools import ALL_TOOLS


# ──────────────────── Graph State ────────────────────


class TriageState(TypedDict):
    """Shared state flowing through the LangGraph."""
    question: str
    case_id: str
    # Which specialists should run (decided by supervisor)
    route: List[str]
    # Specialist outputs keyed by agent name
    specialist_outputs: Annotated[Dict[str, str], operator.ior]
    # Final synthesized answer
    final_answer: str


# ──────────────────── Specialist Configs ────────────────────

SPECIALIST_CONFIGS = {
    "ops_summary": {
        "system": (
            "You are the Ops Summary specialist for a semiconductor excursion triage system. "
            "Your role: summarize incident status, severity, and key metrics. "
            "You MUST use the provided tools to retrieve data — never invent numbers. "
            "If information is unavailable, say so explicitly. "
            "Always note limitations (SECOM: synthetic IDs, anonymized features)."
        ),
        "handles": ["summary", "severity", "overview", "status", "what happened", "incident"],
    },
    "hypothesis": {
        "system": (
            "You are the Hypothesis specialist for a semiconductor excursion triage system. "
            "Your role: interpret top drivers, propose falsifiable hypotheses, and explain "
            "which process variables are most associated with fail risk. "
            "You MUST use tools — never invent data. Note that SECOM features are anonymized."
        ),
        "handles": ["driver", "cause", "hypothesis", "why", "root cause", "variable", "feature"],
    },
    "action": {
        "system": (
            "You are the Action Planning specialist for a semiconductor excursion triage system. "
            "Your role: translate recommendations into a clear execution plan with approvals. "
            "You MUST use tools. Never suggest autonomous holds/releases — all actions require human approval."
        ),
        "handles": ["action", "recommend", "next step", "what should", "containment", "hold", "plan"],
    },
    "evidence": {
        "system": (
            "You are the Evidence & Audit specialist for a semiconductor excursion triage system. "
            "Your role: review audit trails, check evidence completeness, and report any gaps. "
            "You MUST use tools — never fabricate audit entries."
        ),
        "handles": ["evidence", "audit", "trace", "log", "proof", "missing"],
    },
}


# ──────────────────── Node Functions ────────────────────


def _build_react_agent(llm, system_prompt: str):
    """Build a ReAct agent using LangGraph's prebuilt create_react_agent."""
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=system_prompt,
    )


def supervisor_node(state: TriageState) -> dict:
    """
    Decide which specialists to invoke based on the question.
    Uses keyword matching first (fast, no API call), with fallback to all specialists.
    """
    q = state["question"].lower()
    matched = []

    for name, config in SPECIALIST_CONFIGS.items():
        if any(keyword in q for keyword in config["handles"]):
            matched.append(name)

    # Broad / ambiguous questions → run all specialists
    if not matched:
        matched = list(SPECIALIST_CONFIGS.keys())

    # Always include ops_summary for context if other specialists are selected
    if "ops_summary" not in matched and len(matched) < len(SPECIALIST_CONFIGS):
        matched.insert(0, "ops_summary")

    return {"route": matched}


def make_specialist_node(specialist_name: str, llm):
    """Factory: returns a node function that runs a ReAct agent for the given specialist."""
    config = SPECIALIST_CONFIGS[specialist_name]
    react_agent = _build_react_agent(llm, config["system"])

    def node_fn(state: TriageState) -> dict:
        case_id = state["case_id"]
        question = state["question"]

        # Invoke the ReAct agent
        result = react_agent.invoke({
            "messages": [
                HumanMessage(content=f"case_id={case_id}\n\nQuestion: {question}")
            ]
        })

        # Extract the final AI message
        messages = result.get("messages", [])
        final_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_text = msg.content
                break

        return {"specialist_outputs": {specialist_name: final_text}}

    return node_fn


def synthesizer_node(state: TriageState) -> dict:
    """Merge specialist outputs into a single coherent answer."""
    outputs = state.get("specialist_outputs", {})
    question = state["question"]

    if not outputs:
        return {"final_answer": "No specialist outputs available. Ensure agents have been run."}

    # Build synthesis (no extra LLM call for simple cases)
    parts = []
    for name, text in outputs.items():
        label = name.replace("_", " ").title()
        parts.append(f"**{label}:**\n{text}")

    combined = "\n\n---\n\n".join(parts)

    limitations = (
        "\n**Standard Limitations:**\n"
        "- SECOM has no real MES genealogy; tool/lot IDs are synthetic.\n"
        "- Feature names are anonymized (x000..); map to real sensor tags in production.\n"
        "- All recommendations require human approval — no autonomous actions."
    )

    return {"final_answer": f"{combined}\n\n{limitations}"}


def make_llm_synthesizer_node(llm):
    """Factory: returns a synthesizer that uses an LLM to merge specialist outputs."""

    def node_fn(state: TriageState) -> dict:
        outputs = state.get("specialist_outputs", {})
        question = state["question"]

        if not outputs:
            return {"final_answer": "No specialist outputs available."}

        specialist_text = "\n\n".join(
            f"[{name.replace('_', ' ').title()}]: {text}"
            for name, text in outputs.items()
        )

        messages = [
            SystemMessage(content=(
                "You are the Supervisor for a semiconductor excursion triage system. "
                "Synthesize the specialist outputs below into ONE concise, operations-ready answer. "
                "Rules: (1) Ground every claim in the specialist outputs — never invent data. "
                "(2) Deduplicate and organize clearly. (3) Always include limitations. "
                "(4) Note that all actions require human approval."
            )),
            HumanMessage(content=(
                f"User question: {question}\n\n"
                f"Specialist outputs:\n{specialist_text}\n\n"
                "Provide a unified, concise answer."
            )),
        ]

        resp = llm.invoke(messages)
        return {"final_answer": resp.content}

    return node_fn


# ──────────────────── Graph Builder ────────────────────


def build_triage_graph(
    model: str = "gpt-4.1-mini",
    use_llm_synthesizer: bool = True,
) -> StateGraph:
    """
    Build and compile the LangGraph multi-agent triage graph.

    Args:
        model: OpenAI model name (or any LangChain-compatible model string).
        use_llm_synthesizer: If True, use an LLM to synthesize specialist outputs.
                             If False, use simple concatenation (no extra API call).
    """
    llm = ChatOpenAI(model=model, temperature=0)

    graph = StateGraph(TriageState)

    # ── Add nodes ──
    graph.add_node("supervisor", supervisor_node)

    for name in SPECIALIST_CONFIGS:
        graph.add_node(name, make_specialist_node(name, llm))

    if use_llm_synthesizer:
        graph.add_node("synthesizer", make_llm_synthesizer_node(llm))
    else:
        graph.add_node("synthesizer", synthesizer_node)

    # ── Entry point ──
    graph.set_entry_point("supervisor")

    # ── Conditional routing from supervisor to specialists ──
    def route_specialists(state: TriageState) -> list[str]:
        return state["route"]

    # Supervisor → each specialist (conditionally)
    graph.add_conditional_edges(
        "supervisor",
        route_specialists,
        {name: name for name in SPECIALIST_CONFIGS},
    )

    # Each specialist → synthesizer
    for name in SPECIALIST_CONFIGS:
        graph.add_edge(name, "synthesizer")

    # Synthesizer → END
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ──────────────────── High-level API ────────────────────


class LangGraphOrchestrator:
    """High-level interface for the LangGraph multi-agent triage system."""

    def __init__(self, model: str = "gpt-4.1-mini", use_llm_synthesizer: bool = True):
        self.graph = build_triage_graph(
            model=model,
            use_llm_synthesizer=use_llm_synthesizer,
        )

    def answer(self, case_id: str, question: str) -> str:
        result = self.graph.invoke({
            "question": question,
            "case_id": case_id,
            "route": [],
            "specialist_outputs": {},
            "final_answer": "",
        })
        return result.get("final_answer", "No answer generated.")
