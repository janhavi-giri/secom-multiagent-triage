"""
LangGraph Multi-Agent Gradio UI — uses ReAct agents + supervisor routing.

LLMs can only answer using tool outputs from the deterministic CaseFile.
"""

import gradio as gr

from .orchestrator import run_pipeline
from .langchain_tools import CaseStore, set_global_store
from .langgraph_agents import LangGraphOrchestrator

CASE_STORE = CaseStore()
set_global_store(CASE_STORE)


def ui_run_agents(case_id: str) -> str:
    case = run_pipeline(case_id.strip() or "SECOM-DEMO-001")
    CASE_STORE.put(case)
    auc = case.artifacts.get("detection_auc")
    fr = case.artifacts.get("raw_fail_rate")
    return (
        f"✅ Case loaded: {case.case_id}\n"
        f"- Overall fail rate: {fr:.3%}\n"
        f"- AUC (test window): {auc:.3f}\n\n"
        "Now ask questions in chat.\n"
        "Architecture: Supervisor → ReAct Specialists → Synthesizer (LangGraph)"
    )


def make_chat_fn(case_id_component, model_component):
    def _fn(message, history):
        case_id = case_id_component
        model = model_component
        if not case_id:
            return "Missing case_id. Run agents first."
        try:
            orch = LangGraphOrchestrator(model=model, use_llm_synthesizer=True)
            return orch.answer(case_id, message)
        except KeyError:
            return "No case loaded. Click **Run Agents** first."
        except Exception as e:
            return f"Error: {e}"
    return _fn


def ui_chat(message, history, case_id, model):
    if not case_id:
        return "Missing case_id. Run agents first."
    try:
        orch = LangGraphOrchestrator(model=model, use_llm_synthesizer=True)
        response = orch.answer(case_id.strip(), message)
        return response
    except KeyError:
        return "No case loaded. Click **Run Agents** first."
    except Exception as e:
        return f"Error: {e}"


def main():
    with gr.Blocks(title="LangGraph Multi-Agent Fab Triage (SECOM Demo)") as demo:
        gr.Markdown(
            """# LangGraph Multi-Agent Excursion Triage (SECOM Demo)

**Architecture:** Supervisor → ReAct Specialist Agents → Synthesizer (LangGraph + LangChain)

**Grounding:** LLM agents can only answer using tool outputs from the deterministic CaseFile.
**Guardrail:** No autonomous hold/release/tool-down actions.

**Dataset:** SECOM — UCI ML Repository (McCann & Johnston, 2008), DOI: 10.24432/C54305
"""
        )

        with gr.Row():
            case_id_in = gr.Textbox(label="Case ID", value="SECOM-DEMO-001", scale=2)
            model_in = gr.Dropdown(
                label="Model",
                choices=["gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"],
                value="gpt-4.1-mini",
                scale=1,
            )
            run_btn = gr.Button("Run Agents", variant="primary", scale=1)

        status = gr.Textbox(label="Status", value="Click 'Run Agents' to start.", lines=5)
        run_btn.click(ui_run_agents, inputs=[case_id_in], outputs=[status])

        chatbot = gr.Chatbot(label="Multi-Agent Q&A (LangGraph ReAct)", height=450)

        with gr.Row():
            user_in = gr.Textbox(
                label="Ask about the incident",
                placeholder="e.g., What's the summary? What are the top drivers? What should we do?",
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        def chat_wrapper(message, history, case_id, model):
            bot_response = ui_chat(message, history, case_id, model)
            history = history + [(message, bot_response)]
            return history, ""

        send_btn.click(
            fn=chat_wrapper,
            inputs=[user_in, chatbot, case_id_in, model_in],
            outputs=[chatbot, user_in],
        )
        user_in.submit(
            fn=chat_wrapper,
            inputs=[user_in, chatbot, case_id_in, model_in],
            outputs=[chatbot, user_in],
        )

        gr.Markdown(
            """### Quick Questions
Click any button to ask a pre-built question:"""
        )
        with gr.Row():
            for label, prompt in [
                ("Summary", "Give me the full incident summary"),
                ("Severity", "What is the severity level and fail rate?"),
                ("Blast Radius", "What is the blast radius? Which tools are suspect?"),
                ("Top Drivers", "What are the top drivers of fail risk?"),
                ("Actions", "What actions do you recommend?"),
                ("Evidence", "Show me the evidence and audit trail"),
            ]:
                btn = gr.Button(label)
                btn.click(
                    fn=chat_wrapper,
                    inputs=[gr.State(prompt), chatbot, case_id_in, model_in],
                    outputs=[chatbot, user_in],
                )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
