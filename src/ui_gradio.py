"""
Deterministic Gradio Ops Console — no LLM required.
Answers are grounded purely in CaseFile artifacts.
"""

import gradio as gr

from .orchestrator import run_pipeline
from .qa import answer, header, severity

# Use a dict instead of a bare global for slightly better isolation
_state = {"case": None}


def ui_run_agents(case_id: str):
    case = run_pipeline(case_id.strip() or "SECOM-DEMO-001")
    _state["case"] = case
    sev = severity(case)
    status = (
        f"✅ Agents complete.\n"
        f"{header(case)}\n"
        f"Severity (heuristic): {sev['severity']} | "
        f"Pred fail rate in window: {sev['pred_fail_rate_window']}\n\n"
        f"References:\n"
        f"- SECOM dataset: UCI ML Repository (McCann & Johnston, 2008), DOI: 10.24432/C54305\n"
        f"- https://archive.ics.uci.edu/ml/datasets/SECOM\n"
    )
    return status, []


def ui_chat(user_msg, chat_history):
    bot = answer(_state["case"], user_msg)
    chat_history = chat_history + [(user_msg, bot)]
    return chat_history, ""


def ui_quick(prompt: str, chat_history):
    return ui_chat(prompt, chat_history)


def main():
    with gr.Blocks(title="SECOM Multi-Agent Excursion Triage (Ops Console)") as demo:
        gr.Markdown(
            """# SECOM Multi-Agent Excursion Triage — Ops Console (Public Demo)
**Grounding:** Answers come *only* from deterministic agent outputs (no LLM).
**Guardrail:** No autonomous hold/release/tool-down actions.

**Dataset:** SECOM — UCI ML Repository (McCann & Johnston, 2008), DOI: 10.24432/C54305
https://archive.ics.uci.edu/ml/datasets/SECOM
"""
        )

        with gr.Row():
            case_id_in = gr.Textbox(label="Case ID", value="SECOM-DEMO-001", scale=3)
            run_btn = gr.Button("Run Agents", variant="primary", scale=1)

        status = gr.Textbox(label="Status", value="Click 'Run Agents' to start.", lines=6)

        with gr.Row():
            b1 = gr.Button("Summary")
            b2 = gr.Button("Severity")
            b3 = gr.Button("Blast Radius / HOLD")
            b4 = gr.Button("Suspect Tools")
            b5 = gr.Button("Top Drivers")
            b6 = gr.Button("Actions")
            b7 = gr.Button("Evidence")
            b8 = gr.Button("Performance")

        chatbot = gr.Chatbot(label="Q&A (grounded in CaseFile)", height=380)

        with gr.Row():
            user_in = gr.Textbox(
                label="Ask a question",
                placeholder="e.g., What's the blast radius? What are the top drivers?",
            )
            send_btn = gr.Button("Send")

        run_btn.click(fn=ui_run_agents, inputs=[case_id_in], outputs=[status, chatbot])
        send_btn.click(fn=ui_chat, inputs=[user_in, chatbot], outputs=[chatbot, user_in])
        user_in.submit(fn=ui_chat, inputs=[user_in, chatbot], outputs=[chatbot, user_in])

        b1.click(fn=ui_quick, inputs=[gr.State("summary"), chatbot], outputs=[chatbot, user_in])
        b2.click(fn=ui_quick, inputs=[gr.State("severity"), chatbot], outputs=[chatbot, user_in])
        b3.click(fn=ui_quick, inputs=[gr.State("blast radius"), chatbot], outputs=[chatbot, user_in])
        b4.click(fn=ui_quick, inputs=[gr.State("suspect tools"), chatbot], outputs=[chatbot, user_in])
        b5.click(fn=ui_quick, inputs=[gr.State("top drivers"), chatbot], outputs=[chatbot, user_in])
        b6.click(fn=ui_quick, inputs=[gr.State("recommendations"), chatbot], outputs=[chatbot, user_in])
        b7.click(fn=ui_quick, inputs=[gr.State("evidence"), chatbot], outputs=[chatbot, user_in])
        b8.click(fn=ui_quick, inputs=[gr.State("performance"), chatbot], outputs=[chatbot, user_in])

    demo.launch(share=True)


if __name__ == "__main__":
    main()
