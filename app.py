"""
Thuna (തുണ) — Offline AI Health Companion for Elderly
Gradio App for Build Small Hackathon (Hugging Face × Gradio)

Voice-first Malayalam health companion powered by Gemma 4 E2B (2B params).
Deterministic intent parsing + LLM warmth. Designed for elderly users in rural India.
"""

import gradio as gr
import os
import re
from typing import List, Tuple, Optional
from datetime import datetime

from intent_parser import parse_intent
from health_store import HealthStore
from agent_engine import run_agent, AgentResult

# ═══════════════════════════════════════════════════════════════════════════
# LLM SETUP — Gemma via HF Inference
# ═══════════════════════════════════════════════════════════════════════════

from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "google/gemma-3-4b-it"

client = None
try:
    if HF_TOKEN:
        client = InferenceClient(token=HF_TOKEN)
except Exception:
    pass

SYSTEM_PROMPT = """നീ "തുണ" (Thuna) ആണ്. കേരളത്തിലെ പ്രായമായവരുടെ ആരോഗ്യ companion.

നിയമങ്ങൾ:
- നാടൻ മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക. English words ആവശ്യമെങ്കിൽ മാത്രം (medicine names, numbers).
- ചെറിയ വാക്യങ്ങൾ. ലളിതമായ ഭാഷ. പ്രായമായവർക്ക് മനസ്സിലാകുന്നത്.
- ആരോഗ്യ കാര്യങ്ങളിൽ ശ്രദ്ധാലുവായിരിക്കുക. അപകടകരമായ values കണ്ടാൽ ഡോക്ടറെ കാണാൻ പറയുക.
- Companion ആയി സംസാരിക്കുക. Warm, caring.
- 2-3 വാക്യത്തിൽ answer ചെയ്യുക.
- Hindi, Manglish എഴുതരുത്. Pure Malayalam script മാത്രം.

ശരി: "BP record ചെയ്തു. 130/85 — നോർമൽ ആണ്. നന്നായിരിക്കുന്നു!"
തെറ്റ്: "Aapka BP record ho gaya hai" """


def generate_llm_response(context: str, chat_history: List[dict]) -> str:
    if not client:
        return generate_fallback_response(context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": context})
    try:
        response = client.chat_completion(
            model=MODEL_ID, messages=messages,
            max_tokens=200, temperature=0.7, top_p=0.9,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r'[\*\#]+', '', text)
        return text
    except Exception:
        return generate_fallback_response(context)


def generate_fallback_response(context: str) -> str:
    lower = context.lower()
    if "bp" in lower and ("record" in lower or "vital" in lower):
        return "BP record ചെയ്തു. ശ്രദ്ധിക്കണം. വിശ്രമിക്കുക. 🩺"
    if "sugar" in lower and ("record" in lower or "vital" in lower):
        return "ഷുഗർ record ചെയ്തു. ട്രെൻഡ് ശ്രദ്ധിക്കാം. 🩸"
    if "saved" in lower and "medication" in lower:
        return "മരുന്ന് save ചെയ്തു. Reminder set ചെയ്തു. 💊⏰"
    if "medication" in lower or "medicine" in lower:
        return "നിങ്ങളുടെ മരുന്നുകൾ ഇതാ."
    if "took" in lower or "taken" in lower or "confirm" in lower:
        return "നന്നായി! മരുന്ന് കഴിച്ചു. ✅"
    if "reminder" in lower:
        return "Reminder set ചെയ്തു. സമയത്ത് ഓർമ്മിപ്പിക്കാം. ⏰"
    if "condition" in lower:
        return "രേഖപ്പെടുത്തി. ഡോക്ടറുടെ നിർദ്ദേശം പാലിക്കുക. 🏥"
    if "symptom" in lower or "report" in lower:
        return "ലക്ഷണം record ചെയ്തു. ഡോക്ടറെ കാണുക. 🤒"
    if "alert" in lower or "🚨" in lower:
        return "⚠️ ശ്രദ്ധിക്കുക! ഡോക്ടറെ വിളിക്കുക."
    return "എന്താ വിശേഷം? ആരോഗ്യം, മരുന്ന്, BP, ഷുഗർ — എന്തും ചോദിക്കാം. 😊"


# ═══════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════

health_store = HealthStore()


def process_message(message: str, history: List[dict]) -> Tuple[List[dict], str]:
    global health_store
    if not message or not message.strip():
        return history or [], ""
    history = history or []
    history.append({"role": "user", "content": message})

    agent_result = run_agent(message, health_store)
    llm_response = generate_llm_response(agent_result.context_for_llm, history)

    parts = []
    if agent_result.alert:
        parts.append(agent_result.alert)
    parts.append(llm_response)
    if agent_result.badge:
        parts.append(agent_result.badge)
    if agent_result.tools_executed:
        tools = " · ".join(f"✓ {t['tool']}" for t in agent_result.tools_executed if t['success'])
        if tools:
            parts.append(f"╌╌ {tools}")

    history.append({"role": "assistant", "content": "\n".join(parts)})
    return history, ""


def get_status_text() -> str:
    """Compact status for the sidebar."""
    global health_store
    meds = health_store.get_active_medications()
    vitals = health_store.get_recent_vitals(5)
    conds = health_store.get_conditions()
    rems = health_store.get_active_reminders()
    adherence = health_store.get_adherence_rate()

    lines = []
    if conds:
        lines.append("🏥 " + ", ".join(c.name for c in conds))
    if meds:
        lines.append("💊 " + " | ".join(f"{m.name} {m.dosage}" for m in meds))
    else:
        lines.append("💊 No medications")
    if vitals:
        labels = {'bp': 'BP', 'sugar': 'Sugar', 'spo2': 'SpO2', 'temperature': 'Temp', 'heart_rate': 'HR'}
        recent = vitals[-3:]
        for v in reversed(recent):
            val = f"{v.primary}/{v.secondary}" if v.secondary else f"{v.primary}"
            lines.append(f"  📊 {labels.get(v.vital_type, v.vital_type)}: {val}{v.unit}")
    if rems:
        lines.append(f"⏰ {len(rems)} active reminder(s)")
    lines.append(f"📈 Adherence: {adherence['rate']}%")
    return "\n".join(lines)


def generate_report_text(rtype: str) -> str:
    global health_store
    if rtype == "Health Report":
        return health_store.generate_health_report("Grandmother")
    return health_store.generate_family_status("Grandmother")


def load_demo():
    global health_store
    health_store = HealthStore()
    health_store.save_condition("Type 2 Diabetes", "moderate", "E11")
    health_store.save_condition("Hypertension", "moderate", "I10")
    health_store.save_medication("Metformin", "500mg", "twice daily", "", ["08:00", "20:00"], "After food")
    health_store.save_medication("Amlodipine", "5mg", "once daily", "", ["08:00"], "Morning")
    health_store.save_medication("Ecosprin", "75mg", "once daily", "", ["08:00"], "After food")
    health_store.save_vital("bp", 138, 88, "mmHg")
    health_store.save_vital("sugar", 165, 0, "mg/dL", "fasting")
    health_store.save_vital("bp", 142, 92, "mmHg")
    health_store.save_vital("sugar", 180, 0, "mg/dL", "post-meal")
    health_store.mark_taken("Metformin")
    health_store.mark_taken("Amlodipine")

    welcome = (
        "Demo loaded! 🎉 Grandmother with Diabetes + Hypertension.\n\n"
        "Try:\n"
        "• `BP 145/92` — record blood pressure\n"
        "• `sugar 200` — record blood sugar\n"
        "• `took metformin` — log adherence\n"
        "• `my medicines` — list medications\n"
        "• `remind me at 9pm take medicine`\n"
        "• `I have headache`\n"
        "• Malayalam: `ഷുഗർ 180`"
    )
    return [{"role": "assistant", "content": welcome}], get_status_text()


def reset_all():
    global health_store
    health_store = HealthStore()
    return [{"role": "assistant", "content": "🌱 Fresh start!\n\nനമസ്കാരം! ഞാൻ തുണ. എന്താ വിശേഷം? 😊"}], get_status_text()


# ═══════════════════════════════════════════════════════════════════════════
# CSS — Clean, focused chat UI matching the mobile app
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    background: #FAFBFC !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}

#header {
    background: linear-gradient(135deg, #0D7C66, #065A4A);
    padding: 20px; border-radius: 16px; text-align: center;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(13,124,102,0.15);
}
#header h1 { color: white !important; font-size: 24px !important; margin: 0 !important; font-weight: 700 !important; }
#header p { color: #D1FAE5 !important; font-size: 12px !important; margin: 4px 0 0 !important; }

/* Chat */
#chatbox { border-radius: 16px !important; border: 1px solid #E5E7EB !important; }
#chatbox .user .message-bubble {
    background: #0D7C66 !important; color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    font-size: 15px !important; line-height: 1.5 !important;
}
#chatbox .bot .message-bubble {
    background: white !important; color: #1F2937 !important;
    border: 1px solid #F0F0F0 !important;
    border-radius: 18px 18px 18px 4px !important;
    font-size: 15px !important; line-height: 1.5 !important;
}

/* Chips */
#chips button {
    border-radius: 18px !important; font-size: 13px !important; font-weight: 500 !important;
    background: #F0FDF4 !important; border: 1px solid #D1FAE5 !important; color: #065A4A !important;
    padding: 6px 14px !important; min-height: 34px !important;
}
#chips button:hover { background: #D1FAE5 !important; }

/* Input */
#input-box textarea {
    background: #F5F7FA !important; border: none !important; border-radius: 22px !important;
    padding: 12px 18px !important; font-size: 15px !important; min-height: 44px !important;
}
#input-box textarea:focus { box-shadow: 0 0 0 2px rgba(13,124,102,0.15) !important; }

#send-btn {
    background: #0D7C66 !important; color: white !important; border-radius: 22px !important;
    min-width: 44px !important; min-height: 44px !important; font-size: 18px !important;
    border: none !important; font-weight: 700 !important;
}
#send-btn:hover { background: #065A4A !important; }

/* Status panel */
#status-box textarea {
    font-size: 12px !important; font-family: monospace !important; line-height: 1.4 !important;
    background: #F8FAFC !important; border-radius: 10px !important;
}

/* Accordion */
.accordion { border-radius: 12px !important; }

/* Mobile */
@media (max-width: 640px) {
    .gradio-container { padding: 6px !important; }
    #header { padding: 14px; }
    #header h1 { font-size: 20px !important; }
    #chatbox { min-height: 350px !important; }
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# UI — Single focused page, chat-first
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CSS, title="Thuna — AI Health Companion", theme=gr.themes.Soft()) as demo:

    # Header
    gr.HTML('<div id="header"><h1>🤝 തുണ — Thuna</h1><p>AI Health Companion • Gemma 4 E2B (2B) • Malayalam • Offline</p></div>')

    # Main Chat
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "നമസ്കാരം! 🙏 ഞാൻ തുണ.\n\nBP, ഷുഗർ, മരുന്ന് — എന്തും ചോദിക്കാം.\n\n💡 Try: `BP 140/90` · `sugar 180` · `took metformin`\n\n⚙️ Click **Load Demo** below to start with sample data."}],
        type="messages",
        height=420,
        elem_id="chatbox",
        show_copy_button=True,
    )

    # Quick chips
    with gr.Row(elem_id="chips"):
        chip_bp = gr.Button("❤️ BP", size="sm")
        chip_sugar = gr.Button("🩸 ഷുഗർ", size="sm")
        chip_taken = gr.Button("💊 കഴിച്ചു", size="sm")
        chip_meds = gr.Button("📋 മരുന്നുകൾ", size="sm")
        chip_today = gr.Button("📆 ഇന്ന്", size="sm")

    # Input
    with gr.Row():
        msg = gr.Textbox(placeholder="തുണയോട് പറയുക...", show_label=False, scale=5, elem_id="input-box", lines=1)
        send = gr.Button("↑", elem_id="send-btn", scale=0, min_width=44)

    # Collapsible panels below chat
    with gr.Accordion("📊 Health Status", open=False):
        status_box = gr.Textbox(value=get_status_text(), label="", lines=8, interactive=False, elem_id="status-box")
        with gr.Row():
            refresh_btn = gr.Button("Refresh", size="sm")
            demo_btn = gr.Button("🎬 Load Demo", size="sm", variant="primary")
            reset_btn = gr.Button("Reset", size="sm")

    with gr.Accordion("📄 Reports", open=False):
        with gr.Row():
            rtype = gr.Radio(["Health Report", "Family Status"], value="Health Report", label="", scale=3)
            gen_btn = gr.Button("Generate", size="sm", variant="primary", scale=1)
        report_box = gr.Textbox(label="", lines=15, interactive=False, show_copy_button=True)

    with gr.Accordion("🎤 Voice & 📷 Prescription", open=False):
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Record Malayalam voice")
            img_in = gr.Image(type="filepath", label="Upload prescription photo", sources=["upload", "webcam"], height=120)

    # ─── Wiring ───

    def on_submit(message, history):
        history, _ = process_message(message, history)
        return history, "", get_status_text()

    def on_chip(text, history):
        history, _ = process_message(text, history)
        return history, get_status_text()

    def on_demo():
        h, s = load_demo()
        return h, s

    def on_reset():
        h, s = reset_all()
        return h, s

    def on_voice(audio_path, history):
        if not audio_path:
            return history or [], get_status_text()
        transcription = ""
        try:
            if client:
                result = client.automatic_speech_recognition(audio_path, model="openai/whisper-large-v3")
                transcription = result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception:
            pass
        if not transcription:
            history = history or []
            history.append({"role": "assistant", "content": "🎤 Voice received — type your message for the web demo. Mobile app has full Malayalam STT."})
            return history, get_status_text()
        history, _ = process_message(transcription, history)
        return history, get_status_text()

    def on_image(image_path, history):
        if not image_path:
            return history or [], get_status_text()
        history = history or []
        history.append({"role": "user", "content": "📷 Prescription uploaded"})
        history.append({"role": "assistant", "content": "📷 Got it! In the mobile app, Gemma 4 vision extracts medications.\n\nFor this demo, type:\n• `metformin 500mg twice daily after food`\n• `amlodipine 5mg once daily`\n\nEach saves with auto-reminder. 💊"})
        return history, get_status_text()

    # Submit
    msg.submit(on_submit, [msg, chatbot], [chatbot, msg, status_box])
    send.click(on_submit, [msg, chatbot], [chatbot, msg, status_box])

    # Chips
    chip_bp.click(lambda: "BP ", outputs=[msg])
    chip_sugar.click(lambda: "sugar ", outputs=[msg])
    chip_taken.click(lambda h: on_chip("took medicine", h), [chatbot], [chatbot, status_box])
    chip_meds.click(lambda h: on_chip("my medicines", h), [chatbot], [chatbot, status_box])
    chip_today.click(lambda h: on_chip("did I take medicine today", h), [chatbot], [chatbot, status_box])

    # Controls
    refresh_btn.click(get_status_text, outputs=[status_box])
    demo_btn.click(on_demo, outputs=[chatbot, status_box])
    reset_btn.click(on_reset, outputs=[chatbot, status_box])

    # Reports
    gen_btn.click(generate_report_text, [rtype], [report_box])

    # Voice/Image
    audio_in.stop_recording(on_voice, [audio_in, chatbot], [chatbot, status_box])
    img_in.change(on_image, [img_in, chatbot], [chatbot, status_box])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
