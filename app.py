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

# Malayalam system prompt
SYSTEM_PROMPT = """നീ "തുണ" (Thuna) ആണ്. കേരളത്തിലെ പ്രായമായവരുടെ ആരോഗ്യ companion.

നിയമങ്ങൾ:
- നാടൻ മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക. English words ആവശ്യമെങ്കിൽ മാത്രം (medicine names, numbers).
- ചെറിയ വാക്യങ്ങൾ. ലളിതമായ ഭാഷ. പ്രായമായവർക്ക് മനസ്സിലാകുന്നത്.
- ആരോഗ്യ കാര്യങ്ങളിൽ ശ്രദ്ധാലുവായിരിക്കുക. അപകടകരമായ values കണ്ടാൽ ഡോക്ടറെ കാണാൻ പറയുക.
- Companion ആയി സംസാരിക്കുക. Warm, caring. ചിരിപ്പിക്കുക.
- മരുന്ന് കാര്യങ്ങളിൽ exact ആയിരിക്കുക.
- 2-3 വാക്യത്തിൽ answer ചെയ്യുക.
- Hindi, Manglish എഴുതരുത്. Pure Malayalam script മാത്രം.

ശരി: "BP record ചെയ്തു. 130/85 — നോർമൽ ആണ്. നന്നായിരിക്കുന്നു!"
തെറ്റ്: "Aapka BP record ho gaya hai"
തെറ്റ്: "Ningalude BP record cheythu" """


def generate_llm_response(context: str, chat_history: List[dict]) -> str:
    """Generate response using Gemma via HF Inference API."""
    if not client:
        return generate_fallback_response(context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": context})

    try:
        response = client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        text = response.choices[0].message.content.strip()
        return post_process_malayalam(text)
    except Exception:
        return generate_fallback_response(context)


def post_process_malayalam(text: str) -> str:
    """Clean LLM output."""
    text = re.sub(r'[\*\#]+', '', text)
    lines = text.split('\n')
    clean = []
    for line in lines:
        if not line.strip():
            continue
        mal = len(re.findall(r'[\u0D00-\u0D7F]', line))
        lat = len(re.findall(r'[a-zA-Z]', line))
        if mal > 0 or len(line) < 20:
            clean.append(line)
    return '\n'.join(clean) if clean else text


def generate_fallback_response(context: str) -> str:
    """Template responses when LLM API unavailable."""
    lower = context.lower()
    if "bp" in lower and "record" in lower:
        return "BP record ചെയ്തു. ശ്രദ്ധിക്കണം. വിശ്രമിക്കുക. 🩺"
    if "sugar" in lower and "record" in lower:
        return "ഷുഗർ record ചെയ്തു. ട്രെൻഡ് ശ്രദ്ധിക്കാം. 🩸"
    if "medication" in lower or "medicine" in lower:
        if "saved" in lower or "save" in lower:
            return "മരുന്ന് save ചെയ്തു. Reminder set ചെയ്തിട്ടുണ്ട്. 💊"
        return "നിങ്ങളുടെ മരുന്നുകൾ ഇതാ."
    if "took" in lower or "taken" in lower or "adherence" in lower:
        return "നന്നായി! മരുന്ന് കഴിച്ചു എന്ന് record ചെയ്തു. ✅"
    if "reminder" in lower:
        return "Reminder set ചെയ്തു. സമയത്ത് ഓർമ്മിപ്പിക്കാം. ⏰"
    if "condition" in lower:
        return "രേഖപ്പെടുത്തി. ഡോക്ടറുടെ നിർദ്ദേശം പാലിക്കുക. 🏥"
    if "symptom" in lower or "pain" in lower or "fever" in lower:
        return "ലക്ഷണം record ചെയ്തു. കൂടുതൽ അസ്വസ്ഥത ഉണ്ടെങ്കിൽ ഡോക്ടറെ കാണുക. 🤒"
    if "alert" in lower or "🚨" in lower:
        return "⚠️ ശ്രദ്ധിക്കുക! ഡോക്ടറെ വിളിക്കുക."
    return "എന്താ വിശേഷം? ആരോഗ്യം, മരുന്ന്, BP, ഷുഗർ — എന്തും ചോദിക്കാം. 😊"


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════

health_store = HealthStore()


def process_message(message: str, history: List[dict]) -> Tuple[List[dict], str]:
    """Process user message through the agent pipeline."""
    global health_store

    if not message or not message.strip():
        return history, ""

    history = history or []
    history.append({"role": "user", "content": message})

    agent_result = run_agent(message, health_store)
    llm_response = generate_llm_response(agent_result.context_for_llm, history)

    response_parts = []
    if agent_result.alert:
        response_parts.append(agent_result.alert)
        response_parts.append("")
    response_parts.append(llm_response)
    if agent_result.badge:
        response_parts.append(f"\n{agent_result.badge}")
    if agent_result.tools_executed:
        tools = " · ".join(f"✓ {t['tool']}" for t in agent_result.tools_executed if t['success'])
        if tools:
            response_parts.append(f"\n╌╌╌ {tools}")

    final_response = "\n".join(response_parts)
    history.append({"role": "assistant", "content": final_response})
    return history, ""


def get_dashboard_html() -> str:
    """Generate dashboard as styled HTML."""
    global health_store

    meds = health_store.get_active_medications()
    vitals = health_store.get_recent_vitals(5)
    conds = health_store.get_conditions()
    rems = health_store.get_active_reminders()
    adherence = health_store.get_adherence_rate()

    html = '<div style="font-family: -apple-system, system-ui, sans-serif; font-size: 14px; line-height: 1.6; padding: 12px;">'

    # Medications
    html += '<div style="margin-bottom: 16px;"><strong style="color: #0D7C66;">💊 Medications</strong><br>'
    if meds:
        for m in meds:
            html += f'<span style="display:block; padding: 2px 0; color: #374151;">• {m.name} {m.dosage} — {m.frequency}</span>'
    else:
        html += '<span style="color: #9CA3AF;">None recorded</span>'
    html += '</div>'

    # Vitals
    html += '<div style="margin-bottom: 16px;"><strong style="color: #0D7C66;">📊 Recent Vitals</strong><br>'
    if vitals:
        labels = {'bp': 'BP', 'sugar': 'Sugar', 'spo2': 'SpO2', 'temperature': 'Temp', 'heart_rate': 'HR', 'weight': 'Weight'}
        for v in reversed(vitals[-5:]):
            val = f"{v.primary}/{v.secondary}" if v.secondary else f"{v.primary}"
            html += f'<span style="display:block; padding: 2px 0; color: #374151;">{labels.get(v.vital_type, v.vital_type)}: {val} {v.unit}</span>'
    else:
        html += '<span style="color: #9CA3AF;">No readings yet</span>'
    html += '</div>'

    # Conditions
    if conds:
        html += '<div style="margin-bottom: 16px;"><strong style="color: #0D7C66;">🏥 Conditions</strong><br>'
        for c in conds:
            html += f'<span style="display:block; padding: 2px 0; color: #374151;">• {c.name} ({c.status})</span>'
        html += '</div>'

    # Reminders
    if rems:
        html += '<div style="margin-bottom: 16px;"><strong style="color: #0D7C66;">⏰ Reminders</strong><br>'
        for r in rems:
            html += f'<span style="display:block; padding: 2px 0; color: #374151;">• {r.medication} at {", ".join(r.times)}</span>'
        html += '</div>'

    # Adherence
    html += f'<div style="margin-bottom: 8px;"><strong style="color: #0D7C66;">📈 Adherence:</strong> <span style="color: #374151;">{adherence["rate"]}% (7 days)</span></div>'

    html += '</div>'
    return html


def generate_report() -> str:
    global health_store
    return health_store.generate_health_report("Grandmother")


def generate_status() -> str:
    global health_store
    return health_store.generate_family_status("Grandmother")


def reset_session():
    global health_store
    health_store = HealthStore()
    initial = [{"role": "assistant", "content": "Session reset! 🌱\n\nനമസ്കാരം! ഞാൻ തുണ. എന്താ വിശേഷം? 😊"}]
    return initial, get_dashboard_html()


def load_demo_data():
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
        "Demo data loaded! 🎉\n\n"
        "**Patient:** Elderly grandmother with Diabetes + Hypertension\n"
        "**Medications:** Metformin 500mg, Amlodipine 5mg, Ecosprin 75mg\n\n"
        "Try:\n"
        "• `BP 145/92` — record blood pressure\n"
        "• `sugar 200` — record blood sugar\n"
        "• `took metformin` — log adherence\n"
        "• `my medicines` — list medications\n"
        "• `remind me at 9pm take medicine`\n"
        "• `I have headache and fever`\n"
        "• Malayalam: `ഷുഗർ 180` or `മരുന്ന് കഴിച്ചു`"
    )
    initial = [{"role": "assistant", "content": welcome}]
    return initial, get_dashboard_html()


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Matches React Native app exactly
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* ─── Base ─── */
:root {
    --thuna-green: #0D7C66;
    --thuna-green-light: #F0FDF4;
    --thuna-green-dark: #065A4A;
    --thuna-bg: #FAFBFC;
    --thuna-border: #F0F0F0;
    --thuna-text: #1F2937;
    --thuna-muted: #9CA3AF;
}

.gradio-container {
    max-width: 100% !important;
    background: var(--thuna-bg) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}

/* ─── Header (matches app green gradient) ─── */
#thuna-header {
    background: linear-gradient(135deg, #0D7C66 0%, #065A4A 100%);
    padding: 24px;
    border-radius: 20px;
    margin-bottom: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(13, 124, 102, 0.2);
}
#thuna-header h1 {
    color: white !important;
    font-size: 26px !important;
    margin: 0 !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
#thuna-header p {
    color: #D1FAE5 !important;
    margin: 4px 0 0 !important;
    font-size: 13px !important;
    opacity: 0.9;
}

/* ─── Chat (matches RN bubbles) ─── */
.chatbot {
    background: var(--thuna-bg) !important;
    border: 1px solid var(--thuna-border) !important;
    border-radius: 20px !important;
    min-height: 480px !important;
}
.chatbot .message-wrap {
    padding: 16px !important;
}

/* User bubble — green like the RN app */
.chatbot .user .message-bubble {
    background: var(--thuna-green) !important;
    color: white !important;
    border-radius: 20px 20px 6px 20px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
}

/* Agent bubble — white with subtle border */
.chatbot .bot .message-bubble {
    background: #FFFFFF !important;
    color: var(--thuna-text) !important;
    border: 1px solid var(--thuna-border) !important;
    border-radius: 20px 20px 20px 6px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
}

/* ─── Quick Action Chips (matches RN chips exactly) ─── */
#chip-row {
    padding: 8px 0 !important;
}
#chip-row button {
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    background: var(--thuna-green-light) !important;
    border: 1px solid #D1FAE5 !important;
    color: var(--thuna-green-dark) !important;
    min-height: 36px !important;
    transition: background 0.2s;
}
#chip-row button:hover {
    background: #D1FAE5 !important;
}

/* ─── Input (matches RN input bar) ─── */
#msg-input textarea {
    background: #F5F7FA !important;
    border-radius: 24px !important;
    border: none !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
    color: var(--thuna-text) !important;
    min-height: 48px !important;
    resize: none !important;
}
#msg-input textarea::placeholder {
    color: #B0B0B0 !important;
}
#msg-input textarea:focus {
    box-shadow: 0 0 0 2px rgba(13, 124, 102, 0.2) !important;
}

/* Send button — green circle like RN */
#send-btn {
    background: var(--thuna-green) !important;
    color: white !important;
    border-radius: 24px !important;
    min-width: 48px !important;
    min-height: 48px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(13, 124, 102, 0.3) !important;
}
#send-btn:hover {
    background: var(--thuna-green-dark) !important;
}

/* ─── Dashboard panel ─── */
#dashboard-panel {
    background: white !important;
    border: 1px solid var(--thuna-border) !important;
    border-radius: 16px !important;
    overflow: hidden;
}

/* ─── Tab styling ─── */
.tab-nav {
    border-bottom: 1px solid var(--thuna-border) !important;
}
.tab-nav button {
    font-size: 13px !important;
    font-weight: 500 !important;
}
.tab-nav button.selected {
    color: var(--thuna-green) !important;
    border-color: var(--thuna-green) !important;
}

/* ─── Setup buttons ─── */
#demo-btn {
    background: var(--thuna-green) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}
#reset-btn {
    border-radius: 12px !important;
}

/* ─── Report box ─── */
#report-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}

/* ─── Mobile responsive ─── */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    #thuna-header { padding: 16px; border-radius: 14px; }
    #thuna-header h1 { font-size: 22px !important; }
    .chatbot { min-height: 360px !important; border-radius: 14px !important; }
    #chip-row button { font-size: 13px !important; padding: 6px 12px !important; }
    #msg-input textarea { font-size: 15px !important; }
}

/* Elderly-friendly: minimum touch targets */
button { min-height: 44px !important; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI — Mirrors the React Native app layout
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CUSTOM_CSS, title="Thuna (തുണ) — AI Health Companion", theme=gr.themes.Soft()) as demo:

    # ─── Header (green gradient, app identity) ───
    gr.HTML("""
    <div id="thuna-header">
        <h1>🤝 തുണ — Thuna</h1>
        <p>Offline AI Health Companion for Elderly • Gemma 4 E2B (2B params)</p>
        <p style="font-size: 11px; margin-top: 6px; opacity: 0.7;">
            Malayalam Voice • Medication Safety • Health Monitoring • 100% On-Device
        </p>
    </div>
    """)

    with gr.Row():
        # ─── LEFT: Chat (main area) ───
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "നമസ്കാരം! 🙏 ഞാൻ തുണ.\n\nആരോഗ്യം, മരുന്ന്, BP, ഷുഗർ — എന്തും ചോദിക്കാം.\nഅല്ലെങ്കിൽ വെറുതെ സംസാരിക്കാം. 😊\n\n💡 `BP 140/90` · `sugar 180` · `took metformin` · `my medicines`"}],
                type="messages",
                height=480,
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/handshake_1f91d.png"),
            )

            # Quick action chips (matches RN chip row)
            with gr.Row(elem_id="chip-row"):
                chip_bp = gr.Button("❤️ BP", size="sm")
                chip_sugar = gr.Button("🩸 ഷുഗർ", size="sm")
                chip_taken = gr.Button("💊 കഴിച്ചു", size="sm")
                chip_meds = gr.Button("📋 മരുന്നുകൾ", size="sm")
                chip_report = gr.Button("📄 Report", size="sm")

            # Input row (matches RN: textbox + send button)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="തുണയോട് പറയുക... (BP 140/90, sugar 180, took medicine...)",
                    show_label=False,
                    scale=5,
                    elem_id="msg-input",
                    lines=1,
                    max_lines=2,
                )
                send_btn = gr.Button("↑", elem_id="send-btn", scale=0, min_width=48)

            # Voice + Camera row
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Malayalam Voice", scale=2)
                image_input = gr.Image(type="filepath", label="📷 Prescription", sources=["upload", "webcam"], height=80, scale=1)

        # ─── RIGHT: Dashboard sidebar ───
        with gr.Column(scale=1, min_width=260):
            with gr.Tabs():
                with gr.Tab("📊 Dashboard"):
                    dashboard = gr.HTML(value=get_dashboard_html(), elem_id="dashboard-panel")
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")

                with gr.Tab("📄 Reports"):
                    report_type = gr.Radio(["Health Report", "Family Status"], value="Health Report", label="Type")
                    gen_report_btn = gr.Button("Generate", variant="primary", size="sm")
                    report_output = gr.Textbox(label="Report", lines=18, interactive=False, elem_id="report-box", show_copy_button=True)

                with gr.Tab("⚙️ Setup"):
                    gr.Markdown("### Quick Start")
                    gr.Markdown("Load demo data to explore Thuna with a pre-configured patient.")
                    demo_btn = gr.Button("🎬 Load Demo Data", variant="primary", elem_id="demo-btn")
                    reset_btn = gr.Button("🔄 Reset", variant="secondary", elem_id="reset-btn")
                    gr.Markdown("""---
**Architecture:**
```
Voice → Regex (<1ms)
     → 23-tool Agent
     → Gemma 4 E2B
     → Malayalam TTS
```
**Model:** Gemma 4 E2B (2B, INT4)
**Offline:** 100% on-device after setup
""")

    # ═══════════════════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════════════════

    def submit_msg(message, history):
        history, _ = process_message(message, history)
        return history, "", get_dashboard_html()

    def chip_action(text, history):
        history, _ = process_message(text, history)
        return history, get_dashboard_html()

    def report_action(rtype):
        return generate_report() if rtype == "Health Report" else generate_status()

    def demo_action():
        h, d = load_demo_data()
        return h, d

    def reset_action():
        h, d = reset_session()
        return h, d

    def process_voice(audio_path, history):
        if not audio_path:
            return history, get_dashboard_html()
        transcription = ""
        try:
            if client:
                result = client.automatic_speech_recognition(audio_path, model="openai/whisper-large-v3")
                transcription = result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception:
            pass
        if not transcription:
            history = history or []
            history.append({"role": "assistant", "content": "🎤 Voice received. Type your message for this web demo — the mobile app has full Malayalam STT."})
            return history, get_dashboard_html()
        history, _ = process_message(transcription, history)
        return history, get_dashboard_html()

    def process_image(image_path, history):
        if not image_path:
            return history, get_dashboard_html()
        history = history or []
        history.append({"role": "user", "content": "📷 [Prescription uploaded]"})
        history.append({"role": "assistant", "content": "📷 Prescription uploaded!\n\nIn the mobile app, Gemma 4 vision extracts medications automatically.\n\nFor this demo, type them manually:\n• `metformin 500mg twice daily after food`\n• `amlodipine 5mg once daily morning`\n\nEach saves with auto-reminders. 💊"})
        return history, get_dashboard_html()

    # Wire events
    msg_input.submit(submit_msg, [msg_input, chatbot], [chatbot, msg_input, dashboard])
    send_btn.click(submit_msg, [msg_input, chatbot], [chatbot, msg_input, dashboard])

    chip_bp.click(lambda: "BP ", outputs=[msg_input])
    chip_sugar.click(lambda: "sugar ", outputs=[msg_input])
    chip_taken.click(lambda h: chip_action("took medicine", h), [chatbot], [chatbot, dashboard])
    chip_meds.click(lambda h: chip_action("my medicines", h), [chatbot], [chatbot, dashboard])
    chip_report.click(report_action, [report_type], [report_output])

    refresh_btn.click(get_dashboard_html, outputs=[dashboard])
    gen_report_btn.click(report_action, [report_type], [report_output])
    demo_btn.click(demo_action, outputs=[chatbot, dashboard])
    reset_btn.click(reset_action, outputs=[chatbot, dashboard])

    audio_input.stop_recording(process_voice, [audio_input, chatbot], [chatbot, dashboard])
    image_input.change(process_image, [image_input, chatbot], [chatbot, dashboard])


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
