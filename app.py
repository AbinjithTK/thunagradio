"""
Thuna (തുണ) — Offline AI Health Companion for Elderly
Gradio App for Build Small Hackathon (Hugging Face × Gradio)

Voice-first Malayalam health companion powered by Gemma 4 E2B (2B params).
"""

import gradio as gr
import os
import re
from typing import List, Tuple
from datetime import datetime

from intent_parser import parse_intent
from health_store import HealthStore
from agent_engine import run_agent, AgentResult

# ═══════════════════════════════════════════════════════════════════════════
# LLM
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

SYSTEM_PROMPT = """You are "തുണ" (Thuna), a health companion for elderly people in Kerala, India.

CRITICAL RULE: You MUST reply ONLY in Malayalam (മലയാളം) script. 
DO NOT use Tamil (தமிழ்). DO NOT use Hindi. DO NOT use English sentences.
Use English only for medicine names and numbers.

Keep answers to 2-3 short sentences. Be warm and caring.

Example correct response: "BP record ചെയ്തു. 140/90 — ചെറിയ ഉയർച്ച ഉണ്ട്. വിശ്രമിക്കുക."
Example WRONG (Tamil): "மாத்திரை எடுத்துக்கட்டா" ← NEVER do this
Example WRONG (Hindi): "Aapka BP theek hai" ← NEVER do this"""


def generate_llm_response(context: str, chat_history: List[dict]) -> str:
    if not client:
        return generate_fallback_response(context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append(msg)
    messages.append({"role": "user", "content": context})
    try:
        response = client.chat_completion(
            model=MODEL_ID, messages=messages, max_tokens=150, temperature=0.7,
        )
        text = re.sub(r'[\*\#]+', '', response.choices[0].message.content.strip())
        # Reject if Tamil (U+0B80-0BFF) or Hindi (U+0900-097F) detected
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        if tamil_chars > 3 or hindi_chars > 3:
            return generate_fallback_response(context)
        return text if text else generate_fallback_response(context)
    except Exception:
        return generate_fallback_response(context)


def generate_fallback_response(context: str) -> str:
    lower = context.lower()
    if "bp" in lower and ("record" in lower or "vital" in lower):
        return "BP record ചെയ്തു. ശ്രദ്ധിക്കണം. 🩺"
    if "sugar" in lower and ("record" in lower or "vital" in lower):
        return "ഷുഗർ record ചെയ്തു. ട്രെൻഡ് ശ്രദ്ധിക്കാം. 🩸"
    if "saved" in lower and "medication" in lower:
        return "മരുന്ന് save ചെയ്തു. Reminder set ചെയ്തു. 💊"
    if "medication" in lower or "medicine" in lower:
        return "നിങ്ങളുടെ മരുന്നുകൾ ഇതാ."
    if "took" in lower or "taken" in lower or "confirm" in lower:
        return "നന്നായി! മരുന്ന് കഴിച്ചു. ✅"
    if "reminder" in lower:
        return "Reminder set ചെയ്തു. ⏰"
    if "condition" in lower:
        return "രേഖപ്പെടുത്തി. 🏥"
    if "symptom" in lower or "report" in lower:
        return "ലക്ഷണം record ചെയ്തു. ഡോക്ടറെ കാണുക. 🤒"
    if "alert" in lower or "🚨" in lower:
        return "⚠️ ശ്രദ്ധിക്കുക! ഡോക്ടറെ വിളിക്കുക."
    return "എന്താ വിശേഷം? എന്തും ചോദിക്കാം. 😊"


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
        tools = " · ".join(f"✓{t['tool']}" for t in agent_result.tools_executed if t['success'])
        if tools:
            parts.append(f"⎯ {tools}")

    history.append({"role": "assistant", "content": "\n".join(parts)})
    return history, ""


def get_status_text() -> str:
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
        for v in reversed(vitals[-3:]):
            val = f"{v.primary}/{v.secondary}" if v.secondary else f"{v.primary}"
            lines.append(f"  📊 {labels.get(v.vital_type, v.vital_type)}: {val}{v.unit}")
    if rems:
        lines.append(f"⏰ {len(rems)} reminder(s)")
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
        "Demo loaded! 🎉\n\n"
        "Grandmother: Diabetes + Hypertension\n"
        "Meds: Metformin, Amlodipine, Ecosprin\n\n"
        "Try: BP 145/92 · sugar 200 · took metformin · my medicines"
    )
    return [{"role": "assistant", "content": welcome}], get_status_text()


def reset_all():
    global health_store
    health_store = HealthStore()
    return [{"role": "assistant", "content": "🌱 Fresh start! നമസ്കാരം, ഞാൻ തുണ. 😊"}], get_status_text()


# ═══════════════════════════════════════════════════════════════════════════
# CSS — Light theme, mobile-first, matching the Android app
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
/* Force light theme everywhere */
.gradio-container, .dark, [data-testid] {
    --background-fill-primary: #FAFBFC !important;
    --background-fill-secondary: #FFFFFF !important;
    --border-color-primary: #E8ECF0 !important;
    --body-background-fill: #FAFBFC !important;
    --block-background-fill: #FFFFFF !important;
    --input-background-fill: #F5F7FA !important;
    --body-text-color: #1F2937 !important;
    --block-label-text-color: #6B7280 !important;
    --color-accent: #0D7C66 !important;
}
.gradio-container {
    max-width: 520px !important;
    margin: 0 auto !important;
    background: #FAFBFC !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
    padding: 12px !important;
}

/* Header */
#header {
    background: linear-gradient(135deg, #0D7C66, #065A4A);
    padding: 18px 16px; border-radius: 16px; text-align: center;
    margin-bottom: 12px;
}
#header h1 { color: white !important; font-size: 22px !important; margin: 0 !important; }
#header p { color: #D1FAE5 !important; font-size: 11px !important; margin: 4px 0 0 !important; }

/* Chat — light background */
#chatbox {
    border-radius: 16px !important;
    border: 1px solid #E8ECF0 !important;
    background: #FFFFFF !important;
}
#chatbox .wrapper, #chatbox .message-wrap, #chatbox .messages { background: #FFFFFF !important; }

/* User bubble — GREEN */
#chatbox .user .message-bubble,
#chatbox .message-row.user .content,
.chatbot .user .message-bubble {
    background: #0D7C66 !important; color: white !important;
    border-radius: 16px 16px 4px 16px !important;
    font-size: 15px !important; line-height: 1.5 !important;
    padding: 10px 14px !important;
}
#chatbox .user .message-bubble *, .chatbot .user .message-bubble * { color: white !important; }

/* Bot bubble — LIGHT GRAY */
#chatbox .bot .message-bubble,
#chatbox .message-row.bot .content,
.chatbot .bot .message-bubble {
    background: #F3F4F6 !important; color: #1F2937 !important;
    border: 1px solid #E8ECF0 !important;
    border-radius: 16px 16px 16px 4px !important;
    font-size: 15px !important; line-height: 1.5 !important;
    padding: 10px 14px !important;
}
#chatbox .bot .message-bubble *, .chatbot .bot .message-bubble * { color: #1F2937 !important; }

/* Code blocks — visible on both bubbles */
#chatbox code, .chatbot code {
    background: rgba(0,0,0,0.08) !important; color: inherit !important;
    border-radius: 4px !important; padding: 1px 5px !important; font-size: 13px !important;
}
#chatbox .user code, .chatbot .user code {
    background: rgba(255,255,255,0.25) !important; color: white !important;
}

/* Chips row */
#chips { margin: 8px 0 !important; gap: 6px !important; }
#chips button {
    border-radius: 16px !important; font-size: 13px !important; font-weight: 500 !important;
    background: #F0FDF4 !important; border: 1px solid #D1FAE5 !important; color: #065A4A !important;
    padding: 6px 12px !important; min-height: 32px !important;
}
#chips button:hover { background: #D1FAE5 !important; }

/* Input bar area */
#input-box textarea {
    background: #F5F7FA !important; border: 1px solid #E8ECF0 !important;
    border-radius: 22px !important;
    padding: 10px 14px !important; font-size: 15px !important;
    min-height: 40px !important; resize: none !important;
    color: #1F2937 !important;
}
#input-box textarea::placeholder { color: #9CA3AF !important; }
#input-box textarea:focus { 
    border-color: #0D7C66 !important;
    box-shadow: 0 0 0 2px rgba(13,124,102,0.1) !important; 
}

#send-btn {
    background: #0D7C66 !important; color: white !important; border-radius: 20px !important;
    min-width: 40px !important; max-width: 40px !important;
    min-height: 40px !important; max-height: 40px !important;
    font-size: 16px !important; border: none !important;
}
#send-btn:hover { background: #065A4A !important; }

#mic-btn {
    background: transparent !important; color: #0D7C66 !important;
    border: 1px solid #D1FAE5 !important; border-radius: 20px !important;
    min-width: 40px !important; max-width: 40px !important;
    min-height: 40px !important; max-height: 40px !important;
    font-size: 18px !important;
}
#mic-btn:hover { background: #F0FDF4 !important; }

/* Accordion — light */
.accordion {
    border: 1px solid #E8ECF0 !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
    margin-top: 8px !important;
}
.accordion > .label-wrap {
    background: #FFFFFF !important;
    color: #374151 !important;
    padding: 10px 14px !important;
}
.accordion > .label-wrap span { color: #374151 !important; font-size: 13px !important; }

/* Status box */
#status-box textarea {
    font-size: 12px !important; font-family: monospace !important;
    background: #F8FAFB !important; border: 1px solid #E8ECF0 !important;
    border-radius: 8px !important; color: #374151 !important;
}

/* Buttons inside accordions */
.accordion button {
    border-radius: 10px !important; font-size: 13px !important;
}

/* Report */
#report-box textarea {
    font-family: monospace !important; font-size: 11px !important;
    background: #F8FAFB !important; color: #374151 !important;
}

/* Hide dark mode artifacts */
footer { display: none !important; }
.built-with { display: none !important; }

/* Audio component styling */
#voice-section audio { border-radius: 8px !important; }
#voice-section .wrap { background: #F8FAFB !important; border: 1px solid #E8ECF0 !important; border-radius: 10px !important; }

/* Image upload */
#voice-section .image-container { border-radius: 10px !important; border: 1px solid #E8ECF0 !important; }
"""

# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CSS, title="Thuna — AI Health Companion", theme=gr.themes.Default()) as demo:

    # Header
    gr.HTML('<div id="header"><h1>🤝 തുണ — Thuna</h1><p>AI Health Companion • Gemma 4 E2B (2B) • Malayalam • Offline</p></div>')

    # Chat
    chatbot = gr.Chatbot(
        value=[{"role": "assistant", "content": "നമസ്കാരം! 🙏 ഞാൻ തുണ.\n\nBP, ഷുഗർ, മരുന്ന് — എന്തും ചോദിക്കാം.\n\n💡 Try: BP 140/90 · sugar 180 · took metformin\n\n⚙️ Tap Load Demo in Health Status below."}],
        type="messages", height=380, elem_id="chatbox", show_copy_button=True,
    )

    # Quick chips
    with gr.Row(elem_id="chips"):
        chip_bp = gr.Button("❤️ BP", size="sm")
        chip_sugar = gr.Button("🩸 ഷുഗർ", size="sm")
        chip_taken = gr.Button("💊 കഴിച്ചു", size="sm")
        chip_meds = gr.Button("📋 മരുന്നുകൾ", size="sm")
        chip_today = gr.Button("📆 ഇന്ന്", size="sm")

    # Input bar: text + mic + send
    with gr.Row():
        msg = gr.Textbox(
            placeholder="തുണയോട് പറയുക...",
            show_label=False, scale=5, elem_id="input-box",
            lines=1, max_lines=2,
        )
        mic_btn = gr.Button("🎤", elem_id="mic-btn", scale=0, min_width=40)
        send = gr.Button("↑", elem_id="send-btn", scale=0, min_width=40)

    # Hidden audio for voice recording (shown when mic is clicked)
    with gr.Accordion("🎤 Voice Recording", open=False, visible=True) as voice_acc:
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="Tap record, speak in Malayalam, then stop")

    # Health Status
    with gr.Accordion("📊 Health Status", open=False):
        status_box = gr.Textbox(value=get_status_text(), label="", lines=7, interactive=False, elem_id="status-box")
        with gr.Row():
            demo_btn = gr.Button("🎬 Load Demo", size="sm", variant="primary")
            refresh_btn = gr.Button("🔄", size="sm")
            reset_btn = gr.Button("Reset", size="sm")

    # Reports
    with gr.Accordion("📄 Reports", open=False):
        with gr.Row():
            rtype = gr.Radio(["Health Report", "Family Status"], value="Health Report", label="", scale=3)
            gen_btn = gr.Button("Generate", size="sm", variant="primary", scale=1)
        report_box = gr.Textbox(label="", lines=12, interactive=False, elem_id="report-box", show_copy_button=True)

    # Prescription
    with gr.Accordion("📷 Scan Prescription", open=False):
        img_in = gr.Image(type="filepath", label="Upload or take photo", sources=["upload", "webcam"], height=150)

    # ═══════════════════════════════════════════════════════════════════════
    # EVENTS
    # ═══════════════════════════════════════════════════════════════════════

    def on_submit(message, history):
        history, _ = process_message(message, history)
        return history, "", get_status_text()

    def on_chip(text, history):
        history, _ = process_message(text, history)
        return history, get_status_text()

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
            history.append({"role": "assistant", "content": "🎤 Voice received — for this web demo, type your message. Mobile app has full Malayalam STT."})
            return history, get_status_text()
        history, _ = process_message(transcription, history)
        return history, get_status_text()

    def on_image(image_path, history):
        if not image_path:
            return history or [], get_status_text()
        history = history or []
        history.append({"role": "user", "content": "📷 Prescription"})
        history.append({"role": "assistant", "content": "📷 Got it! Type medications:\n• `metformin 500mg twice daily after food`\n• `amlodipine 5mg once daily`\n\nEach saves with auto-reminder. 💊"})
        return history, get_status_text()

    # Wire
    msg.submit(on_submit, [msg, chatbot], [chatbot, msg, status_box])
    send.click(on_submit, [msg, chatbot], [chatbot, msg, status_box])

    chip_bp.click(lambda: "BP ", outputs=[msg])
    chip_sugar.click(lambda: "sugar ", outputs=[msg])
    chip_taken.click(lambda h: on_chip("took medicine", h), [chatbot], [chatbot, status_box])
    chip_meds.click(lambda h: on_chip("my medicines", h), [chatbot], [chatbot, status_box])
    chip_today.click(lambda h: on_chip("did I take medicine today", h), [chatbot], [chatbot, status_box])

    demo_btn.click(lambda: load_demo(), outputs=[chatbot, status_box])
    refresh_btn.click(get_status_text, outputs=[status_box])
    reset_btn.click(lambda: reset_all(), outputs=[chatbot, status_box])
    gen_btn.click(generate_report_text, [rtype], [report_box])

    audio_in.stop_recording(on_voice, [audio_in, chatbot], [chatbot, status_box])
    img_in.change(on_image, [img_in, chatbot], [chatbot, status_box])

    # TTS — browser speaks responses in Malayalam
    tts_js = """
    () => {
        const chatEl = document.querySelector('#chatbox');
        if (!chatEl) return;
        const observer = new MutationObserver(() => {
            const msgs = chatEl.querySelectorAll('.bot .message-bubble');
            if (!msgs.length) return;
            const last = msgs[msgs.length - 1];
            if (last.dataset.spoken === 'true') return;
            last.dataset.spoken = 'true';
            let text = (last.innerText || '')
                .replace(/[✓⎯·🩺🩸💊⏰🏥🤒✅❤️📊📈🚨⚠️📷🎉🌱🙏]/g, '')
                .replace(/```[\\s\\S]*?```/g, '').replace(/`[^`]*`/g, '')
                .replace(/\\n+/g, '. ').trim();
            if (text.length < 3 || text.length > 250) return;
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 0.85; u.lang = 'ml-IN';
            const voices = window.speechSynthesis.getVoices();
            const ml = voices.find(v => v.lang.startsWith('ml'));
            if (ml) u.voice = ml;
            window.speechSynthesis.speak(u);
        });
        observer.observe(chatEl, { childList: true, subtree: true });
        window.speechSynthesis.getVoices();
    }
    """
    demo.load(fn=None, js=tts_js)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
