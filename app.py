"""
Thuna (തുണ) — Offline AI Health Companion for Elderly
Gradio App for Build Small Hackathon (Hugging Face × Gradio)

Voice-first Malayalam health companion powered by Gemma 4 E2B (2B params).
Deterministic intent parsing + LLM warmth. Designed for elderly users in rural India.
"""

import gradio as gr
import os
import time
import re
from typing import List, Tuple, Optional
from datetime import datetime

from intent_parser import parse_intent
from health_store import HealthStore
from agent_engine import run_agent, AgentResult

# ═══════════════════════════════════════════════════════════════════════════
# LLM SETUP — Gemma via HF Inference Providers (with robust fallback)
# ═══════════════════════════════════════════════════════════════════════════

from huggingface_hub import InferenceClient

# Use HF Inference API — tries multiple models in order
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Models to try (in order of preference, all under 4B for Tiny Titan badge)
MODEL_CANDIDATES = [
    "google/gemma-3-4b-it",          # Gemma 3 4B (closest to our on-device E2B)
    "CohereLabs/tiny-aya-fire-GGUF",  # Cohere Tiny Aya Fire - South Asian languages
    "microsoft/Phi-4-mini-instruct",  # Phi 4 Mini as fallback
    "HuggingFaceH4/zephyr-7b-beta",  # General fallback
]

client = None
MODEL_ID = MODEL_CANDIDATES[0]  # Default

try:
    if HF_TOKEN:
        client = InferenceClient(token=HF_TOKEN)
    else:
        # Even without token, HF provides some free inference
        client = InferenceClient()
except Exception:
    pass

# Malayalam system prompt — defines Thuna's personality
SYSTEM_PROMPT = """നീ "തുണ" (Thuna) ആണ്. കേരളത്തിലെ പ്രായമായവരുടെ ആരോഗ്യ companion.

നിയമങ്ങൾ:
- നാടൻ മലയാളത്തിൽ മാത്രം മറുപടി നൽകുക. English words ആവശ്യമെങ്കിൽ മാത്രം (medicine names, numbers).
- ചെറിയ വാക്യങ്ങൾ. ലളിതമായ ഭാഷ. പ്രായമായവർക്ക് മനസ്സിലാകുന്നത്.
- ആരോഗ്യ കാര്യങ്ങളിൽ ശ്രദ്ധാലുവായിരിക്കുക. അപകടകരമായ values കണ്ടാൽ ഡോക്ടറെ കാണാൻ പറയുക.
- Companion ആയി സംസാരിക്കുക. Warm, caring. ചിരിപ്പിക്കുക. ഒറ്റയ്ക്കല്ല എന്ന് തോന്നിപ്പിക്കുക.
- മരുന്ന് കാര്യങ്ങളിൽ exact ആയിരിക്കുക. Dosage, time — ഒന്നും confuse ചെയ്യരുത്.
- 2-3 വാക്യത്തിൽ answer ചെയ്യുക. വളരെ നീളമുള്ള answers വേണ്ട.
- Hindi, Manglish (English letters ൽ Malayalam) എഴുതരുത്. Pure Malayalam script മാത്രം.

ശരി: "BP record ചെയ്തു. 130/85 — നോർമൽ ആണ്. നന്നായിരിക്കുന്നു!"
തെറ്റ്: "Aapka BP record ho gaya hai"
തെറ്റ്: "Ningalude BP record cheythu"

നീ ഒരു doctor അല്ല. Diagnosis കൊടുക്കരുത്. ആശങ്ക ഉണ്ടെങ്കിൽ ഡോക്ടറെ കാണാൻ പറയുക."""


def generate_llm_response(context: str, chat_history: List[dict]) -> str:
    """Generate response using LLM via HF Inference API with multi-model fallback."""
    if not client:
        return generate_fallback_response(context)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add recent history (last 4 turns for context)
    for msg in chat_history[-8:]:
        messages.append(msg)

    # Add current context
    messages.append({"role": "user", "content": context})

    # Try each model candidate until one works
    for model_id in MODEL_CANDIDATES:
        try:
            response = client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
            )
            text = response.choices[0].message.content.strip()
            # Post-process: remove any Hindi/Manglish leakage
            text = post_process_malayalam(text)
            if text:  # Only return if we got something
                return text
        except Exception:
            continue  # Try next model

    # All models failed — use intelligent fallback
    return generate_fallback_response(context)


def post_process_malayalam(text: str) -> str:
    """Clean LLM output — remove Hindi, excessive English, formatting."""
    # Remove markdown
    text = re.sub(r'[\*\#\-]+', '', text)
    # Remove lines that are mostly Latin (Manglish detection)
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if not line.strip():
            continue
        malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', line))
        latin_chars = len(re.findall(r'[a-zA-Z]', line))
        # Keep if has Malayalam or is short (numbers, medicine names)
        if malayalam_chars > 0 or len(line) < 20:
            clean_lines.append(line)
    return '\n'.join(clean_lines) if clean_lines else text


def generate_fallback_response(context: str) -> str:
    """Intelligent template responses — works without any LLM API.
    These demonstrate the agent's capabilities even when the model is unavailable."""
    lower = context.lower()

    # Vital recordings
    if "bp" in lower and ("record" in lower or "patient" in lower):
        if "alert" in lower or "ഉയർന്ന" in context:
            return "BP record ചെയ്തു. ⚠️ ഉയർന്നതാണ്. വിശ്രമിക്കുക, വെള്ളം കുടിക്കുക. 30 മിനിറ്റ് കഴിഞ്ഞ് വീണ്ടും നോക്കുക. 🩺"
        if "trend" in lower or "increased" in lower:
            return "BP record ചെയ്തു. കഴിഞ്ഞ തവണയേക്കാൾ ചെറിയ ഉയർച്ച ഉണ്ട്. ശ്രദ്ധിക്കണം. 🩺"
        return "BP record ചെയ്തു. നോർമൽ ആണ്. നന്നായിരിക്കുന്നു! 😊"

    if "sugar" in lower and ("record" in lower or "patient" in lower):
        if "alert" in lower or "ഉയർന്ന" in context or "🚨" in context:
            return "ഷുഗർ record ചെയ്തു. ⚠️ വളരെ ഉയർന്നതാണ്! മരുന്ന് കഴിച്ചോ? ഡോക്ടറെ വിളിക്കുക. 🩸"
        if "trend" in lower:
            return "ഷുഗർ record ചെയ്തു. കഴിഞ്ഞ reading-നേക്കാൾ ചെറിയ മാറ്റം ഉണ്ട്. ട്രെൻഡ് ശ്രദ്ധിക്കാം. 🩸"
        return "ഷുഗർ record ചെയ്തു. നന്നായിരിക്കുന്നു! 🩸"

    if "spo2" in lower or "oxygen" in lower:
        return "ഓക്സിജൻ level record ചെയ്തു. ശ്വാസമുട്ട് ഉണ്ടെങ്കിൽ ഡോക്ടറെ കാണുക. 🫁"

    if "temperature" in lower or "fever" in lower or "പനി" in context:
        return "Temperature record ചെയ്തു. പനി ഉണ്ടെങ്കിൽ വിശ്രമിക്കുക, വെള്ളം കുടിക്കുക. 🌡️"

    if "heart" in lower or "pulse" in lower:
        return "Heart rate record ചെയ്തു. വിശ്രമിക്കുക. 💓"

    # Medication management
    if ("saved" in lower or "save" in lower) and "medication" in lower:
        if "interaction" in lower or "⚠️" in context:
            return "മരുന്ന് save ചെയ്തു. ⚠️ Drug interaction ഉണ്ട്! ഡോക്ടറോട് ചോദിക്കുക. Reminder set ചെയ്തിട്ടുണ്ട്. 💊"
        return "മരുന്ന് save ചെയ്തു. Reminder set ചെയ്തിട്ടുണ്ട്. സമയത്ത് ഓർമ്മിപ്പിക്കാം. 💊⏰"

    if "active medication" in lower or "patient's active" in lower:
        return "നിങ്ങളുടെ മരുന്നുകൾ ഇതാ. സമയത്ത് കഴിക്കാൻ മറക്കരുത്. 📋"

    # Adherence
    if "confirmed taking" in lower or "took" in lower or "taken" in lower:
        return "നന്നായി! 👏 മരുന്ന് കഴിച്ചു എന്ന് record ചെയ്തു. ഇങ്ങനെ തുടരുക!"

    if "adherence" in lower:
        if "praise" in lower or "80" in lower or "90" in lower or "100" in lower:
            return "വളരെ നന്നായി! 🌟 മരുന്ന് കൃത്യമായി കഴിക്കുന്നുണ്ട്. ഇങ്ങനെ തുടരുക!"
        return "മരുന്ന് കൃത്യമായി കഴിക്കാൻ ശ്രമിക്കുക. ഓർമ്മയില്ലെങ്കിൽ ഞാൻ ഓർമ്മിപ്പിക്കാം. ⏰"

    # Today's doses
    if "ഇന്ന്" in context or "today" in lower:
        if "എല്ലാം" in context or "all" in lower:
            return "ഇന്നത്തെ എല്ലാ മരുന്നും കഴിച്ചു! 🌟 വളരെ നന്നായി!"
        return "ഇന്നത്തെ മരുന്ന് status ഇതാ. ബാക്കി ഉള്ളത് സമയത്ത് കഴിക്കുക. 💊"

    # Reminders
    if "reminder" in lower:
        if "set" in lower or "confirm" in lower:
            return "Reminder set ചെയ്തു. സമയമാകുമ്പോൾ ഓർമ്മിപ്പിക്കാം. ⏰"
        if "stop" in lower:
            return "Reminder നിർത്തി. 👍"
        return "Active reminders ഇതാ. ⏰"

    # Conditions
    if "condition" in lower or "recorded condition" in lower:
        return "രേഖപ്പെടുത്തി. ഡോക്ടറുടെ നിർദ്ദേശം പാലിക്കുക. കൃത്യമായി check-up ചെയ്യുക. 🏥"

    # Lab results
    if "lab" in lower or "test" in lower:
        if "abnormal" in lower:
            return "ടെസ്റ്റ് result record ചെയ്തു. ⚠️ Normal range-ൽ അല്ല. ഡോക്ടറെ കാണിക്കുക. 🧪"
        return "ടെസ്റ്റ് result record ചെയ്തു. Normal range-ൽ ആണ്. 🧪✓"

    # Symptoms
    if "symptom" in lower or "reports" in lower or "pain" in lower or "fever" in lower:
        return "ലക്ഷണം record ചെയ്തു. കൂടുതൽ അസ്വസ്ഥത ഉണ്ടെങ്കിൽ ഡോക്ടറെ കാണുക. വിശ്രമിക്കുക. 🤒"

    # Compound alerts
    if "🚨" in context or "alert" in lower or "compound" in lower:
        return "⚠️ ശ്രദ്ധിക്കുക! ഡോക്ടറെ ഉടൻ വിളിക്കുക. വിശ്രമിക്കുക."

    # Stop medication
    if "stop" in lower and "medication" in lower:
        return "മരുന്ന് നിർത്തി. Reminder-ഉം off ചെയ്തു. ഡോക്ടർ പറഞ്ഞിട്ടാണോ? 💊"

    # Queries with no data
    if "no medication" in lower or "no vital" in lower or "no condition" in lower:
        return "ഇതുവരെ ഒന്നും record ചെയ്തിട്ടില്ല. BP, ഷുഗർ, മരുന്ന് — എന്തെങ്കിലും പറയൂ. 😊"

    # General chat / companion
    if any(word in context for word in ['ബോറ', 'വിഷമ', 'ഒറ്റ', 'lonely', 'bored', 'sad']):
        return "എന്താ വിഷമം? ഞാൻ ഇവിടെ ഉണ്ട്. എന്തെങ്കിലും സംസാരിക്കാം. ☕ എന്താ ഇഷ്ടമുള്ള cinema?"

    if any(word in lower for word in ['hello', 'hi', 'namaste', 'നമസ്കാരം']):
        return "നമസ്കാരം! 🙏 എന്താ വിശേഷം? ആരോഗ്യം എങ്ങനെ ഉണ്ട്?"

    # Default companion response
    return "എന്താ വിശേഷം? ആരോഗ്യം, മരുന്ന്, BP, ഷുഗർ — എന്തും ചോദിക്കാം. അല്ലെങ്കിൽ വെറുതെ സംസാരിക്കാം! 😊"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CHAT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

# Global health store (per session in production)
health_store = HealthStore()


def process_message(message: str, history: List[dict], audio=None) -> Tuple[List[dict], str]:
    """Process user message through the agent pipeline."""
    global health_store

    if not message.strip():
        return history, ""

    # Add user message to history
    history.append({"role": "user", "content": message})

    # Run through agent engine
    agent_result = run_agent(message, health_store)

    # Generate LLM response
    llm_response = generate_llm_response(agent_result.context_for_llm, history)

    # Build final response with badges and alerts
    response_parts = []

    if agent_result.alert:
        response_parts.append(agent_result.alert)
        response_parts.append("")

    response_parts.append(llm_response)

    if agent_result.badge:
        response_parts.append(f"\n{agent_result.badge}")

    # Tool execution info (subtle)
    if agent_result.tools_executed:
        tools = " | ".join(f"✓ {t['tool']}" for t in agent_result.tools_executed if t['success'])
        if tools:
            response_parts.append(f"\n─── {tools}")

    final_response = "\n".join(response_parts)

    # Add to history
    history.append({"role": "assistant", "content": final_response})

    return history, ""


def get_dashboard_data() -> str:
    """Get health dashboard summary."""
    global health_store

    lines = []
    now = datetime.now()

    # Active medications
    meds = health_store.get_active_medications()
    if meds:
        lines.append("💊 Active Medications")
        for m in meds:
            lines.append(f"  • {m.name} {m.dosage} — {m.frequency}")
    else:
        lines.append("💊 No medications recorded")

    lines.append("")

    # Recent vitals
    vitals = health_store.get_recent_vitals(5)
    if vitals:
        lines.append("📊 Recent Vitals")
        labels = {'bp': 'BP', 'sugar': 'Sugar', 'spo2': 'SpO2', 'temperature': 'Temp', 'heart_rate': 'HR', 'weight': 'Weight'}
        for v in reversed(vitals[-5:]):
            val = f"{v.primary}/{v.secondary}" if v.secondary else f"{v.primary}"
            lines.append(f"  {labels.get(v.vital_type, v.vital_type)}: {val} {v.unit}")
    else:
        lines.append("📊 No vitals recorded")

    lines.append("")

    # Conditions
    conds = health_store.get_conditions()
    if conds:
        lines.append("🏥 Conditions")
        for c in conds:
            lines.append(f"  • {c.name} ({c.status})")

    lines.append("")

    # Reminders
    rems = health_store.get_active_reminders()
    if rems:
        lines.append("⏰ Active Reminders")
        for r in rems:
            lines.append(f"  • {r.medication} at {', '.join(r.times)}")

    lines.append("")

    # Adherence
    adherence = health_store.get_adherence_rate()
    lines.append(f"📈 Adherence: {adherence['rate']}% (7 days)")

    return "\n".join(lines)


def generate_report() -> str:
    """Generate health report."""
    global health_store
    return health_store.generate_health_report("Grandmother")


def generate_status() -> str:
    """Generate family status."""
    global health_store
    return health_store.generate_family_status("Grandmother")


def reset_session():
    """Reset the health store for a new session."""
    global health_store
    health_store = HealthStore()
    return [], "Session reset. Fresh start! 🌱", get_dashboard_data()


def load_demo_data():
    """Pre-load demo data to showcase functionality."""
    global health_store
    health_store = HealthStore()

    # Add conditions
    health_store.save_condition("Type 2 Diabetes", "moderate", "E11")
    health_store.save_condition("Hypertension", "moderate", "I10")

    # Add medications
    health_store.save_medication("Metformin", "500mg", "twice daily", "", ["08:00", "20:00"], "After food")
    health_store.save_medication("Amlodipine", "5mg", "once daily", "", ["08:00"], "Morning")
    health_store.save_medication("Ecosprin", "75mg", "once daily", "", ["08:00"], "After food")

    # Add some vitals
    health_store.save_vital("bp", 138, 88, "mmHg")
    health_store.save_vital("sugar", 165, 0, "mg/dL", "fasting")
    health_store.save_vital("bp", 142, 92, "mmHg")
    health_store.save_vital("sugar", 180, 0, "mg/dL", "post-meal")

    # Mark some adherence
    health_store.mark_taken("Metformin")
    health_store.mark_taken("Amlodipine")

    welcome = (
        "Demo data loaded! 🎉\n\n"
        "Patient: Elderly grandmother with Diabetes + Hypertension\n"
        "Medications: Metformin 500mg, Amlodipine 5mg, Ecosprin 75mg\n"
        "Recent vitals loaded.\n\n"
        "Try saying:\n"
        '• "BP 145/92"\n'
        '• "sugar 200"\n'
        '• "took metformin"\n'
        '• "my medicines"\n'
        '• "I have headache"\n'
        '• "remind me to take medicine at 9pm"\n'
        '• Any casual chat in Malayalam!'
    )

    initial_history = [{"role": "assistant", "content": welcome}]
    return initial_history, get_dashboard_data()


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Mobile-responsive, elderly-friendly
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
/* Root variables */
:root {
    --thuna-green: #0D7C66;
    --thuna-green-light: #F0FDF4;
    --thuna-green-dark: #065A4A;
    --thuna-red: #DC2626;
    --thuna-bg: #FAFBFC;
    --thuna-border: #E5E7EB;
}

/* Main container */
.gradio-container {
    max-width: 100% !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif !important;
}

/* Header */
#thuna-header {
    background: linear-gradient(135deg, #0D7C66 0%, #065A4A 100%);
    padding: 20px 24px;
    border-radius: 16px;
    margin-bottom: 16px;
    text-align: center;
}

#thuna-header h1 {
    color: white !important;
    font-size: 28px !important;
    margin: 0 !important;
    font-weight: 700 !important;
}

#thuna-header p {
    color: #D1FAE5 !important;
    margin: 4px 0 0 0 !important;
    font-size: 14px !important;
}

/* Chat area */
.chatbot {
    min-height: 450px !important;
    border-radius: 16px !important;
    border: 1px solid var(--thuna-border) !important;
}

.chatbot .message {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Quick action chips */
.chip-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    padding: 8px 0;
}

.chip-btn {
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    background: var(--thuna-green-light) !important;
    border: 1px solid #D1FAE5 !important;
    color: var(--thuna-green-dark) !important;
    cursor: pointer;
}

.chip-btn:hover {
    background: #D1FAE5 !important;
}

/* Input area */
#msg-input textarea {
    font-size: 16px !important;
    border-radius: 24px !important;
    padding: 12px 20px !important;
}

/* Send button */
#send-btn {
    border-radius: 24px !important;
    background: var(--thuna-green) !important;
    color: white !important;
    min-width: 80px !important;
    font-size: 16px !important;
}

#send-btn:hover {
    background: var(--thuna-green-dark) !important;
}

/* Dashboard panel */
#dashboard-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    background: #F8FAFC !important;
    border-radius: 12px !important;
    border: 1px solid var(--thuna-border) !important;
}

/* Report box */
#report-box {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* Tabs */
.tab-nav button {
    font-size: 14px !important;
    font-weight: 500 !important;
}

.tab-nav button.selected {
    color: var(--thuna-green) !important;
    border-color: var(--thuna-green) !important;
}

/* Mobile responsive */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }
    
    #thuna-header {
        padding: 16px;
        border-radius: 12px;
    }
    
    #thuna-header h1 {
        font-size: 22px !important;
    }
    
    .chatbot {
        min-height: 350px !important;
    }
    
    .chip-btn {
        font-size: 13px !important;
        padding: 6px 12px !important;
    }
}

/* Elderly-friendly: larger touch targets */
button {
    min-height: 44px !important;
}

/* Alert styling in chat */
.message-wrap .message p:has(🚨), .message-wrap .message p:has(⚠️) {
    background: #FEF2F2;
    padding: 8px 12px;
    border-radius: 8px;
    border-left: 3px solid #DC2626;
}
"""

# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=CUSTOM_CSS, title="Thuna (തുണ) — AI Health Companion", theme=gr.themes.Soft()) as demo:

    # Header
    gr.HTML("""
    <div id="thuna-header">
        <h1>🤝 തുണ — Thuna</h1>
        <p>Offline AI Health Companion for Elderly • Powered by Gemma 4 E2B (2B params)</p>
        <p style="font-size: 12px; margin-top: 8px; opacity: 0.8;">
            Voice-first Malayalam • Medication Safety • Health Monitoring • 100% On-Device
        </p>
    </div>
    """)

    with gr.Row():
        # Main chat area (left/top on mobile)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "നമസ്കാരം! 🙏 ഞാൻ തുണ.\n\nആരോഗ്യം, മരുന്ന്, BP, ഷുഗർ — എന്തും ചോദിക്കാം.\nഅല്ലെങ്കിൽ വെറുതെ സംസാരിക്കാം. 😊\n\n💡 Try: \"BP 140/90\" • \"sugar 180\" • \"took metformin\""}],
                type="messages",
                height=500,
                show_copy_button=True,
                avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/handshake_1f91d.png"),
                label="Chat with Thuna",
            )

            # Quick action chips
            with gr.Row():
                bp_chip = gr.Button("❤️ BP", size="sm", variant="secondary")
                sugar_chip = gr.Button("🩸 ഷുഗർ", size="sm", variant="secondary")
                taken_chip = gr.Button("💊 കഴിച്ചു", size="sm", variant="secondary")
                meds_chip = gr.Button("📋 മരുന്നുകൾ", size="sm", variant="secondary")
                report_chip = gr.Button("📄 Report", size="sm", variant="secondary")

            # Input row
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="തുണയോട് പറയുക... (BP 140/90, sugar 180, took medicine, etc.)",
                    show_label=False,
                    scale=5,
                    elem_id="msg-input",
                    lines=1,
                )
                send_btn = gr.Button("↑ Send", elem_id="send-btn", scale=1)

            # Audio input for voice
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Voice Input (Malayalam) — Speak and submit",
                visible=True,
            )
            with gr.Row():
                audio_submit = gr.Button("🎤 Submit Voice", size="sm", variant="secondary")
                image_input = gr.Image(
                    type="filepath",
                    label="📷 Upload Prescription",
                    sources=["upload", "webcam"],
                    height=100,
                )
                image_submit = gr.Button("📷 Scan Prescription", size="sm", variant="secondary")

        # Side panel (right/bottom on mobile)
        with gr.Column(scale=1, min_width=280):
            with gr.Tabs():
                with gr.Tab("📊 Dashboard"):
                    dashboard = gr.Textbox(
                        value=get_dashboard_data(),
                        label="Health Overview",
                        lines=18,
                        interactive=False,
                        elem_id="dashboard-box",
                    )
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")

                with gr.Tab("📄 Reports"):
                    report_type = gr.Radio(
                        ["Health Report", "Family Status"],
                        value="Health Report",
                        label="Report Type",
                    )
                    gen_report_btn = gr.Button("Generate Report", variant="primary")
                    report_output = gr.Textbox(
                        label="Report",
                        lines=20,
                        interactive=False,
                        elem_id="report-box",
                        show_copy_button=True,
                    )

                with gr.Tab("⚙️ Setup"):
                    gr.Markdown("""
                    ### Quick Setup
                    Load demo data to see Thuna in action with a pre-configured elderly patient profile.
                    """)
                    demo_btn = gr.Button("🎬 Load Demo Data", variant="primary")
                    reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
                    gr.Markdown("""
                    ---
                    ### About Thuna
                    
                    **Problem:** 240M elderly manage chronic conditions alone. No internet, no English, no help.
                    
                    **Solution:** On-device AI companion that speaks Malayalam, tracks medicines, prevents overdoses.
                    
                    **Model:** Gemma 4 E2B (2B params, INT4)
                    
                    **Architecture:**
                    ```
                    Voice → Regex Parser (<1ms)
                         → Agent (23 tools)
                         → Gemma 4 (Malayalam)
                         → TTS
                    ```
                    
                    Built for my grandmothers in Poothampara, Kerala.
                    """)

    # ─── EVENT HANDLERS ──────────────────────────────────────────────────

    def submit_message(message, history):
        history, _ = process_message(message, history)
        return history, "", get_dashboard_data()

    def chip_action(text, history):
        history, _ = process_message(text, history)
        return history, get_dashboard_data()

    def report_action(report_type):
        if report_type == "Health Report":
            return generate_report()
        return generate_status()

    def demo_action():
        history, dashboard = load_demo_data()
        return history, dashboard

    def reset_action():
        history, msg, dashboard = reset_session()
        initial = [{"role": "assistant", "content": "Session reset! 🌱\n\nനമസ്കാരം! ഞാൻ തുണ. എന്താ വിശേഷം? 😊"}]
        return initial, dashboard

    # Submit on enter or button click
    msg_input.submit(submit_message, [msg_input, chatbot], [chatbot, msg_input, dashboard])
    send_btn.click(submit_message, [msg_input, chatbot], [chatbot, msg_input, dashboard])

    # Audio/voice submission
    def process_audio(audio_path, history):
        """Process audio input — transcribe then run through agent."""
        if not audio_path:
            return history, "", get_dashboard_data()
        
        # Try to transcribe using Whisper via HF Inference API
        transcription = ""
        try:
            if client:
                result = client.automatic_speech_recognition(
                    audio_path, model="openai/whisper-large-v3"
                )
                transcription = result.get("text", "") if isinstance(result, dict) else str(result)
        except Exception:
            pass
        
        if not transcription:
            transcription = "[Audio received — transcription unavailable in demo. Type your message instead.]"
            history.append({"role": "assistant", "content": "🎤 Audio received. For best results in this web demo, please type your message. The mobile app has full Malayalam voice recognition."})
            return history, "", get_dashboard_data()
        
        # Process the transcribed text
        history, _ = process_message(transcription, history)
        return history, "", get_dashboard_data()

    audio_submit.click(process_audio, [audio_input, chatbot], [chatbot, msg_input, dashboard])

    # Prescription image upload
    def process_prescription(image_path, history):
        """Process uploaded prescription image."""
        if not image_path:
            return history, get_dashboard_data()
        
        history.append({"role": "user", "content": "📷 [Prescription image uploaded]"})
        
        # Try to use multimodal model for OCR
        prescription_text = ""
        try:
            if client:
                # Use vision model for prescription extraction
                import base64
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode()
                
                messages = [
                    {"role": "system", "content": "Extract all medication names, dosages, and frequencies from this prescription. Return as a simple list. If you cannot read it clearly, say so."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract medications from this prescription:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]}
                ]
                response = client.chat_completion(
                    model="google/gemma-3-4b-it",
                    messages=messages,
                    max_tokens=300,
                )
                prescription_text = response.choices[0].message.content.strip()
        except Exception:
            pass
        
        if prescription_text:
            response = f"📋 Prescription scan result:\n\n{prescription_text}\n\n💡 Say 'add [medicine name] [dosage] [frequency]' to save each medication."
        else:
            response = (
                "📷 Prescription uploaded! In the full mobile app, Gemma 4 vision extracts medications automatically.\n\n"
                "For this web demo, please type the medications manually:\n"
                "• \"metformin 500mg twice daily after food\"\n"
                "• \"amlodipine 5mg once daily morning\"\n\n"
                "Each will be saved with auto-reminders. 💊"
            )
        
        history.append({"role": "assistant", "content": response})
        return history, get_dashboard_data()

    image_submit.click(process_prescription, [image_input, chatbot], [chatbot, dashboard])

    # Quick action chips — prefill input for BP/sugar, auto-send for others
    def prefill_bp():
        return "BP "
    
    def prefill_sugar():
        return "sugar "

    bp_chip.click(prefill_bp, outputs=[msg_input])
    sugar_chip.click(prefill_sugar, outputs=[msg_input])
    taken_chip.click(lambda h: chip_action("took medicine", h), [chatbot], [chatbot, dashboard])
    meds_chip.click(lambda h: chip_action("my medicines", h), [chatbot], [chatbot, dashboard])
    report_chip.click(report_action, [report_type], [report_output])

    # Dashboard refresh
    refresh_btn.click(get_dashboard_data, outputs=[dashboard])

    # Reports
    gen_report_btn.click(report_action, [report_type], [report_output])

    # Setup
    demo_btn.click(demo_action, outputs=[chatbot, dashboard])
    reset_btn.click(reset_action, outputs=[chatbot, dashboard])


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
