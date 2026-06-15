"""
AgentEngine — Executes intents deterministically, builds context for LLM.
Ported from TypeScript. Bridges IntentParser → HealthStore → LLM context.
"""

from intent_parser import parse_intent, Intent, VitalData, MedicationData, ConditionData, LabResultData, ReminderData
from health_store import HealthStore
from typing import Dict, List, Optional


class AgentResult:
    def __init__(self):
        self.intent_type: str = ""
        self.tools_executed: List[Dict] = []
        self.context_for_llm: str = ""
        self.alert: Optional[str] = None
        self.badge: str = ""


def run_agent(user_input: str, store: HealthStore) -> AgentResult:
    """Process user input through intent parsing and tool execution."""
    intent = parse_intent(user_input)
    result = AgentResult()
    result.intent_type = intent.type

    try:
        if intent.type == "add_vital":
            data: VitalData = intent.data
            save_result = store.save_vital(
                data.type, data.primary, data.secondary, data.unit, data.context
            )
            result.tools_executed.append({"tool": "save_vital", "success": True})
            result.alert = save_result.get("alert")
            result.badge = "❤️"

            # Check compound risks
            compound_alerts = store.check_compound_risks()
            if compound_alerts:
                result.alert = compound_alerts[0]

            label = save_result["label"]
            value_str = save_result["value_str"]
            unit = save_result["unit"]
            context = save_result.get("context", "")
            trend = save_result.get("trend_info", "")
            alert_text = f" ALERT: {result.alert}" if result.alert else " Value normal."

            result.context_for_llm = (
                f"Patient recorded {label}: {value_str} {unit}"
                f"{' (' + context + ')' if context else ''}."
                f"{alert_text}{trend} "
                f"Respond in nadan Malayalam — acknowledge, note if normal/abnormal, mention trend."
            )

        elif intent.type == "add_medication":
            data: MedicationData = intent.data
            save_result = store.save_medication(
                data.name, data.dosage, data.frequency, data.duration, data.times, data.notes
            )
            result.tools_executed.append({"tool": "save_medication", "success": True})
            result.tools_executed.append({"tool": "schedule_reminder", "success": True})
            result.badge = "💊 ⏰"

            interaction = save_result.get("interaction")
            if interaction:
                result.alert = interaction
                result.tools_executed.append({"tool": "drug_interaction_check", "success": True})

            result.context_for_llm = (
                f"Saved medication: {data.name} {data.dosage} {data.frequency}"
                f"{' for ' + data.duration if data.duration else ''}. "
                f"Reminder set at {', '.join(data.times)}. "
                f"{data.notes if data.notes else ''}"
                f"{'⚠️ INTERACTION: ' + interaction if interaction else ''} "
                f"Confirm in nadan Malayalam."
            )

        elif intent.type == "add_condition":
            data: ConditionData = intent.data
            store.save_condition(data.name, data.severity, data.icd_code)
            result.tools_executed.append({"tool": "save_condition", "success": True})
            result.badge = "🏥"
            result.context_for_llm = (
                f"Recorded condition: {data.name} ({data.severity}, ICD: {data.icd_code}). "
                f"Ask follow-up: when diagnosed, current treatment. Respond in nadan Malayalam."
            )

        elif intent.type == "add_lab_result":
            data: LabResultData = intent.data
            save_result = store.save_lab_result(
                data.test_name, data.value, data.unit, data.ref_low, data.ref_high
            )
            result.tools_executed.append({"tool": "save_lab_result", "success": True})
            result.badge = "🧪"
            abnormal = "ABNORMAL — outside normal range." if save_result["is_abnormal"] else "Within normal range."
            result.context_for_llm = (
                f"Lab result: {data.test_name} = {data.value} {data.unit} "
                f"(normal: {data.ref_low}-{data.ref_high}). {abnormal} "
                f"Explain in nadan Malayalam."
            )

        elif intent.type == "mark_taken":
            store.mark_taken(intent.medication)
            result.tools_executed.append({"tool": "log_adherence", "success": True})
            result.badge = "✅"
            result.context_for_llm = (
                f"Patient confirmed taking {intent.medication}. "
                f"Acknowledge in nadan Malayalam, encourage them."
            )

        elif intent.type == "set_reminder":
            data: ReminderData = intent.data
            store.set_reminder(data.medication, data.times, data.dosage)
            result.tools_executed.append({"tool": "set_reminder", "success": True})
            result.badge = "⏰"
            result.context_for_llm = (
                f"Reminder set: \"{data.medication}\" at {', '.join(data.times)}. "
                f"Confirm in nadan Malayalam."
            )

        elif intent.type == "stop_medication":
            stop_result = store.stop_medication(intent.medication)
            if stop_result["success"]:
                result.tools_executed.append({"tool": "stop_medication", "success": True})
                result.context_for_llm = (
                    f"Stopped medication: {stop_result['name']}. Reminders also deactivated. "
                    f"Confirm in nadan Malayalam. Ask if doctor advised stopping."
                )
            else:
                result.context_for_llm = (
                    f"Could not find active medication matching \"{intent.medication}\". "
                    f"Ask user to clarify. Respond in nadan Malayalam."
                )

        elif intent.type == "stop_reminder":
            stop_result = store.stop_reminder(intent.medication)
            if stop_result["success"]:
                result.tools_executed.append({"tool": "stop_reminder", "success": True})
                result.context_for_llm = f"Reminder stopped. Confirm in nadan Malayalam."
            else:
                result.context_for_llm = "No active reminder found. Tell them in nadan Malayalam."

        elif intent.type == "query_medications":
            meds = store.get_active_medications()
            if meds:
                med_list = ", ".join(f"{m.name} {m.dosage} ({m.frequency})" for m in meds)
                result.context_for_llm = f"Patient's active medications: {med_list}. List them in nadan Malayalam."
            else:
                result.context_for_llm = "No medications recorded. Tell them in nadan Malayalam."

        elif intent.type == "query_conditions":
            conds = store.get_conditions()
            if conds:
                cond_list = ", ".join(f"{c.name} ({c.status})" for c in conds)
                result.context_for_llm = f"Patient's conditions: {cond_list}. Summarize in nadan Malayalam."
            else:
                result.context_for_llm = "No conditions recorded. Tell them in nadan Malayalam."

        elif intent.type == "query_reminders":
            rems = store.get_active_reminders()
            if rems:
                rem_list = ", ".join(f"{r.medication} {r.dosage} at {', '.join(r.times)}" for r in rems)
                result.context_for_llm = f"Active reminders: {rem_list}. Tell them in nadan Malayalam."
            else:
                result.context_for_llm = "No active reminders. Tell them in nadan Malayalam."

        elif intent.type == "query_vitals":
            vitals = store.get_recent_vitals(5)
            if vitals:
                v_list = ", ".join(
                    f"{v.vital_type}: {v.primary}/{v.secondary if v.secondary else ''}{v.unit} ({v.recorded_at[:10]})"
                    for v in vitals
                )
                result.context_for_llm = f"Recent vitals: {v_list}. Summarize trends in nadan Malayalam."
            else:
                result.context_for_llm = "No vitals recorded. Tell them in nadan Malayalam."

        elif intent.type == "query_lab_results":
            labs = store.get_lab_results()
            if labs:
                lab_list = ", ".join(
                    f"{l.test_name}: {l.value}{l.unit} ({'⚠️' if l.is_abnormal else '✓'})"
                    for l in labs
                )
                result.context_for_llm = f"Lab results: {lab_list}. Explain in nadan Malayalam."
            else:
                result.context_for_llm = "No lab results recorded. Tell them in nadan Malayalam."

        elif intent.type == "query_today_doses":
            doses = store.get_today_doses()
            taken = doses["taken"]
            remaining = doses["remaining"]
            active = doses["active_meds"]
            if taken:
                taken_list = ", ".join(f"{t.medication_name}" for t in taken)
                remaining_list = ", ".join(f"{m.name} {m.dosage}" for m in remaining)
                result.context_for_llm = (
                    f"ഇന്ന് കഴിച്ചവ: {taken_list}. "
                    f"{'ഇനി: ' + remaining_list if remaining else 'എല്ലാം കഴിച്ചു!'} "
                    f"Tell clearly in Malayalam."
                )
            elif active:
                med_list = ", ".join(f"{m.name} {m.dosage}" for m in active)
                result.context_for_llm = (
                    f"ഇന്ന് ഒന്നും record ചെയ്തിട്ടില്ല. Active: {med_list}. "
                    f"Tell them in Malayalam — gently remind."
                )
            else:
                result.context_for_llm = "No medications set up. Tell them in nadan Malayalam."

        elif intent.type == "query_adherence":
            adherence = store.get_adherence_rate()
            rate = adherence["rate"]
            result.context_for_llm = (
                f"Adherence rate (7 days): {rate}% ({adherence['logged']} doses / ~{adherence['expected']} expected). "
                f"{'Praise them!' if rate >= 80 else 'Gently encourage.'} Respond in nadan Malayalam."
            )

        elif intent.type == "symptom_report":
            store.save_symptom(intent.text or user_input)
            result.tools_executed.append({"tool": "save_symptom", "success": True})
            result.badge = "🤒"

            conds = store.get_conditions()
            cond_info = f" Patient has: {', '.join(c.name for c in conds)}." if conds else ""
            result.context_for_llm = (
                f"Patient reports: {user_input}.{cond_info} Symptom recorded. "
                f"Respond in Malayalam — short, warm. If mild, home care. If concerning, suggest doctor."
            )

        elif intent.type == "general_chat":
            # Pure conversation — no tools, let LLM handle
            result.context_for_llm = user_input

        else:
            result.context_for_llm = user_input

    except Exception as e:
        result.tools_executed.append({"tool": "error", "success": False, "message": str(e)})
        result.context_for_llm = f"Error: {str(e)}. Apologize in Malayalam and ask to try again."

    return result
