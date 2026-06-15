"""
HealthStore — In-memory patient health database for the Gradio demo.
Replaces WatermelonDB from the React Native version.
Stores vitals, medications, conditions, reminders, adherence, lab results.
"""

import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from intent_parser import check_drug_interactions, get_vital_alert


@dataclass
class VitalRecord:
    vital_type: str
    primary: float
    secondary: float = 0
    unit: str = ""
    context: str = ""
    recorded_at: str = ""  # ISO datetime

@dataclass
class MedicationRecord:
    name: str
    dosage: str = ""
    frequency: str = ""
    start_date: str = ""
    end_date: str = ""
    notes: str = ""
    is_active: bool = True

@dataclass
class ConditionRecord:
    name: str
    severity: str = "moderate"
    icd_code: str = ""
    status: str = "active"
    diagnosed_date: str = ""

@dataclass
class ReminderRecord:
    reminder_id: str
    medication: str
    dosage: str = ""
    times: List[str] = field(default_factory=list)
    tts_message: str = ""
    is_active: bool = True

@dataclass
class AdherenceRecord:
    medication_name: str
    taken_at: str = ""
    date: str = ""

@dataclass
class LabResultRecord:
    test_name: str
    value: float
    unit: str
    ref_low: float
    ref_high: float
    is_abnormal: bool = False
    test_date: str = ""

@dataclass
class SymptomRecord:
    description: str
    recorded_at: str = ""


class HealthStore:
    """In-memory patient health data store."""

    def __init__(self):
        self.vitals: List[VitalRecord] = []
        self.medications: List[MedicationRecord] = []
        self.conditions: List[ConditionRecord] = []
        self.reminders: List[ReminderRecord] = []
        self.adherence_log: List[AdherenceRecord] = []
        self.lab_results: List[LabResultRecord] = []
        self.symptoms: List[SymptomRecord] = []
        self._reminder_counter = 0

    # ─── VITALS ──────────────────────────────────────────────────────────────

    def save_vital(self, vital_type: str, primary: float, secondary: float = 0,
                   unit: str = "", context: str = "") -> Dict:
        """Save a vital reading and return result with trend info."""
        now = datetime.now().isoformat()
        record = VitalRecord(
            vital_type=vital_type, primary=primary, secondary=secondary,
            unit=unit, context=context, recorded_at=now
        )
        self.vitals.append(record)

        # Trend analysis
        prev = [v for v in self.vitals[:-1] if v.vital_type == vital_type]
        trend_info = ""
        if len(prev) >= 1:
            last = prev[-1].primary
            diff = primary - last
            if abs(diff) > 0:
                direction = "↑" if diff > 0 else "↓"
                trend_info = f" {direction} കഴിഞ്ഞ തവണ {last}{unit} ആയിരുന്നു."
            if len(prev) >= 3:
                avg = sum(v.primary for v in prev[-3:]) / 3
                trend_info += f" 3-reading avg: {avg:.0f}{unit}."

        # Alert
        alert = get_vital_alert(vital_type, primary, secondary)

        type_labels = {
            'bp': 'BP', 'sugar': 'Sugar', 'spo2': 'SpO2',
            'temperature': 'Temperature', 'heart_rate': 'Heart Rate',
            'weight': 'Weight', 'pain': 'Pain'
        }
        label = type_labels.get(vital_type, vital_type)
        value_str = f"{primary}/{secondary}" if secondary else f"{primary}"

        return {
            "success": True,
            "tool": "save_vital",
            "label": label,
            "value_str": value_str,
            "unit": unit,
            "context": context,
            "trend_info": trend_info,
            "alert": alert,
        }

    # ─── MEDICATIONS ─────────────────────────────────────────────────────────

    def save_medication(self, name: str, dosage: str = "", frequency: str = "",
                        duration: str = "", times: List[str] = None, notes: str = "") -> Dict:
        """Save a medication and auto-create reminder."""
        today = datetime.now().strftime("%Y-%m-%d")
        times = times or ["08:00"]

        # Drug interaction check
        existing = [m.name for m in self.medications if m.is_active]
        interaction = check_drug_interactions(name, existing)

        # Save medication
        end_date = ""
        if duration:
            import re
            d_match = re.search(r'(\d+)', duration)
            if d_match:
                days = int(d_match.group(1))
                end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        self.medications.append(MedicationRecord(
            name=name, dosage=dosage, frequency=frequency,
            start_date=today, end_date=end_date, notes=notes, is_active=True
        ))

        # Auto-create reminder
        self._reminder_counter += 1
        rem_id = f"rem_{self._reminder_counter}"
        tts_msg = f"{name} {dosage} കഴിക്കാൻ സമയമായി."
        if notes:
            if "after food" in notes.lower():
                tts_msg += " ഭക്ഷണം കഴിച്ചതിന് ശേഷം."
            elif "before food" in notes.lower():
                tts_msg += " ഭക്ഷണത്തിന് മുമ്പ്."

        self.reminders.append(ReminderRecord(
            reminder_id=rem_id, medication=name, dosage=dosage,
            times=times, tts_message=tts_msg, is_active=True
        ))

        return {
            "success": True,
            "tools": ["save_medication", "schedule_reminder"],
            "name": name,
            "dosage": dosage,
            "frequency": frequency,
            "times": times,
            "notes": notes,
            "interaction": interaction,
            "reminder_id": rem_id,
        }

    # ─── CONDITIONS ──────────────────────────────────────────────────────────

    def save_condition(self, name: str, severity: str = "moderate", icd_code: str = "") -> Dict:
        """Save a diagnosed condition."""
        today = datetime.now().strftime("%Y-%m-%d")
        self.conditions.append(ConditionRecord(
            name=name, severity=severity, icd_code=icd_code,
            status="active", diagnosed_date=today
        ))
        return {"success": True, "tool": "save_condition", "name": name, "severity": severity}

    # ─── LAB RESULTS ─────────────────────────────────────────────────────────

    def save_lab_result(self, test_name: str, value: float, unit: str,
                        ref_low: float, ref_high: float) -> Dict:
        """Save a lab result."""
        today = datetime.now().strftime("%Y-%m-%d")
        is_abnormal = value < ref_low or value > ref_high
        self.lab_results.append(LabResultRecord(
            test_name=test_name, value=value, unit=unit,
            ref_low=ref_low, ref_high=ref_high,
            is_abnormal=is_abnormal, test_date=today
        ))
        return {
            "success": True, "tool": "save_lab_result",
            "test_name": test_name, "value": value, "unit": unit,
            "ref_range": f"{ref_low}-{ref_high}", "is_abnormal": is_abnormal,
        }

    # ─── ADHERENCE ───────────────────────────────────────────────────────────

    def mark_taken(self, medication: str) -> Dict:
        """Log that a medication was taken."""
        now = datetime.now()
        self.adherence_log.append(AdherenceRecord(
            medication_name=medication,
            taken_at=now.isoformat(),
            date=now.strftime("%Y-%m-%d")
        ))
        return {"success": True, "tool": "log_adherence", "medication": medication}

    # ─── REMINDERS ───────────────────────────────────────────────────────────

    def set_reminder(self, title: str, times: List[str], dosage: str = "") -> Dict:
        """Set a custom reminder."""
        self._reminder_counter += 1
        rem_id = f"rem_{self._reminder_counter}"
        self.reminders.append(ReminderRecord(
            reminder_id=rem_id, medication=title, dosage=dosage,
            times=times, tts_message=f"{title} — സമയമായി", is_active=True
        ))
        return {"success": True, "tool": "set_reminder", "title": title, "times": times, "id": rem_id}

    # ─── STOP ────────────────────────────────────────────────────────────────

    def stop_medication(self, name: str) -> Dict:
        """Deactivate a medication."""
        found = None
        for med in self.medications:
            if med.is_active and name.lower() in med.name.lower():
                med.is_active = False
                found = med.name
                # Also stop reminders
                for rem in self.reminders:
                    if rem.is_active and rem.medication.lower() == med.name.lower():
                        rem.is_active = False
                break
        if found:
            return {"success": True, "tool": "stop_medication", "name": found}
        return {"success": False, "message": f"No active medication matching '{name}'"}

    def stop_reminder(self, medication: str = "") -> Dict:
        """Deactivate a reminder."""
        for rem in reversed(self.reminders):
            if rem.is_active and (not medication or medication.lower() in rem.medication.lower()):
                rem.is_active = False
                return {"success": True, "tool": "stop_reminder", "medication": rem.medication}
        return {"success": False, "message": "No active reminder found"}

    # ─── SYMPTOMS ────────────────────────────────────────────────────────────

    def save_symptom(self, description: str) -> Dict:
        """Record a symptom."""
        self.symptoms.append(SymptomRecord(
            description=description, recorded_at=datetime.now().isoformat()
        ))
        return {"success": True, "tool": "save_symptom", "description": description[:80]}

    # ─── QUERIES ─────────────────────────────────────────────────────────────

    def get_active_medications(self) -> List[MedicationRecord]:
        return [m for m in self.medications if m.is_active]

    def get_conditions(self) -> List[ConditionRecord]:
        return self.conditions

    def get_active_reminders(self) -> List[ReminderRecord]:
        return [r for r in self.reminders if r.is_active]

    def get_recent_vitals(self, n: int = 10, vital_type: str = "") -> List[VitalRecord]:
        filtered = [v for v in self.vitals if (not vital_type or v.vital_type == vital_type)]
        return filtered[-n:]

    def get_lab_results(self) -> List[LabResultRecord]:
        return self.lab_results[-10:]

    def get_today_doses(self) -> Dict:
        """Get today's adherence status."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_logs = [a for a in self.adherence_log if a.date == today]
        active_meds = self.get_active_medications()
        taken_names = {a.medication_name.lower() for a in today_logs}
        remaining = [m for m in active_meds if m.name.lower() not in taken_names]
        return {"taken": today_logs, "remaining": remaining, "active_meds": active_meds}

    def get_adherence_rate(self, days: int = 7) -> Dict:
        """Calculate adherence rate over N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        logs = [a for a in self.adherence_log if a.date >= cutoff]
        active_meds = self.get_active_medications()
        expected = len(active_meds) * days
        rate = round((len(logs) / expected) * 100) if expected > 0 else 0
        return {"rate": min(rate, 100), "logged": len(logs), "expected": expected}

    # ─── PROACTIVE HEALTH ────────────────────────────────────────────────────

    def check_compound_risks(self) -> List[str]:
        """Check for dangerous vital combinations."""
        alerts = []
        cond_names = [c.name.lower() for c in self.conditions]
        has_diabetes = any('diabetes' in c for c in cond_names)

        # Get latest vitals by type
        latest: Dict[str, VitalRecord] = {}
        for v in self.vitals[-20:]:
            latest[v.vital_type] = v

        # Diabetes + High Sugar + High BP
        if has_diabetes and 'sugar' in latest and latest['sugar'].primary > 200:
            if 'bp' in latest and latest['bp'].primary > 140:
                alerts.append("🚨 ഷുഗറും BP-യും ഒരുമിച്ച് ഉയർന്നിരിക്കുന്നു. ഡോക്ടറെ വിളിക്കുക.")

        # Low SpO2 + Fever
        if 'spo2' in latest and latest['spo2'].primary < 94:
            if 'temperature' in latest and latest['temperature'].primary > 100.4:
                alerts.append("🚨 ഓക്സിജൻ കുറവും പനിയും ഒരുമിച്ച്. ഉടൻ ആശുപത്രിയിൽ പോകുക.")

        # Hypoglycemia in diabetic
        if has_diabetes and 'sugar' in latest and latest['sugar'].primary < 70:
            alerts.append("🚨 ഷുഗർ വളരെ കുറവ്! ഉടനെ മധുരം കഴിക്കുക.")

        return alerts

    # ─── HEALTH REPORT ───────────────────────────────────────────────────────

    def generate_health_report(self, patient_name: str = "Patient") -> str:
        """Generate a formatted health report."""
        lines = []
        lines.append("═══════════════════════════════════════")
        lines.append("       THUNA HEALTH REPORT")
        lines.append("═══════════════════════════════════════")
        lines.append(f"Patient: {patient_name}")
        lines.append(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        lines.append("")

        # Conditions
        lines.append("── CONDITIONS ──────────────────────")
        if not self.conditions:
            lines.append("  None recorded")
        else:
            for c in self.conditions:
                lines.append(f"  • {c.name} ({c.status}) — {c.severity}")
        lines.append("")

        # Medications
        lines.append("── ACTIVE MEDICATIONS ──────────────")
        active_meds = self.get_active_medications()
        if not active_meds:
            lines.append("  None")
        else:
            for m in active_meds:
                lines.append(f"  • {m.name} {m.dosage} — {m.frequency}")
        lines.append("")

        # Vitals
        lines.append("── RECENT VITALS ───────────────────")
        recent = self.get_recent_vitals(15)
        if not recent:
            lines.append("  No readings")
        else:
            by_type: Dict[str, List[VitalRecord]] = {}
            for v in recent:
                by_type.setdefault(v.vital_type, []).append(v)
            labels = {'bp': 'Blood Pressure', 'sugar': 'Blood Sugar', 'spo2': 'SpO2',
                      'temperature': 'Temperature', 'heart_rate': 'Heart Rate', 'weight': 'Weight'}
            for vtype, readings in by_type.items():
                lines.append(f"  {labels.get(vtype, vtype)}:")
                for r in readings[-5:]:
                    val = f"{r.primary}/{r.secondary}" if r.secondary else f"{r.primary}"
                    dt = r.recorded_at[:16] if r.recorded_at else ""
                    lines.append(f"    {dt}: {val} {r.unit}")
        lines.append("")

        # Lab Results
        if self.lab_results:
            lines.append("── LAB RESULTS ─────────────────────")
            for l in self.lab_results[-5:]:
                flag = " ⚠️" if l.is_abnormal else ""
                lines.append(f"  • {l.test_name}: {l.value} {l.unit} (ref: {l.ref_low}-{l.ref_high}){flag}")
            lines.append("")

        # Adherence
        adherence = self.get_adherence_rate()
        lines.append("── ADHERENCE (7 days) ──────────────")
        lines.append(f"  Rate: {adherence['rate']}%")
        lines.append(f"  Doses logged: {adherence['logged']}")
        lines.append("")

        lines.append("═══════════════════════════════════════")
        lines.append("Generated by Thuna (തുണ)")
        lines.append("Powered by Gemma 4 E2B (on-device)")

        return "\n".join(lines)

    # ─── FAMILY STATUS ───────────────────────────────────────────────────────

    def generate_family_status(self, patient_name: str = "Patient") -> str:
        """Generate a brief shareable status for family."""
        today = datetime.now().strftime("%Y-%m-%d")
        today_vitals = [v for v in self.vitals if v.recorded_at.startswith(today)]
        today_logs = [a for a in self.adherence_log if a.date == today]
        active_meds = self.get_active_medications()

        lines = []
        lines.append(f"📋 {patient_name} — {datetime.now().strftime('%d/%m/%Y')} Status")
        lines.append("")

        # Vitals today
        if today_vitals:
            lines.append("📊 Vitals:")
            labels = {'bp': 'BP', 'sugar': 'Sugar', 'spo2': 'SpO2', 'temperature': 'Temp', 'heart_rate': 'HR'}
            for v in today_vitals:
                val = f"{v.primary}/{v.secondary}" if v.secondary else f"{v.primary}"
                lines.append(f"  {labels.get(v.vital_type, v.vital_type)}: {val} {v.unit}")
        else:
            lines.append("📊 No vitals today")

        lines.append("")
        lines.append(f"💊 Medicines: {len(today_logs)}/{len(active_meds)} taken")
        for l in today_logs:
            lines.append(f"  ✅ {l.medication_name}")
        missed = [m for m in active_meds if m.name.lower() not in {l.medication_name.lower() for l in today_logs}]
        for m in missed:
            lines.append(f"  ⏳ {m.name} {m.dosage} — pending")

        # Symptoms
        today_symptoms = [s for s in self.symptoms if s.recorded_at.startswith(today)]
        if today_symptoms:
            lines.append("")
            lines.append("🤒 Reported:")
            for s in today_symptoms:
                lines.append(f"  {s.description[:60]}")

        lines.append("")
        lines.append("— Sent from Thuna (തുണ)")
        return "\n".join(lines)
