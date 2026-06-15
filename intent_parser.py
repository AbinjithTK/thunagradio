"""
IntentParser — Deterministic intent extraction from user input.
Ported from TypeScript to Python. Handles Malayalam + English.
Covers: Vitals, Medications, Conditions, Lab Results, Queries, Symptoms, Reminders.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VitalData:
    type: str       # bp, sugar, spo2, temperature, heart_rate, weight, pain
    primary: float
    secondary: float = 0
    unit: str = ""
    context: str = ""  # fasting, post-meal, morning, etc.

@dataclass
class MedicationData:
    name: str
    dosage: str = ""
    frequency: str = "as prescribed"
    duration: str = ""
    times: List[str] = field(default_factory=lambda: ["08:00"])
    notes: str = ""

@dataclass
class ConditionData:
    name: str
    severity: str = "moderate"
    icd_code: str = ""

@dataclass
class LabResultData:
    test_name: str
    value: float
    unit: str
    ref_low: float
    ref_high: float

@dataclass
class ReminderData:
    medication: str
    dosage: str = ""
    times: List[str] = field(default_factory=lambda: ["08:00"])
    tts_message: str = ""

@dataclass
class Intent:
    type: str  # add_vital, add_medication, add_condition, add_lab_result, set_reminder,
               # stop_medication, stop_reminder, mark_taken, query_medications,
               # query_conditions, query_reminders, query_vitals, query_lab_results,
               # query_adherence, query_today_doses, update_profile, symptom_report, general_chat
    data: object = None
    medication: str = ""
    field_name: str = ""
    value: str = ""
    text: str = ""

# ═══════════════════════════════════════════════════════════════════════════
# MEDICATION DATABASE
# ═══════════════════════════════════════════════════════════════════════════

MEDICATIONS = [
    'paracetamol', 'amoxicillin', 'metformin', 'amlodipine', 'atorvastatin',
    'omeprazole', 'pantoprazole', 'cetirizine', 'azithromycin', 'ibuprofen',
    'dolo', 'crocin', 'combiflam', 'augmentin', 'calpol', 'meftal',
    'rantac', 'pan', 'shelcal', 'ecosprin', 'thyronorm', 'glycomet',
    'telma', 'stamlo', 'aten', 'zifi', 'oflox', 'montair',
    'insulin', 'glimepiride', 'losartan', 'telmisartan', 'aspirin',
    'clopidogrel', 'warfarin', 'levothyroxine', 'prednisolone',
    'salbutamol', 'budesonide', 'montelukast', 'folic acid',
    'calcium', 'vitamin d', 'iron', 'b12', 'multivitamin',
    'atenolol', 'ramipril', 'sitagliptin', 'pioglitazone', 'rosuvastatin',
    'lisinopril', 'enalapril', 'vildagliptin', 'gliclazide', 'voglibose',
    'rabeprazole', 'domperidone', 'ondansetron', 'loperamide', 'doxycycline',
    'ciprofloxacin', 'norfloxacin', 'fluconazole', 'aceclofenac', 'diclofenac',
    'tramadol', 'gabapentin', 'pregabalin', 'duloxetine', 'sertraline',
]

# ═══════════════════════════════════════════════════════════════════════════
# VITAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

def parse_bp(text: str) -> Optional[VitalData]:
    """Parse blood pressure readings."""
    patterns = [
        r'(?:bp|blood\s*pressure|ബിപി|രക്തസമ്മർദ്ദം|pressure)\s*:?\s*(\d{2,3})\s*[\/\-\s]\s*(\d{2,3})',
        r'(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmhg|bp|ബിപി)?',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            sys_val = int(m.group(1))
            dia_val = int(m.group(2))
            if 70 <= sys_val <= 250 and 40 <= dia_val <= 150:
                return VitalData(type="bp", primary=sys_val, secondary=dia_val, unit="mmHg")
    return None

def parse_sugar(text: str) -> Optional[VitalData]:
    """Parse blood sugar readings."""
    patterns = [
        r'(?:sugar|glucose|ഷുഗർ|പഞ്ചസാര|fasting|pp)\s*:?\s*(\d{2,3})',
        r'(\d{2,3})\s*(?:ആണ്|anu|aayi)?\s*(?:sugar|ഷുഗർ|പഞ്ചസാര)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 30 <= val <= 600:
                context = "fasting" if re.search(r'fasting|ഫാസ്റ്റിംഗ്|empty|വെറും\s*വയറ്', text, re.IGNORECASE) else \
                          "post-meal" if re.search(r'pp|post|after\s*food|ഭക്ഷണ.*ശേഷം', text, re.IGNORECASE) else "random"
                return VitalData(type="sugar", primary=val, unit="mg/dL", context=context)
    return None

def parse_spo2(text: str) -> Optional[VitalData]:
    """Parse SpO2 readings."""
    m = re.search(r'(?:spo2|oxygen|saturation|ഓക്സിജൻ)\s*:?\s*(\d{2,3})\s*%?', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 50 <= val <= 100:
            return VitalData(type="spo2", primary=val, unit="%")
    return None

def parse_temperature(text: str) -> Optional[VitalData]:
    """Parse temperature readings."""
    m = re.search(r'(?:temp|temperature|fever|പനി)\s*:?\s*(\d{2,3}\.?\d?)\s*(?:°?[cf])?', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if val < 45:
            val = val * 9/5 + 32  # Celsius to Fahrenheit
        if 90 <= val <= 110:
            return VitalData(type="temperature", primary=round(val, 1), unit="°F")
    return None

def parse_heart_rate(text: str) -> Optional[VitalData]:
    """Parse heart rate readings."""
    m = re.search(r'(?:pulse|heart\s*rate|hr|പൾസ്)\s*:?\s*(\d{2,3})', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 30 <= val <= 220:
            return VitalData(type="heart_rate", primary=val, unit="bpm", context="resting")
    return None

def parse_weight(text: str) -> Optional[VitalData]:
    """Parse weight readings."""
    m = re.search(r'(?:weight|ഭാരം|തൂക്കം)\s*:?\s*(\d{2,3}\.?\d?)\s*(?:kg)?', text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        if 20 <= val <= 250:
            return VitalData(type="weight", primary=val, unit="kg")
    return None

def parse_pain(text: str) -> Optional[VitalData]:
    """Parse pain score."""
    m = re.search(r'(?:pain|വേദന)\s*(?:score|level)?\s*:?\s*(\d{1,2})\s*(?:\/10)?', text, re.IGNORECASE)
    if m:
        val = min(int(m.group(1)), 10)
        return VitalData(type="pain", primary=val, unit="/10")
    return None

# ═══════════════════════════════════════════════════════════════════════════
# LAB RESULT PATTERNS
# ═══════════════════════════════════════════════════════════════════════════

LAB_PATTERNS = [
    (r'hba1c\s*:?\s*(\d+\.?\d*)', 'HbA1c', '%', 4.0, 5.6),
    (r'tsh\s*:?\s*(\d+\.?\d*)', 'TSH', 'mIU/L', 0.4, 4.0),
    (r'creatinine\s*:?\s*(\d+\.?\d*)', 'Creatinine', 'mg/dL', 0.6, 1.2),
    (r'(?:hemoglobin|hb)\s*:?\s*(\d+\.?\d*)', 'Hemoglobin', 'g/dL', 12.0, 17.0),
    (r'cholesterol\s*:?\s*(\d+)', 'Total Cholesterol', 'mg/dL', 0, 200),
    (r'triglyceride\s*:?\s*(\d+)', 'Triglycerides', 'mg/dL', 0, 150),
    (r'uric\s*acid\s*:?\s*(\d+\.?\d*)', 'Uric Acid', 'mg/dL', 3.5, 7.2),
]

# ═══════════════════════════════════════════════════════════════════════════
# CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

CONDITIONS = [
    (r'diabetes|sugar\s*disease|പ്രമേഹം|ഷുഗർ', 'Type 2 Diabetes', 'E11'),
    (r'hypertension|high\s*bp|ഉയർന്ന\s*ബിപി|രക്തസമ്മർദ്ദം', 'Hypertension', 'I10'),
    (r'asthma|ആസ്ത്മ|ശ്വാസം\s*മുട്ട്', 'Asthma', 'J45'),
    (r'thyroid|തൈറോയ്ഡ്', 'Thyroid Disorder', 'E03'),
    (r'arthritis|joint\s*pain|സന്ധി\s*വേദന', 'Arthritis', 'M13'),
    (r'cholesterol|കൊളസ്ട്രോൾ', 'Hyperlipidemia', 'E78'),
    (r'kidney|ckd|വൃക്ക', 'Chronic Kidney Disease', 'N18'),
    (r'heart|cardiac|ഹൃദയം', 'Heart Disease', 'I25'),
    (r'copd|ശ്വാസകോശ', 'COPD', 'J44'),
    (r'anemia|രക്തക്കുറവ്', 'Anemia', 'D50'),
    (r'migraine|headache|തലവേദന', 'Migraine', 'G43'),
    (r'depression|വിഷാദം', 'Depression', 'F32'),
]

# ═══════════════════════════════════════════════════════════════════════════
# DRUG INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════

DRUG_INTERACTIONS = [
    (['warfarin', 'aspirin'], 'Warfarin + Aspirin: Increased bleeding risk. Consult doctor.'),
    (['warfarin', 'ibuprofen'], 'Warfarin + Ibuprofen: High bleeding risk. Avoid combination.'),
    (['metformin', 'alcohol'], 'Metformin + Alcohol: Risk of lactic acidosis.'),
    (['amlodipine', 'atorvastatin'], 'Amlodipine + Atorvastatin: Monitor for muscle pain (myopathy).'),
    (['losartan', 'potassium'], 'Losartan + Potassium: Risk of hyperkalemia. Monitor levels.'),
    (['metformin', 'glimepiride'], 'Metformin + Glimepiride: Monitor for hypoglycemia (low sugar).'),
    (['aspirin', 'clopidogrel'], 'Aspirin + Clopidogrel: Increased bleeding risk. Usually intentional but monitor.'),
    (['ramipril', 'losartan'], 'ACE inhibitor + ARB: Dual RAAS blockade. Risk of kidney injury.'),
    (['ramipril', 'telmisartan'], 'ACE inhibitor + ARB: Dual RAAS blockade. Risk of kidney injury.'),
    (['enalapril', 'losartan'], 'ACE inhibitor + ARB: Dual RAAS blockade. Risk of kidney injury.'),
    (['ciprofloxacin', 'theophylline'], 'Ciprofloxacin + Theophylline: Toxic theophylline levels.'),
    (['omeprazole', 'clopidogrel'], 'Omeprazole reduces Clopidogrel effectiveness. Use pantoprazole instead.'),
    (['gabapentin', 'pregabalin'], 'Gabapentin + Pregabalin: Duplicate therapy. Excessive sedation risk.'),
    (['tramadol', 'sertraline'], 'Tramadol + SSRI: Serotonin syndrome risk. Monitor closely.'),
    (['tramadol', 'duloxetine'], 'Tramadol + SNRI: Serotonin syndrome risk. Monitor closely.'),
]

# ═══════════════════════════════════════════════════════════════════════════
# VITAL ALERTS
# ═══════════════════════════════════════════════════════════════════════════

def get_vital_alert(vital_type: str, primary: float, secondary: float = 0) -> Optional[str]:
    """Check if a vital reading triggers a critical alert."""
    if vital_type == "bp":
        if primary >= 180 or secondary >= 120:
            return "🚨 BP വളരെ ഉയർന്നതാണ്! ഉടനെ ഡോക്ടറെ കാണുക."
        if primary >= 140 or secondary >= 90:
            return "⚠️ BP ഉയർന്നതാണ്. വിശ്രമിക്കുക, 30 മിനിറ്റ് കഴിഞ്ഞ് വീണ്ടും നോക്കുക."
        if primary < 90 or secondary < 60:
            return "⚠️ BP കുറവാണ്. വെള്ളം കുടിക്കുക, കിടക്കുക."
    elif vital_type == "sugar":
        if primary > 300:
            return "🚨 ഷുഗർ വളരെ ഉയർന്നതാണ്! ഉടനെ ആശുപത്രിയിൽ പോകുക."
        if primary > 200:
            return "⚠️ ഷുഗർ ഉയർന്നതാണ്. മരുന്ന് കഴിച്ചോ? ഡോക്ടറെ വിളിക്കുക."
        if primary < 70:
            return "🚨 ഷുഗർ വളരെ കുറവാണ്! ഉടനെ മധുരം കഴിക്കുക."
    elif vital_type == "spo2":
        if primary < 90:
            return "🚨 ഓക്സിജൻ വളരെ കുറവാണ്! ഉടനെ ആശുപത്രിയിൽ പോകുക."
        if primary < 94:
            return "⚠️ ഓക്സിജൻ കുറവാണ്. ആഴത്തിൽ ശ്വസിക്കുക."
    elif vital_type == "temperature":
        if primary > 103:
            return "🚨 പനി വളരെ കൂടുതലാണ്! ഉടനെ ആശുപത്രിയിൽ പോകുക."
        if primary > 100.4:
            return "⚠️ പനിയുണ്ട്. പാരസെറ്റമോൾ കഴിക്കുക, തണുത്ത തുണി വയ്ക്കുക."
    elif vital_type == "heart_rate":
        if primary > 120:
            return "⚠️ ഹൃദയമിടിപ്പ് കൂടുതലാണ്. വിശ്രമിക്കുക."
        if primary < 50:
            return "⚠️ ഹൃദയമിടിപ്പ് കുറവാണ്. തലകറക്കം ഉണ്ടെങ്കിൽ ഡോക്ടറെ വിളിക്കുക."
    return None

# ═══════════════════════════════════════════════════════════════════════════
# CHECK DRUG INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════

def check_drug_interactions(new_med: str, existing_meds: List[str]) -> Optional[str]:
    """Check for dangerous drug interactions."""
    lower_new = new_med.lower()
    lower_existing = [m.lower() for m in existing_meds]

    for drugs, warning in DRUG_INTERACTIONS:
        drug1, drug2 = drugs
        if (drug1 in lower_new and any(drug2 in e for e in lower_existing)) or \
           (drug2 in lower_new and any(drug1 in e for e in lower_existing)):
            return warning

    # Duplicate check
    if lower_new in lower_existing:
        return f"{new_med} is already in your active medications. Duplicate?"

    return None

# ═══════════════════════════════════════════════════════════════════════════
# FREQUENCY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

FREQUENCY_PATTERNS = [
    (r'3\s*times?\s*(?:daily|a day|per day)|thrice|ദിവസം\s*3|മൂന്ന്\s*നേരം', '3 times daily', ['08:00', '14:00', '20:00']),
    (r'twice\s*(?:daily|a day)|2\s*times?\s*(?:daily|a day)|ദിവസം\s*2|രണ്ട്\s*നേരം', 'twice daily', ['08:00', '20:00']),
    (r'once\s*(?:daily|a day)|1\s*time|ദിവസം\s*1|ഒരു\s*നേരം', 'once daily', ['08:00']),
    (r'every\s*8\s*hours?', 'every 8 hours', ['08:00', '16:00', '00:00']),
    (r'every\s*12\s*hours?', 'every 12 hours', ['08:00', '20:00']),
    (r'morning|രാവിലെ', 'morning', ['08:00']),
    (r'night|രാത്രി|bed\s*time', 'at night', ['21:00']),
    (r'before\s*food|ഭക്ഷണത്തിന്\s*മുമ്പ്|empty\s*stomach', 'before food', ['07:30', '19:30']),
    (r'after\s*food|ഭക്ഷണത്തിന്\s*ശേഷം', 'after food', ['08:30', '20:30']),
]

def extract_medication(text: str, med_name: str) -> MedicationData:
    """Extract full medication details from text."""
    dosage_match = re.search(r'(\d+\.?\d*)\s*(mg|ml|mcg|g|iu|units?|tablet|tab|cap|capsule|drops|puff)', text, re.IGNORECASE)
    dosage = dosage_match.group(0) if dosage_match else ""

    frequency = "as prescribed"
    times = ["08:00"]
    for pat, freq, t in FREQUENCY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            frequency = freq
            times = t
            break

    duration = ""
    dur_match = re.search(r'(\d+)\s*(days?|ദിവസം|weeks?|ആഴ്ച|months?|മാസം)', text, re.IGNORECASE)
    if dur_match:
        num = int(dur_match.group(1))
        unit = dur_match.group(2).lower()
        if 'week' in unit or 'ആഴ്ച' in unit:
            duration = f"{num * 7} days"
        elif 'month' in unit or 'മാസം' in unit:
            duration = f"{num * 30} days"
        else:
            duration = f"{num} days"

    notes = ""
    if re.search(r'after\s*food|ഭക്ഷണത്തിന്\s*ശേഷം', text, re.IGNORECASE):
        notes = "After food"
    elif re.search(r'before\s*food|ഭക്ഷണത്തിന്\s*മുമ്പ്|empty\s*stomach', text, re.IGNORECASE):
        notes = "Before food"

    return MedicationData(
        name=med_name.capitalize(),
        dosage=dosage,
        frequency=frequency,
        duration=duration,
        times=times,
        notes=notes,
    )

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_intent(text: str) -> Intent:
    """Parse user input into a structured intent."""
    if not text or not isinstance(text, str):
        return Intent(type="general_chat", text="")
    
    text = text.strip()[:2000]
    lower = text.lower()

    # PRIORITY 1: "Did I take medicine today?"
    if re.search(r'did i take|did i have|കഴിച്ചോ|എടുത്തോ|ഇന്ന്.*മരുന്ന്|today.*medicine|already.*took|ഇന്ന്.*കഴിച്ച|മരുന്ന്.*കഴിച്ചോ|ഗുളിക.*കഴിച്ചോ', text, re.IGNORECASE):
        return Intent(type="query_today_doses")

    # PRIORITY 2: Mark medicine as taken
    if re.search(r'took|taken|കഴിച്ചു|എടുത്തു|had my|മരുന്ന്\s*കഴിച്ചു|ഗുളിക\s*കഴിച്ചു|tablet\s*കഴിച്ചു', text, re.IGNORECASE) and \
       not re.search(r'did|ചോ\?|ഓ\?|കഴിച്ചോ', text, re.IGNORECASE):
        med = next((m for m in MEDICATIONS if m.lower() in lower), None)
        if med:
            return Intent(type="mark_taken", medication=med)
        if re.search(r'sugar|ഷുഗർ|diabetes', text, re.IGNORECASE):
            return Intent(type="mark_taken", medication="diabetes medicine")
        if re.search(r'pressure|bp|ബിപി', text, re.IGNORECASE):
            return Intent(type="mark_taken", medication="BP medicine")
        if re.search(r'thyroid|തൈറോയ്ഡ്', text, re.IGNORECASE):
            return Intent(type="mark_taken", medication="thyroid medicine")
        if re.search(r'medicine|med|tablet|മരുന്ന്|ഗുളിക|ടാബ്ലറ്റ്', text, re.IGNORECASE):
            return Intent(type="mark_taken", medication="medicine")

    # PRIORITY 3: Vital recording
    vital_parsers = [parse_bp, parse_sugar, parse_spo2, parse_temperature, parse_heart_rate, parse_weight, parse_pain]
    for parser in vital_parsers:
        result = parser(text)
        if result:
            return Intent(type="add_vital", data=result)

    # PRIORITY 4: Reminder (but not stop/cancel/delete reminder)
    is_stopping = bool(re.search(r'stop|cancel|remove|delete|off|നിർത്ത', text, re.IGNORECASE))
    has_reminder_word = bool(re.search(r'remind|ഓർമ്മ|alarm|അലാറം|schedule|notify|alert|set.*time|timer', text, re.IGNORECASE))
    has_time_word = bool(re.search(r'\d\s*(am|pm|മണി)|morning|evening|night|afternoon|tomorrow|രാവിലെ|വൈകുന്നേരം|രാത്രി', text, re.IGNORECASE))
    has_action_word = bool(re.search(r'need to|have to|should|want to|must|വേണം|കഴിക്കണം|ചെയ്യണം|മറക്കരുത്', text, re.IGNORECASE))
    is_querying = bool(re.search(r'show|list|what.*reminder|my.*reminder|എന്റെ.*ഓർമ്മ', text, re.IGNORECASE))

    if (has_reminder_word or (has_time_word and has_action_word)) and not is_querying and not is_stopping:
        times = ["08:00"]
        # Parse time
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.IGNORECASE)
        hour_match = re.search(r'(\d{1,2})\s*(am|pm|മണി)', text, re.IGNORECASE)
        
        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2))
            if time_match.group(3) and 'pm' in time_match.group(3).lower() and h < 12:
                h += 12
            times = [f"{h:02d}:{m:02d}"]
        elif hour_match:
            h = int(hour_match.group(1))
            if 'pm' in hour_match.group(2).lower() and h < 12:
                h += 12
            times = [f"{h:02d}:00"]
        elif re.search(r'morning|രാവിലെ', text, re.IGNORECASE):
            times = ["08:00"]
        elif re.search(r'evening|വൈകുന്നേരം', text, re.IGNORECASE):
            times = ["18:00"]
        elif re.search(r'night|രാത്രി', text, re.IGNORECASE):
            times = ["21:00"]

        title = re.sub(r'remind.*me.*to|remind me|set.*remind|set.*alarm|ഓർമ്മിപ്പിക്ക|remind|reminder|alarm|please|at|need to|have to', '', text, flags=re.IGNORECASE).strip()
        if len(title) < 2:
            title = "Reminder"

        return Intent(type="set_reminder", data=ReminderData(
            medication=title, times=times, tts_message=f"{title} — സമയമായി"
        ))

    # PRIORITY 5: Stop medication/reminder
    if re.search(r'stop.*remind|cancel.*remind|remove.*remind|delete.*remind|റിമൈൻഡർ.*നിർത്ത|alarm.*off', text, re.IGNORECASE):
        med = next((m for m in MEDICATIONS if m.lower() in lower), "")
        return Intent(type="stop_reminder", medication=med)
    if re.search(r'stop|remove|discontinue|നിർത്ത|no more', text, re.IGNORECASE):
        med = next((m for m in MEDICATIONS if m.lower() in lower), None)
        if med or re.search(r'stop.*med|remove.*med|നിർത്ത.*മരുന്ന്|മരുന്ന്.*നിർത്ത', text, re.IGNORECASE):
            return Intent(type="stop_medication", medication=med or "")

    # PRIORITY 6: Lab results
    for pat, test_name, unit, ref_low, ref_high in LAB_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m and m.group(1):
            return Intent(type="add_lab_result", data=LabResultData(
                test_name=test_name, value=float(m.group(1)), unit=unit, ref_low=ref_low, ref_high=ref_high
            ))

    # PRIORITY 7: Medication addition
    dosage_match = re.search(r'(\d+\.?\d*)\s*(mg|ml|mcg|g|iu|units?|tablet|tab|cap)', text, re.IGNORECASE)
    med_match = next((m for m in MEDICATIONS if m.lower() in lower), None)
    if med_match and dosage_match:
        return Intent(type="add_medication", data=extract_medication(text, med_match))

    # PRIORITY 8: Condition
    for pat, name, icd in CONDITIONS:
        if re.search(pat, text, re.IGNORECASE) and re.search(r'have|diagnosed|i have|എനിക്ക്|ഉണ്ട്|ആണ്|രോഗം|disease|problem', text, re.IGNORECASE):
            # Skip if it reads like a symptom report rather than a condition declaration
            if re.search(r'headache|fever|pain|cough|cold|tired|dizzy|vomit|nausea|sore|burning|തലവേദന|പനി|ചുമ|വേദന|ക്ഷീണം', text, re.IGNORECASE):
                break  # Fall through to symptom detection
            severity = "severe" if re.search(r'severe|serious|ഗുരുതര', text, re.IGNORECASE) else "moderate"
            return Intent(type="add_condition", data=ConditionData(name=name, severity=severity, icd_code=icd))

    # PRIORITY 9: Queries
    if re.search(r'what.*medication|my.*med|medicine.*taking|മരുന്ന്.*എന്ത|show.*med|എന്റെ.*മരുന്ന്|medicine.*list', text, re.IGNORECASE):
        return Intent(type="query_medications")
    if re.search(r'what.*condition|my.*disease|health.*issue|രോഗ.*എന്ത|condition|എന്റെ.*രോഗ', text, re.IGNORECASE):
        return Intent(type="query_conditions")
    if re.search(r'my.*reminder|show.*reminder|list.*reminder|what.*reminder|എന്റെ.*ഓർമ്മ', text, re.IGNORECASE):
        return Intent(type="query_reminders")
    if re.search(r'vitals?|readings?|bp.*history|sugar.*history|my.*bp|my.*sugar|എന്റെ.*bp|ബിപി.*എത്ര|ഷുഗർ.*എത്ര', text, re.IGNORECASE):
        return Intent(type="query_vitals")
    if re.search(r'lab.*result|test.*result|report|ടെസ്റ്റ്.*റിസൾട്ട്|ലാബ്|blood.*test', text, re.IGNORECASE):
        return Intent(type="query_lab_results")
    if re.search(r'adherence|compliance|how.*regular|എത്ര.*കഴിച്ചു|track|മരുന്ന്.*മുടങ്ങി', text, re.IGNORECASE):
        return Intent(type="query_adherence")

    # PRIORITY 10: Symptoms
    if re.search(r'fever|cough|pain|headache|vomit|diarr|rash|breathing|dizz|nausea|tired|weak|swelling|itching|burning|cold|sore|cramp|നീർക്കെട്ട്|പനി|ചുമ|വേദന|ഛർദ്ദി|തലവേദന|ശ്വാസം|ക്ഷീണം|ചൊറിച്ചിൽ|വയറുവേദന|നെഞ്ചുവേദന|തലചുറ്റൽ|ഓക്കാനം|ജലദോഷം', text, re.IGNORECASE):
        return Intent(type="symptom_report", text=text)

    # DEFAULT: General chat
    return Intent(type="general_chat", text=text)
