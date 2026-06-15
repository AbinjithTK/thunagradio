---
title: Thuna (തുണ) — AI Health Companion
emoji: 🤝
colorFrom: green
colorTo: green
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: true
license: mit
tags:
  - backyard-ai
  - tiny-titan
  - off-the-grid
  - field-notes
  - best-demo
  - best-agent
short_description: Offline AI health companion for elderly, Gemma 4 2B
---

# 🤝 Thuna (തുണ) — Offline AI Health Companion for Elderly

> *"I can't always be there for my grandmothers. But Thuna can."*

**Thuna** is a voice-first AI health companion that runs 100% offline on Android, built for elderly people in rural India who can't read, can't type, and have no reliable internet.

## 👤 Team

- **HF Username:** [AbinjithTK](https://huggingface.co/AbinjithTK)

## 🎬 Demo Video

[![Demo Video](https://img.youtube.com/vi/m4DD0gsQ_FU/0.jpg)](https://youtu.be/m4DD0gsQ_FU)

**Watch:** https://youtu.be/m4DD0gsQ_FU

## 🗣️ Social Post

**X/Twitter:** https://x.com/ABI_N_JITH/status/XXXXXXXX

<!-- ⚠️ REPLACE the link above with your actual tweet URL after posting -->

## 🎯 The Problem

My grandmother has diabetes. Last month she took **double** her medication for 30 days because the pharmacy changed her tablet strength. Nobody noticed until she started fainting.

My other grandmother has polio. She can't go to hospitals. She can't read English. She gets health advice from neighbors — often wrong, sometimes dangerous.

**240 million elderly people** globally manage chronic conditions alone — without internet, without English, without anyone to ask "did I take my medicine today?"

## 🛠️ What Thuna Does

| Feature | How |
|---------|-----|
| **Malayalam Voice Interaction** | Speak naturally, hear responses. No typing needed. |
| **Medication Safety** | Tracks exact dosages, detects 15 drug interactions, prevents overdoses |
| **Health Monitoring** | "BP 140/90" → saved, trend analyzed, alerts on danger |
| **Prescription OCR** | Photo → Gemma 4 vision extracts all medications |
| **Companion AI** | Fights loneliness. Talks about movies, cooking, family. |
| **Health Reports** | One tap: full timeline for doctor visits |
| **Family Status** | Daily update shared to family |

## 🧠 Architecture

```
Malayalam Voice Input
    → IntentParser (Regex, <1ms, 100% reliable)
    → AgentEngine (23 tools, DB operations)
    → Gemma 4 E2B (warm Malayalam response generation)
    → TTS Output
```

**Why hybrid?** My grandmother's overdose happened because of confusion. A regex will **NEVER** confuse "2mg twice daily" with "two 2mg tablets." For health-critical data, reliability beats flexibility. Gemma 4 handles the warmth; regex handles the precision.

## 🏔️ Model: Gemma 4 E2B (2B params)

- **INT4 quantized** via Cactus v1.7 — fits in ~1.5GB RAM
- **~2 second inference** on mid-range Android
- **100% offline** after one-time download
- **Malayalam personality** — warm companion, not clinical bot
- **Multimodal** — prescription photo → structured medication extraction

## 🎖️ Badges Targeted

| Badge | Qualification |
|-------|--------------|
| 🐜 **Tiny Titan** | Gemma 4 E2B = 2B parameters (well under 4B cap) |
| 🔌 **Off the Grid** | Runs 100% on-device, zero cloud APIs in production |
| 🤖 **Best Agent** | 23-tool multi-step agent with deterministic planning |
| 🎬 **Best Demo** | Real grandmother testing + personal story |
| 📓 **Field Notes** | This writeup documents the build process |

## 📱 Full Mobile App

This Gradio Space is a web demo of Thuna's agent engine. The full experience runs on Android with voice:

- **APK Download:** [Google Drive](https://drive.google.com/drive/folders/17SjktvnP5EXh-kDmtXxRWF73h8EdzWrG?usp=sharing)
- **Source Code:** [GitHub](https://github.com/AbinjithTK/Thuna)

## 💡 How to Use This Demo

1. Click **"Load Demo Data"** to set up a sample patient (diabetic grandmother)
2. Try typing:
   - `BP 145/92` — records blood pressure with trend analysis
   - `sugar 200` — records blood sugar, triggers alert if dangerous
   - `took metformin` — logs medication adherence
   - `my medicines` — lists all active medications
   - `remind me to take medicine at 9pm` — sets a reminder
   - `I have headache and fever` — symptom recording
   - Any casual chat in Malayalam!
3. Check the **Dashboard** tab for real-time health overview
4. Generate **Reports** for doctor visits or family sharing

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Gemma 4 E2B (2B, INT4) via Cactus v1.7 |
| **Intent Parser** | Deterministic regex (Malayalam + English, <1ms) |
| **Agent Engine** | 23 tools — vitals, medications, reminders, drug interactions |
| **Drug Safety** | 15 interaction pairs, duplicate therapy detection |
| **Health Alerts** | Compound risk correlation (diabetes+sugar+BP, SpO2+fever) |
| **Proactive Checks** | Missed dose detection, medication expiry, follow-up reminders |
| **UI** | Gradio 5 (web) / React Native 0.82 (mobile) |
| **Database** | In-memory (web demo) / WatermelonDB + openEHR (mobile) |
| **Voice** | Browser mic + Whisper (web) / Cactus Whisper + Android STT (mobile) |

## 📓 Field Notes — What I Learned

**Started with Flutter, switched to React Native.** Cactus SDK's Flutter binding was a stub — no real native inference. React Native had full Nitro Modules support with C++ bridge. Rewrote the entire app in 3 days.

**Gemma 4 E2B outputs Manglish.** The 2B model often writes Malayalam in English letters. Built a post-processing filter that detects Latin-heavy output and replaces with proper Malayalam. Added explicit grammar rules with examples in the system prompt.

**Health-critical data needs deterministic parsing.** An LLM might confuse "2mg twice daily" with "two 2mg tablets." For medication dosages, a regex that never hallucinates is safer than a model that's right 99% of the time. The 1% is what caused my grandmother's overdose.

**Proactive alerts polluted casual chat.** When grandmother asked about a movie, the AI responded with health reminders. Fixed by only injecting proactive alerts for health-related intents.

**Real testing with grandmother.** When she said "ഷുഗർ ഗുളിക കഴിച്ചു" (took sugar tablet) and Thuna confirmed in Malayalam, she understood immediately. That's the moment I knew it worked.

## 🌍 Impact

Built and tested with my actual grandmother in Poothampara, a rural village in Kerala, India. 240 million elderly people manage chronic conditions alone. Thuna works where internet doesn't — because the AI lives on their phone.

---

*Made with ❤️ in Kerala for grandmothers everywhere.*
