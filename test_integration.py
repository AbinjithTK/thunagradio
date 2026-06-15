"""Integration test — validates the full agent pipeline."""
import sys
sys.path.insert(0, '.')

from intent_parser import parse_intent
from health_store import HealthStore
from agent_engine import run_agent

def test_full_pipeline():
    store = HealthStore()
    print("=" * 60)
    print("THUNA INTEGRATION TEST")
    print("=" * 60)

    # 1. Add condition
    r = run_agent("I have diabetes", store)
    assert r.intent_type == "add_condition", f"Expected add_condition, got {r.intent_type}"
    assert r.badge == "🏥"
    print("✓ Add condition: diabetes")

    # 2. Add medication
    r = run_agent("metformin 500mg twice daily after food", store)
    assert r.intent_type == "add_medication", f"Expected add_medication, got {r.intent_type}"
    assert r.badge == "💊 ⏰"
    assert len(r.tools_executed) >= 2
    print("✓ Add medication: metformin 500mg")

    # 3. Add BP vital
    r = run_agent("BP 145/92", store)
    assert r.intent_type == "add_vital", f"Expected add_vital, got {r.intent_type}"
    assert r.badge == "❤️"
    assert r.alert and "ഉയർന്നത" in r.alert  # High BP alert in Malayalam
    print(f"✓ Add vital (BP 145/92) — Alert: {r.alert[:30]}...")

    # 4. Add sugar vital (high - compound risk)
    r = run_agent("sugar 250", store)
    assert r.intent_type == "add_vital"
    assert r.alert  # Should trigger alert (diabetes + high sugar)
    print(f"✓ Add vital (sugar 250) — Alert: {r.alert[:30]}...")

    # 5. Mark taken
    r = run_agent("took metformin", store)
    assert r.intent_type == "mark_taken"
    assert r.badge == "✅"
    print("✓ Mark taken: metformin")

    # 6. Query medications
    r = run_agent("my medicines", store)
    assert r.intent_type == "query_medications"
    assert "metformin" in r.context_for_llm.lower() or "Metformin" in r.context_for_llm
    print("✓ Query medications")

    # 7. Query today's doses
    r = run_agent("did I take medicine today", store)
    assert r.intent_type == "query_today_doses"
    print("✓ Query today's doses")

    # 8. Set reminder
    r = run_agent("remind me at 9pm to take medicine", store)
    assert r.intent_type == "set_reminder"
    assert r.badge == "⏰"
    print("✓ Set reminder")

    # 9. Symptom report
    r = run_agent("I have headache and fever", store)
    assert r.intent_type == "symptom_report"
    assert r.badge == "🤒"
    print("✓ Symptom report")

    # 10. General chat (Malayalam)
    r = run_agent("ബോറടിക്കുന്നു", store)
    assert r.intent_type == "general_chat"
    print("✓ General chat (Malayalam)")

    # 11. Drug interaction
    r = run_agent("glimepiride 2mg once daily morning", store)
    assert r.intent_type == "add_medication"
    assert r.alert and "hypoglycemia" in r.alert.lower() or "Monitor" in str(r.alert)
    print(f"✓ Drug interaction detected: {r.alert[:50]}...")

    # 12. Lab result
    r = run_agent("HbA1c 7.2", store)
    assert r.intent_type == "add_lab_result"
    assert r.badge == "🧪"
    print("✓ Lab result: HbA1c 7.2")

    # 13. Health report
    report = store.generate_health_report("Test Patient")
    assert "THUNA HEALTH REPORT" in report
    assert "Metformin" in report
    assert "Diabetes" in report
    print("✓ Health report generated")

    # 14. Family status
    status = store.generate_family_status("Test Patient")
    assert "Test Patient" in status
    print("✓ Family status generated")

    # 15. Adherence
    adherence = store.get_adherence_rate()
    assert "rate" in adherence
    print(f"✓ Adherence rate: {adherence['rate']}%")

    # 16. Stop medication
    r = run_agent("stop metformin", store)
    assert r.intent_type == "stop_medication"
    assert any(t['success'] for t in r.tools_executed)
    print("✓ Stop medication")

    # Verify it's stopped
    active = store.get_active_medications()
    assert not any(m.name == "Metformin" for m in active)
    print("✓ Metformin confirmed deactivated")

    print()
    print("=" * 60)
    print("ALL 16 TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_full_pipeline()
