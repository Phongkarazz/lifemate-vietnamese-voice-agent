"""Regression tests for the LifeMate Control Tower Streamlit MVP backend."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from demo.scenarios import (
    SCENARIO_CATALOG,
    analyze_vietnamese_utterance,
    get_architecture_layers_vi,
    get_v9_evolution_rows,
    run_audit_suite_vi,
    run_fintech_chaos_matrix,
    run_sandbox_scenario,
    run_workbench_turn,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStreamlitMVPBackend(unittest.TestCase):
    def test_app_py_compiles_cleanly(self) -> None:
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        ast.parse(src, filename="app.py")
        compile(src, "app.py", "exec")

    def test_sandbox_scenarios_guardrail_on_vs_off(self) -> None:
        res_off = run_sandbox_scenario("fintech_bill_fallback", guardrail_enabled=False, wallet_balance=120000)
        res_on = run_sandbox_scenario("fintech_bill_fallback", guardrail_enabled=True, wallet_balance=120000)
        self.assertGreater(res_off["soft_noops"], 0)
        self.assertEqual(res_on["soft_noops"], 0)
        self.assertEqual(res_on["final_state"]["bills"][0]["status"], "paid_via_bank")

        food_off = run_sandbox_scenario("v8_food_menu_dependency", guardrail_enabled=False)
        food_on = run_sandbox_scenario("v8_food_menu_dependency", guardrail_enabled=True)
        self.assertGreater(food_off["soft_noops"], 0)
        self.assertEqual(food_on["soft_noops"], 0)
        self.assertEqual([c["name"] for c in food_on["executed_calls"]], ["food_delivery.list_restaurants", "food_delivery.get_menu", "food_delivery.place_order"])

    def test_all_scenarios_and_chaos_matrix_pass(self) -> None:
        for sid in SCENARIO_CATALOG:
            r = run_sandbox_scenario(sid, guardrail_enabled=True)
            self.assertEqual(r["soft_noops"], 0, f"{sid} had soft_noops under guardrail!")

        matrix = run_fintech_chaos_matrix()
        self.assertEqual(len(matrix), 7)
        self.assertTrue(all(row["Safe"] for row in matrix))

        evo_rows = get_v9_evolution_rows()
        self.assertGreaterEqual(len(evo_rows), 6)
        self.assertIn("Kiến trúc Schema-First", evo_rows[0])

    def test_analyze_vietnamese_utterance(self) -> None:
        out = analyze_vietnamese_utterance("Chuyển hai triệu năm trăm nghìn sang ví lúc 8 rưỡi tối mai", top_k=4)
        self.assertEqual(out["parsed_number"], 2500000)
        self.assertEqual(out["monetary_amount"], 2500000)
        self.assertIsNotNone(out["resolved_iso"])
        self.assertFalse(out["is_cancellation"])
        self.assertGreater(out["char_savings_pct"], 0.0)

        food_out = analyze_vietnamese_utterance("Đặt cho mình 1 tô phở tái nạm ở quán Phở Thìn Sài Gòn nhé.", top_k=4)
        self.assertIsNone(food_out["monetary_amount"])
        self.assertEqual(food_out["recommended_apps"][0], "food_delivery")

    def test_workbench_turn_and_audit_suite_vi(self) -> None:
        # 1. Preset turn with V9 ON vs OFF
        on_res = run_workbench_turn(
            SCENARIO_CATALOG["fintech_bill_fallback"]["utterance"],
            guardrail_enabled=True,
            wallet_balance=120_000,
        )
        off_res = run_workbench_turn(
            SCENARIO_CATALOG["fintech_bill_fallback"]["utterance"],
            guardrail_enabled=False,
            wallet_balance=120_000,
        )
        self.assertEqual(on_res["mode"], "preset")
        self.assertEqual(on_res["receipt_card"]["kind"], "warn")
        self.assertEqual(off_res["receipt_card"]["kind"], "err")
        self.assertGreater(len(on_res["steps"]), 0)

        # 2. Custom Vietnamese utterance flows end-to-end (anchored deterministic check)
        custom_res = run_workbench_turn(
            "Chuyển hai triệu năm trăm nghìn cho chị Lan lúc 8 giờ rưỡi tối mai nhé",
            guardrail_enabled=True,
            current_time_iso="2026-11-17T09:00:00+07:00",
        )
        self.assertEqual(custom_res["mode"], "custom")
        self.assertIn(custom_res["nlp"]["recommended_apps"][0], ("ewallet", "banking"))
        self.assertEqual(custom_res["nlp"]["monetary_amount"], 2500000)
        self.assertEqual(custom_res["nlp"]["resolved_iso"], "2026-11-18T20:30:00+07:00")
        self.assertGreaterEqual(len(custom_res["steps"]), 1)

        # 2b. Spoken half-hour without 'giờ' ('sáu rưỡi ngày mai') + user phở order
        pho_res = run_workbench_turn(
            "Đặt cho mình 1 tô phở tái nạm ở quán Phở Thìn Sài Gòn nhé, giá 20000, sáu rưỡi ngày mai",
            guardrail_enabled=True,
            current_time_iso="2026-11-22T14:28:00+07:00",
        )
        self.assertEqual(pho_res["nlp"]["recommended_apps"][0], "food_delivery")
        self.assertEqual(pho_res["nlp"]["monetary_amount"], 20000)
        self.assertEqual(pho_res["nlp"]["resolved_iso"], "2026-11-23T06:30:00+07:00")

        # 2c. Default live Vietnam wall-clock resolves 'ngày mai' relative to real today (UTC+7)
        from datetime import datetime, timedelta, timezone

        expected_tomorrow = (datetime.now(timezone(timedelta(hours=7))).date() + timedelta(days=1)).strftime("%Y-%m-%d")
        live_res = run_workbench_turn("Đặt lịch họp sáu rưỡi ngày mai", guardrail_enabled=True)
        self.assertEqual(live_res["nlp"]["resolved_iso"], f"{expected_tomorrow}T06:30:00+07:00")

        # 3. Clean Vietnamese audit suite passes 7/7 with zero English dev jargon
        audit = run_audit_suite_vi()
        self.assertEqual(audit["passed"], 7)
        self.assertEqual(audit["total"], 7)
        joined = " ".join(r["title"] + " " + r["risk"] + " " + r["v9_handling"] for r in audit["rows"])
        for jargon in ("SOFT_NOOP", "done=False", "Premature Regex Cancel", "Aborts (0 calls)", "May confuse hotel_id"):
            self.assertNotIn(jargon, joined)

        self.assertEqual(len(get_architecture_layers_vi()), 5)

        # 4. Zero vibe-coding 'Lá chắn V9' / 'Bảo vệ V9' / 'v9.0' copy in UI or outputs
        app_src_l = (ROOT / "app.py").read_text(encoding="utf-8").lower()
        sample_outputs = " ".join(
            [
                on_res["assistant_reply"],
                custom_res["assistant_reply"],
                custom_res["receipt_card"]["badge"],
                joined,
            ]
        ).lower()
        for banned in ("lá chắn v9", "bảo vệ v9", "phân tích trực tiếp v9", ">v9.0<", "(không tham số)", "lm-delta", "lm-rail-status"):
            self.assertNotIn(banned, app_src_l)
            self.assertNotIn(banned, sample_outputs)

        for step in pho_res["steps"]:
            self.assertNotIn("UNVERIFIED_INPUT", str(step.get("args", {})))


if __name__ == "__main__":
    unittest.main()
