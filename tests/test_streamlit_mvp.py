"""Regression tests for the LifeMate Control Tower Streamlit MVP backend."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

from demo.scenarios import (
    SCENARIO_CATALOG,
    analyze_vietnamese_utterance,
    get_v9_evolution_rows,
    run_fintech_chaos_matrix,
    run_sandbox_scenario,
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
        self.assertIn("LifeMate V9", evo_rows[0])

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


if __name__ == "__main__":
    unittest.main()
