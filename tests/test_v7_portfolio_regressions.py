"""Regression tests for V7 inference fixes and the standalone portfolio offline demo."""
from __future__ import annotations

import unittest

from examples.offline_agent_demo import run_demo
from src.agent import ReActSessionState, step_react_state
from src.guardrail import PureSchemaStateTracker
from src.hybrid_engine import HybridProgramAidedEngine


class _MiniEnv:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def validate(self, name: str, args: dict) -> dict:
        return {"ok": True}

    def call(self, name: str, args: dict) -> dict:
        self.calls.append({"name": name, "arguments": args})
        if name == "parking.find":
            return {"ok": True, "lots": [{"id": "lot_01", "available_spots": 5}]}
        return {"ok": True}

    def get_tool_calls(self) -> list[dict]:
        return list(self.calls)

    def get_state_hash(self) -> str:
        return f"calls:{len(self.calls)}"


class TestV7PortfolioRegressions(unittest.TestCase):
    def test_conditional_weather_utterance_not_cancelled_at_step_zero(self) -> None:
        text = "Mai nếu có mưa thì đặt nhắc nhở nhé. Còn trời không mưa rồi thì cứ để nguyên."
        self.assertFalse(HybridProgramAidedEngine.is_complete_cancellation(text))
        self.assertTrue(HybridProgramAidedEngine.is_complete_cancellation("Thôi đừng đặt nữa, à mà thôi."))

    def test_spoken_alphanumeric_codes_do_not_bypass_read_before_write(self) -> None:
        catalog = {
            "ecommerce.order_status": {
                "name": "ecommerce.order_status",
                "kind": "read",
                "returns": "ecommerce.orders[]",
                "parameters": {"type": "object", "properties": {}},
            },
            "ecommerce.request_return": {
                "name": "ecommerce.request_return",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "x-ref": "ecommerce.orders"}},
                    "required": ["order_id"],
                },
            },
        }
        sample = {
            "id": "v7_reg_spoken",
            "context": {
                "installed_apps": {
                    "ecommerce": {"tools": ["ecommerce.order_status", "ecommerce.request_return"]}
                }
            },
        }
        tracker = PureSchemaStateTracker(sample, "Kiểm tra đơn V21 mã RV157 tàu SE7 phòng B05", catalog)
        self.assertEqual(len(tracker.all_known_tokens), 0)
        self.assertEqual(len(tracker.user_explicit_id_tokens), 0)
        self.assertIsNotNone(tracker.check_pre_execution("ecommerce.request_return", {"order_id": "V21"}))

    def test_parking_reserve_not_suppressed_after_ride_hailing_bike(self) -> None:
        catalog = {
            "ride_hailing.book": {"name": "ride_hailing.book", "kind": "write", "parameters": {"type": "object", "properties": {}}},
            "parking.find": {"name": "parking.find", "kind": "read", "returns": "parking.lots[]", "parameters": {"type": "object", "properties": {}}},
            "parking.reserve": {
                "name": "parking.reserve",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"lot_id": {"type": "string", "x-ref": "parking.lots"}},
                    "required": ["lot_id"],
                },
            },
        }
        sample = {
            "id": "v7_reg_parking",
            "context": {
                "current_time": "2026-11-16T08:00:00+07:00",
                "installed_apps": {
                    "ride_hailing": {"tools": ["ride_hailing.book"]},
                    "parking": {"tools": ["parking.find", "parking.reserve"]},
                },
            },
            "history": [],
        }
        env = _MiniEnv()
        state = ReActSessionState(
            sample=sample,
            env=env,
            cleaned_user_text="Đặt xe ôm tới bãi xe rồi giữ một chỗ đỗ ở bãi A.",
            cleaned_hist_transcripts={},
            catalog_tools_map=catalog,
        )
        step_react_state(state, '<tool_call>\n{"name": "ride_hailing.book", "arguments": {"vehicle": "bike"}}\n</tool_call>')
        step_react_state(state, '<tool_call>\n{"name": "parking.find", "arguments": {}}\n</tool_call>')
        step_react_state(state, '<tool_call>\n{"name": "parking.reserve", "arguments": {"lot_id": "lot_01"}}\n</tool_call>')
        self.assertEqual([c["name"] for c in env.calls], ["ride_hailing.book", "parking.find", "parking.reserve"])

    def test_offline_portfolio_demo_runs_cleanly(self) -> None:
        results = run_demo()
        self.assertEqual(len(results), 3)
        by_id = {r["id"]: r for r in results}

        # Scenario 1: weather checked, rain=False -> stops without creating reminder
        self.assertEqual([c["name"] for c in by_id["demo_conditional_weather"]["executed_calls"]], ["weather.get_forecast"])
        self.assertEqual(by_id["demo_conditional_weather"]["initial_hash"], by_id["demo_conditional_weather"]["final_hash"])

        # Scenario 2: ungrounded spoken ID SP204 intercepted once, then resolved via search -> add_to_cart
        self.assertEqual(by_id["demo_read_before_write_guardrail"]["guardrail_retries"], 1)
        self.assertEqual(
            [c["name"] for c in by_id["demo_read_before_write_guardrail"]["executed_calls"]],
            ["grocery.search", "grocery.add_to_cart"],
        )

        # Scenario 3: low e-wallet balance (120k < 450k) automatically routed to bank source
        pay_calls = [c for c in by_id["demo_financial_balance_protection"]["executed_calls"] if c["name"] == "bill_payment.pay"]
        self.assertEqual(len(pay_calls), 1)
        self.assertEqual(pay_calls[0]["arguments"]["source"], "bank")

    def test_v8_blocks_place_order_before_get_menu(self) -> None:
        catalog = {
            "food_delivery.list_restaurants": {"name": "food_delivery.list_restaurants", "kind": "read", "returns": "food_delivery.restaurants[]", "parameters": {"type": "object", "properties": {}}},
            "food_delivery.get_menu": {"name": "food_delivery.get_menu", "kind": "read", "returns": "food_delivery.menu[]", "parameters": {"type": "object", "properties": {"restaurant_id": {"type": "string", "x-ref": "food_delivery.restaurants"}}}},
            "food_delivery.place_order": {"name": "food_delivery.place_order", "kind": "write", "parameters": {"type": "object", "properties": {"restaurant_id": {"type": "string", "x-ref": "food_delivery.restaurants"}, "items": {"type": "array"}}, "required": ["restaurant_id", "items"]}},
        }
        sample = {"id": "v8_food", "context": {"installed_apps": {"food_delivery": {"tools": list(catalog.keys())}}}}
        tracker = PureSchemaStateTracker(sample, "Đặt phở", catalog)
        tracker.record_observation("food_delivery.list_restaurants", {}, {"ok": True, "restaurants": [{"id": "rest_01", "name": "Phở Thìn"}]})

        err = tracker.check_pre_execution("food_delivery.place_order", {"restaurant_id": "rest_01", "items": [{"item_id": "pho_1", "quantity": 1}]})
        self.assertIsNotNone(err)
        self.assertIn("food_delivery.get_menu", err)

        tracker.record_observation("food_delivery.get_menu", {"restaurant_id": "rest_01"}, {"ok": True, "menu": [{"id": "dish_01", "name": "Phở tái"}]})
        self.assertIsNone(tracker.check_pre_execution("food_delivery.place_order", {"restaurant_id": "rest_01", "items": [{"item_id": "dish_01", "quantity": 1}]}))

    def test_v8_blocks_request_return_on_ineligible_status(self) -> None:
        catalog = {
            "ecommerce.order_status": {"name": "ecommerce.order_status", "kind": "read", "returns": "ecommerce.orders[]", "parameters": {"type": "object", "properties": {}}},
            "ecommerce.request_return": {"name": "ecommerce.request_return", "kind": "write", "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "x-ref": "ecommerce.orders"}}, "required": ["order_id"]}},
        }
        sample = {"id": "v8_ret", "context": {"installed_apps": {"ecommerce": {"tools": list(catalog.keys())}}}}
        tracker = PureSchemaStateTracker(sample, "Trả đơn hàng", catalog)
        tracker.record_observation("ecommerce.order_status", {}, {"ok": True, "orders": [{"id": "ord_bad", "status": "cancelled"}, {"id": "ord_ok", "status": "delivered"}]})

        self.assertIsNotNone(tracker.check_pre_execution("ecommerce.request_return", {"order_id": "ord_bad"}))
        self.assertIsNone(tracker.check_pre_execution("ecommerce.request_return", {"order_id": "ord_ok"}))


if __name__ == "__main__":
    unittest.main()
