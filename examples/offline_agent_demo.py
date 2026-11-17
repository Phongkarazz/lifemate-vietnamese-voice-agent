"""Zero-dependency offline showcase of the LifeMate ReAct + Guardrail pipeline.

Demonstrates how schema-driven guardrails, read-before-write ID verification,
and hybrid conditional/financial safety checks protect downstream tool execution
using purely synthetic mock environments (no competition data or GPU required).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import ReActSessionState, step_react_state  # noqa: E402
from src.router import SemanticToolRouter  # noqa: E402


class _SyntheticEnv:
    """In-memory deterministic sandbox simulating state transitions and hash tracking."""

    def __init__(self, initial_state: dict[str, Any], responses: dict[str, dict[str, Any]]):
        self.state = json.loads(json.dumps(initial_state))
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"name": name, "arguments": args})
        res = json.loads(json.dumps(self.responses.get(name, {"ok": True})))
        if name == "grocery.add_to_cart" and args.get("product_id") == "prod_demo_01":
            self.state.setdefault("cart", []).append(args)
        elif name == "reminders.create":
            self.state.setdefault("reminders", []).append(args)
        elif name == "bill_payment.pay" and res.get("ok"):
            self.state["paid"] = args
        return res

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return list(self.calls)

    def get_state_hash(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_catalog() -> dict[str, dict[str, Any]]:
    return {
        "weather.get_forecast": {
            "name": "weather.get_forecast",
            "kind": "read",
            "returns": "weather.forecast[]",
            "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}},
        },
        "reminders.create": {
            "name": "reminders.create",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "time": {"type": "string"}},
                "required": ["title"],
            },
        },
        "grocery.search": {
            "name": "grocery.search",
            "kind": "read",
            "returns": "grocery.products[]",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        "grocery.add_to_cart": {
            "name": "grocery.add_to_cart",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "x-ref": "grocery.products"},
                    "quantity": {"type": "integer", "minimum": 1},
                },
                "required": ["product_id", "quantity"],
            },
        },
        "ewallet.get_balance": {
            "name": "ewallet.get_balance",
            "kind": "read",
            "parameters": {"type": "object", "properties": {}},
        },
        "bill_payment.list_bills": {
            "name": "bill_payment.list_bills",
            "kind": "read",
            "returns": "bill_payment.bills[]",
            "parameters": {"type": "object", "properties": {}},
        },
        "bill_payment.pay": {
            "name": "bill_payment.pay",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_id": {"type": "string", "x-ref": "bill_payment.bills"},
                    "source": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["bill_id", "source"],
            },
        },
    }


def _run_scripted_scenario(
    *,
    sid: str,
    utterance: str,
    installed_apps: dict[str, Any],
    env: _SyntheticEnv,
    scripted_replies: list[str],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample = {
        "id": sid,
        "context": {
            "current_time": "2026-11-17T08:30:00+07:00",
            "installed_apps": installed_apps,
        },
        "history": [],
    }
    state = ReActSessionState(
        sample=sample,
        env=env,
        cleaned_user_text=utterance,
        cleaned_hist_transcripts={},
        catalog_tools_map=catalog,
        router=SemanticToolRouter(top_k_recommended=4),
        max_steps=6,
    )
    initial_hash = env.get_state_hash()
    for reply in scripted_replies:
        if state.finished:
            break
        step_react_state(state, reply)
    if not state.finished:
        state._finish()

    return {
        "id": sid,
        "utterance": utterance,
        "initial_hash": initial_hash,
        "final_hash": state.state_hash,
        "guardrail_retries": state.guardrail_retries,
        "executed_calls": state.tool_calls,
    }


def run_demo() -> list[dict[str, Any]]:
    """Run 3 synthetic showcase scenarios illustrating guardrails and hybrid safety."""
    catalog = _build_catalog()
    results = [
        _run_scripted_scenario(
            sid="demo_conditional_weather",
            utterance="Mai nếu có mưa thì đặt nhắc nhở mang áo mưa nhé. Còn trời không mưa thì cứ để nguyên.",
            installed_apps={
                "weather": {"tools": ["weather.get_forecast"]},
                "reminders": {"tools": ["reminders.create"]},
            },
            env=_SyntheticEnv(
                initial_state={"reminders": []},
                responses={"weather.get_forecast": {"ok": True, "forecast": [{"day": "tomorrow", "rain": False}]}},
            ),
            scripted_replies=[
                '<think>Check tomorrow forecast before deciding.</think>\n<tool_call>\n{"name": "weather.get_forecast", "arguments": {"days": 1}}\n</tool_call>',
                '<think>Forecast is rain=False, stop cleanly.</think>',
            ],
            catalog=catalog,
        ),
        _run_scripted_scenario(
            sid="demo_read_before_write_guardrail",
            utterance="Đặt giúp mình 2 hộp mã SP204 từ giỏ tạp hóa nha.",
            installed_apps={"grocery": {"tools": ["grocery.search", "grocery.add_to_cart"]}},
            env=_SyntheticEnv(
                initial_state={"cart": []},
                responses={"grocery.search": {"ok": True, "products": [{"id": "prod_demo_01", "name": "Hộp ngũ cốc"}]}},
            ),
            scripted_replies=[
                # Step 1 tries ungrounded spoken code SP204 -> intercepted by PureSchemaStateTracker!
                '<think>Try spoken product code directly.</think>\n<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "SP204", "quantity": 2}}\n</tool_call>',
                # Step 2 recovers after guardrail feedback by calling grocery.search first
                '<think>Guardrail blocked SP204; search catalog first.</think>\n<tool_call>\n{"name": "grocery.search", "arguments": {"query": "ngũ cốc"}}\n</tool_call>',
                # Step 3 commits with verified ID prod_demo_01
                '<think>Use grounded prod_demo_01.</think>\n<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "prod_demo_01", "quantity": 2}}\n</tool_call>',
                "<think>Completed safely.</think>",
            ],
            catalog=catalog,
        ),
        _run_scripted_scenario(
            sid="demo_financial_balance_protection",
            utterance="Thanh toán hóa đơn điện bằng ví, nếu ví không đủ tiền thì chuyển qua ngân hàng giúp tôi.",
            installed_apps={
                "ewallet": {"tools": ["ewallet.get_balance"]},
                "bill_payment": {"tools": ["bill_payment.list_bills", "bill_payment.pay"]},
            },
            env=_SyntheticEnv(
                initial_state={"paid": None},
                responses={
                    "ewallet.get_balance": {"ok": True, "balance": 120000},
                    "bill_payment.list_bills": {"ok": True, "bills": [{"id": "bill_elec_01", "amount": 450000}]},
                    "bill_payment.pay": {"ok": True},
                },
            ),
            scripted_replies=[
                '<think>Check bills first.</think>\n<tool_call>\n{"name": "bill_payment.list_bills", "arguments": {}}\n</tool_call>',
                '<think>Check ewallet balance.</think>\n<tool_call>\n{"name": "ewallet.get_balance", "arguments": {}}\n</tool_call>',
                # LLM proposes ewallet even though 120,000 < 450,000 -> Hybrid engine redirects to bank!
                '<think>Pay bill.</think>\n<tool_call>\n{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_elec_01", "source": "ewallet", "amount": 450000}}\n</tool_call>',
                "<think>Done.</think>",
            ],
            catalog=catalog,
        ),
    ]
    return results


if __name__ == "__main__":
    summary = run_demo()
    print("=== LifeMate Offline Agent & Guardrail Showcase ===")
    for item in summary:
        print(json.dumps(item, ensure_ascii=False, indent=2))
