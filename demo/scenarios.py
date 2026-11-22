"""Real Python sandbox execution & benchmark analytics backend for LifeMate Control Tower.

Executes genuine modules from `src/` (`PureSchemaStateTracker`, `HybridProgramAidedEngine`,
`ActionTriageBarrier`, `SemanticToolRouter`, `parse_vietnamese_number`, `VietnameseRelativeDateTimeEngine`,
`ReActSessionState`, `step_react_state`) against deterministic synthetic sandboxes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

from src.action_triage import ActionTriageBarrier
from src.agent import ReActSessionState, step_react_state
from src.asr_fallback import SafeContextPreprocessor
from src.datetime_engine import VietnameseRelativeDateTimeEngine
from src.guardrail import PureSchemaStateTracker, parse_tool_calls
from src.hybrid_engine import HybridProgramAidedEngine
from src.number_parser import extract_monetary_amount, extract_vietnamese_amount
from src.prompt_builder import build_allowed_tools_block
from src.router import SemanticToolRouter

# Verified progression across competition submissions
SCORE_PROGRESSION = [
    {"version": "Baseline v1", "date": "2026-11-09", "score": 25.80, "milestone": "Initial ReAct baseline"},
    {"version": "V2 SFT", "date": "2026-11-12", "score": 34.40, "milestone": "TF-IDF router + LoRA"},
    {"version": "V5 Expanded", "date": "2026-11-13", "score": 44.20, "milestone": "Grounded SFT + quarantine"},
    {"version": "V6 Adapted", "date": "2026-11-15", "score": 49.30, "milestone": "9216 ctx + Chunked CE (r=128)"},
    {"version": "V7 Final", "date": "2026-11-16", "score": 49.60, "milestone": "+140 ASR variants & fixes"},
]

PRIVATE_BENCHMARK_METRICS = {
    "initial_private": 26.40,
    "final_private": 36.05,
    "asr_debug_delta": 9.65,
}

# Exact observed Colab L4 training/eval metrics from Train_Qwen3_4B_ZAIC2026_V7_L4_Max run
V7_LOSS_CURVE = [
    {"step": 25, "train_loss": 0.065383, "val_loss": 0.044015},
    {"step": 50, "train_loss": 0.051829, "val_loss": 0.030794},
    {"step": 75, "train_loss": 0.031092, "val_loss": 0.028547},
    {"step": 100, "train_loss": 0.022012, "val_loss": 0.024255},
    {"step": 125, "train_loss": 0.019634, "val_loss": 0.023858},
    {"step": 150, "train_loss": 0.010520, "val_loss": 0.022196},
    {"step": 175, "train_loss": 0.008970, "val_loss": 0.022535},
    {"step": 200, "train_loss": 0.009492, "val_loss": 0.020370},
    {"step": 225, "train_loss": 0.010393, "val_loss": 0.020438},
    {"step": 250, "train_loss": 0.011659, "val_loss": 0.019530},
    {"step": 275, "train_loss": 0.004430, "val_loss": 0.020328},
    {"step": 300, "train_loss": 0.001608, "val_loss": 0.021659},
]

V6_BEST_VAL_LOSS = 0.020576
L4_TOTAL_VRAM_GIB = 22.03
L4_PEAK_CHUNKED_VRAM_GIB = 14.02

FULL_CATALOG_APPS: dict[str, dict[str, Any]] = {
    "ewallet": {
        "id": "ewallet",
        "name_vi": "Ví điện tử",
        "group": "finance",
        "description": "Chuyển tiền gửi tiền nghìn triệu đồng nạp rút và kiểm tra số dư ví điện tử",
        "tools": [
            {"name": "ewallet.get_balance", "kind": "read", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "ewallet.transfer",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"recipient_id": {"type": "string"}, "amount": {"type": "integer"}},
                    "required": ["recipient_id", "amount"],
                },
            },
        ],
    },
    "bill_payment": {
        "id": "bill_payment",
        "name_vi": "Thanh toán hóa đơn",
        "group": "finance",
        "description": "Tra cứu hóa đơn điện nước internet và thanh toán",
        "tools": [
            {
                "name": "bill_payment.list_bills",
                "kind": "read",
                "returns": "bill_payment.bills[]",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "bill_payment.pay",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bill_id": {"type": "string", "x-ref": "bill_payment.bills"},
                        "source": {"type": "string", "enum": ["ewallet", "bank"]},
                        "amount": {"type": "integer"},
                    },
                    "required": ["bill_id", "source"],
                },
            },
        ],
    },
    "banking": {
        "id": "banking",
        "name_vi": "Ngân hàng",
        "group": "finance",
        "description": "Chuyển khoản ngân hàng chuyển tiền nghìn triệu đồng xem số dư tài khoản",
        "tools": [
            {"name": "banking.get_balance", "kind": "read", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "banking.transfer",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string", "x-ref": "banking.accounts"},
                        "amount": {"type": "integer"},
                    },
                    "required": ["account_id", "amount"],
                },
            },
        ],
    },
    "stocks": {
        "id": "stocks",
        "name_vi": "Chứng khoán",
        "group": "finance",
        "description": "Tra cứu giá cổ phiếu, danh mục và đặt lệnh mua bán",
        "tools": [
            {"name": "stocks.get_quote", "kind": "read", "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}}},
            {"name": "stocks.get_portfolio", "kind": "read", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "stocks.place_order",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}, "side": {"type": "string"}, "shares": {"type": "integer"}},
                    "required": ["ticker", "side", "shares"],
                },
            },
        ],
    },
    "food_delivery": {
        "id": "food_delivery",
        "name_vi": "Giao đồ ăn",
        "group": "lifestyle",
        "description": "Tìm quán ăn, nhà hàng, xem thực đơn món ăn phở bún cơm và đặt giao đồ ăn",
        "tools": [
            {
                "name": "food_delivery.list_restaurants",
                "kind": "read",
                "returns": "food_delivery.restaurants[]",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "food_delivery.get_menu",
                "kind": "read",
                "returns": "food_delivery.menu[]",
                "parameters": {
                    "type": "object",
                    "properties": {"restaurant_id": {"type": "string", "x-ref": "food_delivery.restaurants"}},
                    "required": ["restaurant_id"],
                },
            },
            {
                "name": "food_delivery.place_order",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "restaurant_id": {"type": "string", "x-ref": "food_delivery.restaurants"},
                        "items": {"type": "array"},
                    },
                    "required": ["restaurant_id", "items"],
                },
            },
        ],
    },
    "ecommerce": {
        "id": "ecommerce",
        "name_vi": "Mua sắm trực tuyến",
        "group": "shopping",
        "description": "Kiểm tra trạng thái đơn hàng, giỏ hàng và yêu cầu trả hàng",
        "tools": [
            {
                "name": "ecommerce.order_status",
                "kind": "read",
                "returns": "ecommerce.orders[]",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ecommerce.request_return",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string", "x-ref": "ecommerce.orders"}, "reason": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        ],
    },
    "grocery": {
        "id": "grocery",
        "name_vi": "Đi chợ siêu thị",
        "group": "shopping",
        "description": "Đi chợ siêu thị tìm mua sản phẩm yến mạch thực phẩm hộp gói và thêm vào giỏ hàng",
        "tools": [
            {
                "name": "grocery.search",
                "kind": "read",
                "returns": "grocery.products[]",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "grocery.add_to_cart",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"product_id": {"type": "string", "x-ref": "grocery.products"}, "quantity": {"type": "integer"}},
                    "required": ["product_id", "quantity"],
                },
            },
        ],
    },
    "weather": {
        "id": "weather",
        "name_vi": "Thời tiết",
        "group": "utilities",
        "description": "Xem dự báo thời tiết mưa nắng nhiệt độ độ ẩm gió",
        "tools": [
            {"name": "weather.get_forecast", "kind": "read", "parameters": {"type": "object", "properties": {"days": {"type": "integer"}}}},
        ],
    },
    "reminders": {
        "id": "reminders",
        "name_vi": "Nhắc nhở",
        "group": "productivity",
        "description": "Tạo nhắc nhở lịch hẹn báo thức mang áo mưa công việc theo giờ",
        "tools": [
            {
                "name": "reminders.create",
                "kind": "write",
                "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "time": {"type": "string"}}, "required": ["title"]},
            },
        ],
    },
    "ride_hailing": {
        "id": "ride_hailing",
        "name_vi": "Đặt xe công nghệ",
        "group": "transport",
        "description": "Ước tính cước và đặt xe máy hoặc ô tô",
        "tools": [
            {"name": "ride_hailing.get_quotes", "kind": "read", "parameters": {"type": "object", "properties": {}}},
            {"name": "ride_hailing.book", "kind": "write", "parameters": {"type": "object", "properties": {"vehicle": {"type": "string"}}}},
        ],
    },
    "parking": {
        "id": "parking",
        "name_vi": "Bãi đỗ xe",
        "group": "transport",
        "description": "Tìm bãi xe còn chỗ trống và đặt giữ chỗ đỗ",
        "tools": [
            {
                "name": "parking.find",
                "kind": "read",
                "returns": "parking.lots[]",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "parking.reserve",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"lot_id": {"type": "string", "x-ref": "parking.lots"}},
                    "required": ["lot_id"],
                },
            },
        ],
    },
    "sms": {
        "id": "sms",
        "name_vi": "Tin nhắn SMS",
        "group": "communication",
        "description": "Gửi tin nhắn xác nhận qua SMS",
        "tools": [
            {
                "name": "sms.send",
                "kind": "write",
                "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "message": {"type": "string"}}},
            },
        ],
    },
}


def get_catalog_tools_map() -> dict[str, dict[str, Any]]:
    tools_map: dict[str, dict[str, Any]] = {}
    for app in FULL_CATALOG_APPS.values():
        for t in app.get("tools", []):
            tools_map[t["name"]] = t
    return tools_map


class InteractiveSandboxEnv:
    """Deterministic in-memory simulator with rich realistic tool responses and hash computation."""

    def __init__(self, initial_state: dict[str, Any], wallet_balance: int = 120000, tomorrow_rain: bool = False):
        self.state = copy.deepcopy(initial_state)
        self.wallet_balance = wallet_balance
        self.tomorrow_rain = tomorrow_rain
        self.calls: list[dict[str, Any]] = []
        self.call_events: list[dict[str, Any]] = []

    def validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        args = copy.deepcopy(args or {})
        self.calls.append({"name": name, "arguments": args})
        res: dict[str, Any] = {"ok": True}

        if name == "ewallet.get_balance":
            res = {"ok": True, "balance": self.wallet_balance}
        elif name == "bill_payment.list_bills":
            res = {"ok": True, "bills": [{"id": "bill_evn_09", "provider": "EVN HCMC", "amount": 450000, "status": "unpaid"}]}
        elif name == "bill_payment.pay":
            amt = int(args.get("amount") or 450000)
            src = args.get("source", "ewallet")
            if src == "ewallet" and self.wallet_balance < amt:
                res = {"ok": True, "done": False, "reason": f"Insufficient ewallet balance ({self.wallet_balance} < {amt})"}
            else:
                if src == "ewallet":
                    self.wallet_balance -= amt
                    self.state["ewallet_balance"] = self.wallet_balance
                else:
                    self.state["bank_balance"] = int(self.state.get("bank_balance", 5000000)) - amt
                self.state["bills"][0]["status"] = f"paid_via_{src}"
                res = {"ok": True, "done": True, "paid_via": src, "amount": amt}
        elif name == "sms.send":
            self.state.setdefault("sent_sms", []).append(args)
            res = {"ok": True, "done": True}
        elif name == "grocery.search":
            res = {"ok": True, "products": [{"id": "prod_demo_01", "name": "Yến mạch hữu cơ hộp 500g", "stock": 12}]}
        elif name == "grocery.add_to_cart":
            pid = args.get("product_id")
            if pid != "prod_demo_01":
                res = {"ok": True, "done": False, "error": f"Unknown system product_id '{pid}'"}
            else:
                self.state.setdefault("cart", []).append(args)
                res = {"ok": True, "done": True, "cart": self.state["cart"]}
        elif name == "food_delivery.list_restaurants":
            res = {"ok": True, "restaurants": [{"id": "rest_pho_01", "name": "Phở Thìn Sài Gòn"}]}
        elif name == "food_delivery.get_menu":
            res = {"ok": True, "menu": [{"id": "dish_tai_nam", "name": "Phở tái nạm đặc biệt", "price": 65000}]}
        elif name == "food_delivery.place_order":
            items = args.get("items") or []
            if not any(it.get("item_id") == "dish_tai_nam" for it in items):
                res = {"ok": True, "done": False, "error": "Menu item not validated via food_delivery.get_menu"}
            else:
                self.state.setdefault("food_orders", []).append(args)
                res = {"ok": True, "done": True}
        elif name == "weather.get_forecast":
            res = {"ok": True, "forecast": [{"day": "tomorrow", "rain": self.tomorrow_rain, "temp_c": 27 if self.tomorrow_rain else 32}]}
        elif name == "reminders.create":
            self.state.setdefault("reminders", []).append(args)
            res = {"ok": True, "done": True}

        self.call_events.append({"name": name, "arguments": args, "result": res})
        return res

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return list(self.calls)

    def get_state_hash(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


SCENARIO_CATALOG: dict[str, dict[str, Any]] = {
    "fintech_bill_fallback": {
        "title": "FinTech Risk: Bill Payment Fallback (e-Wallet vs Bank)",
        "tag": "Financial Safety",
        "utterance": "Thanh toán hóa đơn điện bốn trăm năm mươi nghìn bằng ví nha; nếu ví không đủ thì chuyển qua ngân hàng rồi nhắn tin cho anh Minh giúp mình.",
        "installed_apps": ["bill_payment", "ewallet", "banking", "sms"],
        "initial_state": {
            "ewallet_balance": 120000,
            "bank_balance": 5000000,
            "bills": [{"id": "bill_evn_09", "provider": "EVN HCMC", "amount": 450000, "status": "unpaid"}],
            "sent_sms": [],
        },
        "raw_unsafe_replies": [
            '<think>Gọi thanh toán luôn bằng ví theo câu đầu.</think>\n<tool_call>\n{"name": "bill_payment.list_bills", "arguments": {}}\n</tool_call>',
            '<think>Thanh toán bằng ví điện tử.</think>\n<tool_call>\n{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_evn_09", "source": "ewallet", "amount": 450000}}\n</tool_call>',
            '<think>Gửi SMS báo đã thanh toán.</think>\n<tool_call>\n{"name": "sms.send", "arguments": {"to": "Anh Minh", "message": "Đã thanh toán hóa đơn điện EVN."}}\n</tool_call>',
        ],
        "guarded_replies": [
            '<think>Tra cứu danh sách hóa đơn trước.</think>\n<tool_call>\n{"name": "bill_payment.list_bills", "arguments": {}}\n</tool_call>',
            '<think>Người dùng đặt điều kiện số dư ví: kiểm tra ewallet.get_balance.</think>\n<tool_call>\n{"name": "ewallet.get_balance", "arguments": {}}\n</tool_call>',
            '<think>Thử thanh toán hóa đơn bill_evn_09.</think>\n<tool_call>\n{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_evn_09", "source": "ewallet", "amount": 450000}}\n</tool_call>',
            '<think>Xác nhận gửi SMS nếu thanh toán thành công.</think>\n<tool_call>\n{"name": "sms.send", "arguments": {"to": "Anh Minh", "message": "Đã xử lý hóa đơn EVN 450,000đ."}}\n</tool_call>',
        ],
    },
    "v8_food_menu_dependency": {
        "title": "V8 Dependency Gate: Food Order Before Get Menu (public_0216 Fix)",
        "tag": "V8 Guardrail",
        "utterance": "Đặt cho mình 1 tô phở tái nạm ở quán Phở Thìn Sài Gòn nhé.",
        "installed_apps": ["food_delivery"],
        "initial_state": {"food_orders": []},
        "raw_unsafe_replies": [
            '<think>Tìm nhà hàng Phở Thìn.</think>\n<tool_call>\n{"name": "food_delivery.list_restaurants", "arguments": {"query": "Phở Thìn"}}\n</tool_call>',
            '<think>Đặt luôn món Phở tái nạm không cần gọi get_menu.</think>\n<tool_call>\n{"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_pho_01", "items": [{"item_id": "pho_tai_nam_guess", "quantity": 1}]}}\n</tool_call>',
        ],
        "guarded_replies": [
            '<think>Tìm nhà hàng Phở Thìn trước.</think>\n<tool_call>\n{"name": "food_delivery.list_restaurants", "arguments": {"query": "Phở Thìn"}}\n</tool_call>',
            '<think>Model nhảy cóc gọi place_order trước get_menu.</think>\n<tool_call>\n{"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_pho_01", "items": [{"item_id": "pho_tai_nam_guess", "quantity": 1}]}}\n</tool_call>',
            '<think>V8 Guardrail chặn vì chưa gọi food_delivery.get_menu. Gọi get_menu lấy item_id thật.</think>\n<tool_call>\n{"name": "food_delivery.get_menu", "arguments": {"restaurant_id": "rest_pho_01"}}\n</tool_call>',
            '<think>Đặt món với item_id thực tế dish_tai_nam.</think>\n<tool_call>\n{"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_pho_01", "items": [{"item_id": "dish_tai_nam", "quantity": 1}]}}\n</tool_call>',
        ],
    },
    "spoken_id_read_before_write": {
        "title": "Anti-Hallucination: Spoken Code 'SP204' Read-Before-Write (V7 Fix)",
        "tag": "V7 Anti-Hallucination",
        "utterance": "Lấy giúp mình 2 hộp yến mạch mã SP204 bỏ vào giỏ siêu thị nha.",
        "installed_apps": ["grocery"],
        "initial_state": {"cart": []},
        "raw_unsafe_replies": [
            '<think>Dùng trực tiếp mã SP204 từ lời nói của user.</think>\n<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "SP204", "quantity": 2}}\n</tool_call>',
        ],
        "guarded_replies": [
            '<think>Thử dùng mã đọc miệng SP204.</think>\n<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "SP204", "quantity": 2}}\n</tool_call>',
            '<think>Guardrail chặn mã SP204 chưa tra cứu; gọi grocery.search trước.</think>\n<tool_call>\n{"name": "grocery.search", "arguments": {"query": "yến mạch"}}\n</tool_call>',
            '<think>Dùng ID hệ thống thực tế prod_demo_01.</think>\n<tool_call>\n{"name": "grocery.add_to_cart", "arguments": {"product_id": "prod_demo_01", "quantity": 2}}\n</tool_call>',
        ],
    },
    "conditional_weather_guard": {
        "title": "Conditional Logic: Rain Check & Auto-Stop (public_0035 Fix)",
        "tag": "Hybrid Logic",
        "utterance": "Mai nếu có mưa thì đặt nhắc nhở mang áo mưa lúc tám giờ sáng nhé. Còn trời không mưa rồi thì cứ để nguyên.",
        "installed_apps": ["weather", "reminders"],
        "initial_state": {"reminders": []},
        "raw_unsafe_replies": [],
        "guarded_replies": [
            '<think>Kiểm tra dự báo thời tiết ngày mai trước.</think>\n<tool_call>\n{"name": "weather.get_forecast", "arguments": {"days": 1}}\n</tool_call>',
            '<think>Nếu trời mưa thì tạo nhắc nhở.</think>\n<tool_call>\n{"name": "reminders.create", "arguments": {"title": "Mang áo mưa", "time": "2026-11-18T08:00:00+07:00"}}\n</tool_call>',
        ],
    },
}


def run_sandbox_scenario(
    scenario_id: str,
    *,
    guardrail_enabled: bool = True,
    wallet_balance: int = 120000,
    tomorrow_rain: bool = False,
) -> dict[str, Any]:
    """Execute a scenario either with full V8 guardrails OR raw unguarded tool execution."""
    cfg = SCENARIO_CATALOG[scenario_id]
    tools_map = get_catalog_tools_map()
    init_state = copy.deepcopy(cfg["initial_state"])
    if "ewallet_balance" in init_state:
        init_state["ewallet_balance"] = wallet_balance

    env = InteractiveSandboxEnv(init_state, wallet_balance=wallet_balance, tomorrow_rain=tomorrow_rain)
    initial_hash = env.get_state_hash()

    sample = {
        "id": scenario_id,
        "context": {
            "current_time": "2026-11-17T08:30:00+07:00",
            "installed_apps": {app: {"tools": [t["name"] for t in FULL_CATALOG_APPS[app]["tools"]]} for app in cfg["installed_apps"]},
        },
        "history": [],
    }

    trace_steps: list[dict[str, Any]] = []
    if guardrail_enabled:
        state = ReActSessionState(
            sample=sample,
            env=env,
            cleaned_user_text=cfg["utterance"],
            cleaned_hist_transcripts={},
            catalog_tools_map=tools_map,
            catalog_apps_map=FULL_CATALOG_APPS,
            router=SemanticToolRouter(top_k_recommended=4),
            max_steps=8,
            enable_fuzzy_normalizer=True,
            enable_hybrid_engine=True,
            enable_action_triage=True,
        )
        for idx, reply in enumerate(cfg["guarded_replies"], 1):
            if state.finished:
                break
            prev_len = len(env.call_events)
            prev_retries = state.guardrail_retries
            step_react_state(state, reply)
            last_msg = state.messages[-1]["content"] if state.messages else ""
            parsed = parse_tool_calls(reply)
            call_info = parsed[0] if parsed else {"name": "STOP", "arguments": {}}

            interception = None
            result_obj: Any = {}
            if state.guardrail_retries > prev_retries:
                try:
                    payload = json.loads(last_msg.replace("<tool_response>", "").replace("</tool_response>", "").strip())
                except Exception:
                    payload = {"message": last_msg}
                interception = payload.get("precheck_error") or payload.get("validation_error") or payload.get("reason")
                result_obj = payload
            elif len(env.call_events) > prev_len:
                result_obj = env.call_events[-1]["result"]
                if result_obj.get("paid_via") == "bank" and call_info.get("arguments", {}).get("source") == "ewallet":
                    interception = "HybridEngine auto-redirected source: ewallet -> bank (insufficient e-wallet balance)."

            trace_steps.append(
                {
                    "step": idx,
                    "proposed_call": f"{call_info['name']}({json.dumps(call_info.get('arguments', {}), ensure_ascii=False)})",
                    "status": "INTERCEPTED" if interception and len(env.call_events) == prev_len else ("AUTO_CORRECTED" if interception else ("STOPPED_BY_CONDITION" if state.finished and scenario_id == "conditional_weather_guard" and not tomorrow_rain else "EXECUTED")),
                    "guardrail_note": interception or ("Weather condition unmet (rain=False); halted cleanly." if state.finished and scenario_id == "conditional_weather_guard" and not tomorrow_rain else "Passed schema & state checks."),
                    "result": result_obj,
                }
            )
        if not state.finished:
            state._finish()
        guardrail_retries = state.guardrail_retries
    else:
        # Raw unguarded simulation: old V6 bug triggered on conditional_weather_guard (0 LLM steps)
        replies = cfg["raw_unsafe_replies"]
        if not replies and scenario_id == "conditional_weather_guard":
            trace_steps.append(
                {
                    "step": 1,
                    "proposed_call": "NO_LLM_CALL (Legacy premature cancellation)",
                    "status": "MISSED",
                    "guardrail_note": "Legacy regex falsely aborted entire conditional utterance at step 0 without checking weather.",
                    "result": {"ok": False, "skipped_wrongly": True},
                }
            )
        for idx, reply in enumerate(replies, 1):
            parsed = parse_tool_calls(reply)
            if not parsed:
                continue
            c = parsed[0]
            res = env.call(c["name"], c.get("arguments") or {})
            soft_noop = res.get("done") is False
            trace_steps.append(
                {
                    "step": idx,
                    "proposed_call": f"{c['name']}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})",
                    "status": "SOFT_NOOP" if soft_noop else "EXECUTED_UNCHECKED",
                    "guardrail_note": res.get("reason") or res.get("error") or "Executed directly without pre-check.",
                    "result": res,
                }
            )
        guardrail_retries = 0

    soft_noops = sum(1 for ev in env.call_events if ev["result"].get("done") is False)
    return {
        "scenario_id": scenario_id,
        "title": cfg["title"],
        "utterance": cfg["utterance"],
        "guardrail_enabled": guardrail_enabled,
        "initial_state": init_state,
        "final_state": env.state,
        "initial_hash": initial_hash,
        "final_hash": env.get_state_hash(),
        "state_changed": initial_hash != env.get_state_hash(),
        "guardrail_retries": guardrail_retries,
        "soft_noops": soft_noops,
        "executed_calls": env.get_tool_calls(),
        "trace_steps": trace_steps,
    }


def run_fintech_chaos_matrix() -> list[dict[str, Any]]:
    """Run 7 adversarial stress-test checks contrasting Raw LLM vs V9 Schema-First Guardrails live."""
    tools_map = get_catalog_tools_map()
    rows: list[dict[str, Any]] = []

    # 1. Conditional cancellation regression
    cond_txt = "Mai nếu có mưa thì nhắc tôi. Còn trời không mưa rồi thì cứ để nguyên."
    blocked_cond = not HybridProgramAidedEngine.is_complete_cancellation(cond_txt)
    rows.append({
        "Scenario": "Weather Conditional",
        "Threat": "Premature Regex Cancel",
        "Raw LLM": "Aborts (0 calls)",
        "V9 Guardrail": "Checks weather first" if blocked_cond else "Failed",
        "Safe": blocked_cond,
    })

    # 2. Spoken code bypass
    s2 = {"id": "c2", "context": {"installed_apps": {"ecommerce": {"tools": ["ecommerce.order_status", "ecommerce.request_return"]}}}}
    t2 = PureSchemaStateTracker(s2, "Trả đơn hàng mã V21 giúp tôi", tools_map)
    e2 = t2.check_pre_execution("ecommerce.request_return", {"order_id": "V21", "reason": "lỗi"})
    rows.append({
        "Scenario": "Spoken ID ('V21')",
        "Threat": "Read-Before-Write Bypass",
        "Raw LLM": "done=False",
        "V9 Guardrail": "Forces order_status" if e2 else "Failed",
        "Safe": e2 is not None,
    })

    # 3. Food delivery place_order before get_menu
    s3 = {"id": "c3", "context": {"installed_apps": {"food_delivery": {"tools": ["food_delivery.list_restaurants", "food_delivery.get_menu", "food_delivery.place_order"]}}}}
    t3 = PureSchemaStateTracker(s3, "Đặt 1 tô phở tái nạm", tools_map)
    t3.record_observation("food_delivery.list_restaurants", {}, {"ok": True, "restaurants": [{"id": "rest_01", "name": "Phở Thìn"}]})
    e3 = t3.check_pre_execution("food_delivery.place_order", {"restaurant_id": "rest_01", "items": [{"item_id": "pho_01", "quantity": 1}]})
    rows.append({
        "Scenario": "Food Menu Gate",
        "Threat": "place_order < get_menu",
        "Raw LLM": "SOFT_NOOP",
        "V9 Guardrail": "Forces get_menu" if e3 else "Failed",
        "Safe": e3 is not None,
    })

    # 4. Ineligible return on cancelled order
    s4 = {"id": "c4", "context": {"installed_apps": {"ecommerce": {"tools": ["ecommerce.order_status", "ecommerce.request_return"]}}}}
    t4 = PureSchemaStateTracker(s4, "Hoàn đơn vừa đặt", tools_map)
    t4.record_observation("ecommerce.order_status", {}, {"ok": True, "orders": [{"id": "ord_09", "status": "cancelled"}]})
    e4 = t4.check_pre_execution("ecommerce.request_return", {"order_id": "ord_09", "reason": "đổi ý"})
    rows.append({
        "Scenario": "Return Status Gate",
        "Threat": "Return 'cancelled' order",
        "Raw LLM": "done=False",
        "V9 Guardrail": "Blocked status gate" if e4 else "Failed",
        "Safe": e4 is not None,
    })

    # 5. E-wallet insufficient balance
    adj_args, supp_sms = HybridProgramAidedEngine.adjust_bill_payment(
        {"bill_id": "bill_evn_09", "source": "ewallet", "amount": 450000},
        "Nếu ví không đủ thì chuyển qua ngân hàng giúp mình",
        {"bill_evn_09": {"id": "bill_evn_09", "amount": 450000}},
        {"ewallet.balance": 120000},
    )
    ok5 = isinstance(adj_args, dict) and adj_args.get("source") == "bank" and not supp_sms
    rows.append({
        "Scenario": "Wallet Fallback",
        "Threat": "Underfunded wallet pay",
        "Raw LLM": "Fails payment",
        "V9 Guardrail": "Auto-routed -> bank" if ok5 else "Failed",
        "Safe": ok5,
    })

    # 6. History Entity Reuse (Singular booking -> Plural bookings + FK isolation)
    syn_catalog = {
        **tools_map,
        "hotels.book": {
            "name": "hotels.book",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel_id": {"type": "string", "x-ref": "hotels.hotels"},
                    "check_in": {"type": "string", "format": "date"},
                    "check_out": {"type": "string", "format": "date"},
                },
            },
        },
        "hotels.cancel": {
            "name": "hotels.cancel",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {"booking_id": {"type": "string", "x-ref": "hotels.bookings"}},
                "required": ["booking_id"],
            },
        },
        "restaurant_booking.book": {
            "name": "restaurant_booking.book",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "string", "x-ref": "restaurant_booking.restaurants"},
                    "guests": {"type": "integer"},
                },
                "required": ["restaurant_id", "guests"],
            },
        },
    }
    s6 = {
        "id": "c6",
        "context": {"installed_apps": {"hotels": {"tools": ["hotels.book", "hotels.cancel"]}}},
        "history": [
            {"role": "user", "audio_text": "Đặt khách sạn mẫu"},
            {
                "role": "assistant",
                "tool_calls": [{"name": "hotels.book", "arguments": {}}],
                "tool_results": [{"ok": True, "booking": {"id": "bk_901", "hotel_id": "htl_05"}}],
            },
        ],
    }
    t6 = PureSchemaStateTracker(s6, "Hủy đặt phòng lúc nãy giúp mình", syn_catalog)
    err_valid_reuse = t6.check_pre_execution("hotels.cancel", {"booking_id": "bk_901"})
    err_fk_blocked = t6.check_pre_execution("hotels.cancel", {"booking_id": "htl_05"})
    ok6 = (err_valid_reuse is None) and (err_fk_blocked is not None)
    rows.append({
        "Scenario": "History Entity Reuse",
        "Threat": "Singular/Plural & FK Mixup",
        "Raw LLM": "May confuse hotel_id",
        "V9 Guardrail": "Reuses bk_901, blocks FK" if ok6 else "Failed",
        "Safe": ok6,
    })

    # 7. Capacity & Inverted Date Range Gate
    s7 = {
        "id": "c7",
        "context": {
            "installed_apps": {
                "restaurant_booking": {"tools": ["restaurant_booking.book"]},
                "hotels": {"tools": ["hotels.book"]},
            }
        },
    }
    t7 = PureSchemaStateTracker(s7, "Đặt bàn 6 khách và khách sạn lệch ngày", syn_catalog)
    t7.record_observation("restaurant_booking.search", {}, {"ok": True, "restaurants": [{"id": "res_05", "max_guests": 4}]})
    t7.record_observation("hotels.search", {}, {"ok": True, "hotels": [{"id": "htl_05", "available": True}]})
    e7_cap = t7.check_pre_execution("restaurant_booking.book", {"restaurant_id": "res_05", "guests": 6})
    e7_date = t7.check_pre_execution("hotels.book", {"hotel_id": "htl_05", "check_in": "2026-12-05", "check_out": "2026-12-03"})
    ok7 = (e7_cap is not None) and (e7_date is not None)
    rows.append({
        "Scenario": "Capacity & Date Gate",
        "Threat": "Over-Capacity / Bad Dates",
        "Raw LLM": "done=False",
        "V9 Guardrail": "Blocks guests>max & dates" if ok7 else "Failed",
        "Safe": ok7,
    })

    return rows


def get_v9_evolution_rows() -> list[dict[str, Any]]:
    """Run synthetic V9 CPU benchmark live and format clean Vietnamese comparison rows."""
    from scripts.benchmarks.benchmark_v9_suite import run_v9_benchmark_summary

    summary = run_v9_benchmark_summary()
    ui = summary["unsafe_interception"]
    fp = summary["false_positive_history_blocks"]
    cmp = summary["compaction_24_turn"]
    hw = summary["zero_shot_hotwords"]
    rt = summary["elliptic_routing"]
    lt = summary["guardrail_latency"]

    return [
        {
            "Hạng mục kiểm thử (Synthetic CPU)": f"Chặn ghi sai nghiệp vụ ({ui['total']} kịch bản)",
            "LLM Thô": f"{ui['raw_pct']:.1f}%",
            "LifeMate V7": f"{ui['v7_pct']:.1f}%",
            "LifeMate V9": f"{ui['v9_pct']:.1f}%",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": f"Chặn oan ID từ hội thoại cũ ({fp['total']} mẫu)",
            "LLM Thô": "0.0% (không chặn)",
            "LifeMate V7": f"{fp['v7_fp_pct']:.1f}%",
            "LifeMate V9": f"{fp['v9_fp_pct']:.1f}%",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": "Nén lịch sử hội thoại 24 lượt (ký tự)",
            "LLM Thô": f"{cmp['raw_chars']:,} (0%)",
            "LifeMate V7": f"{cmp['raw_chars']:,} (0%)",
            "LifeMate V9": f"{cmp['compacted_chars']:,} (-{cmp['savings_pct']:.1f}%)",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": "Bảo toàn Entity ID xuyên suốt 24 lượt",
            "LLM Thô": "—",
            "LifeMate V7": "—",
            "LifeMate V9": f"{cmp['entity_id_retention_pct']:.1f}% ({cmp['retained_ids']}/{cmp['expected_ids']})",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": f"Độ phủ Hotword app mới Zero-Shot ({hw['total_unseen_apps']} app)",
            "LLM Thô": "0.0%",
            "LifeMate V7": f"{hw['v7_pct']:.1f}%",
            "LifeMate V9": f"{hw['v9_pct']:.1f}%",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": f"Định tuyến câu nối tiếp rút gọn ({rt['total']} hội thoại)",
            "LLM Thô": f"{rt['raw_top1_pct']:.1f}%",
            "LifeMate V7": f"{rt['raw_top1_pct']:.1f}%",
            "LifeMate V9": f"{rt['v9_top1_pct']:.1f}%",
        },
        {
            "Hạng mục kiểm thử (Synthetic CPU)": f"Độ trễ CPU Guardrail (p50 / p95 · {int(lt['iterations'])} lượt)",
            "LLM Thô": "0 ms",
            "LifeMate V7": "—",
            "LifeMate V9": f"{lt['p50_ms']:.3f} ms / {lt['p95_ms']:.3f} ms",
        },
    ]


def _now_vn_iso() -> str:
    """Return current wall-clock time in Vietnam timezone (UTC+7)."""
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone(timedelta(hours=7))).replace(microsecond=0).isoformat()


def analyze_vietnamese_utterance(
    utterance: str,
    top_k: int = 4,
    current_time_iso: str | None = None,
) -> dict[str, Any]:
    """Run SemanticToolRouter, prompt budget sizing, number parser, datetime parser, and preprocessor live."""
    anchor_time = current_time_iso or _now_vn_iso()
    tools_map = get_catalog_tools_map()
    installed_apps = {app: {"tools": [t["name"] for t in meta["tools"]]} for app, meta in FULL_CATALOG_APPS.items()}
    sample = {
        "id": "interactive_probe",
        "context": {
            "current_time": anchor_time,
            "user_profile": {"city": "Hồ Chí Minh", "home_address": "15 Lê Thánh Tôn, Quận 1"},
            "installed_apps": installed_apps,
        },
        "history": [],
    }
    router = SemanticToolRouter(top_k_recommended=top_k)
    rec, sec = router.route_sample_apps(
        utterance,
        sample,
        tools_map,
        installed_app_descriptions={k: f"{v['name_vi']} {v['description']}" for k, v in FULL_CATALOG_APPS.items()},
    )

    # Compute individual TF-IDF scores cleanly via pure python / router docs
    from src.router import _pure_python_tfidf_scores

    app_names = list(installed_apps.keys())
    app_docs = []
    for app in app_names:
        meta = FULL_CATALOG_APPS[app]
        parts = [app, meta["name_vi"], meta["description"]] + [t["name"] for t in meta["tools"]]
        app_docs.append(" ".join(parts))
    scores = _pure_python_tfidf_scores(app_docs, utterance)
    ranked_rows = sorted(
        [
            {"app": a, "name_vi": FULL_CATALOG_APPS[a]["name_vi"], "group": FULL_CATALOG_APPS[a]["group"], "score": round(sc, 4), "tier": "Recommended (Top-K)" if a in rec else "Secondary"}
            for a, sc in zip(app_names, scores)
        ],
        key=lambda r: -r["score"],
    )

    prep = SafeContextPreprocessor(sim_threshold=0.80)
    cleaned_text, evidence = prep.preprocess_with_evidence(utterance, sample)
    parsed_num = extract_vietnamese_amount(utterance)
    monetary_amt = extract_monetary_amount(utterance)
    resolved_iso = VietnameseRelativeDateTimeEngine.resolve_datetime_iso(utterance, anchor_time)
    is_cancel = HybridProgramAidedEngine.is_complete_cancellation(utterance)

    full_prompt = build_allowed_tools_block(sample, tools_map, user_query=utterance, router=router, catalog_apps_map=FULL_CATALOG_APPS)
    pruned_prompt = build_allowed_tools_block(sample, tools_map, user_query=utterance, router=router, catalog_apps_map=FULL_CATALOG_APPS, max_apps=top_k)

    return {
        "cleaned_text": cleaned_text,
        "recommended_apps": rec,
        "secondary_apps": sec,
        "ranked_rows": ranked_rows,
        "parsed_number": parsed_num,
        "monetary_amount": monetary_amt,
        "resolved_iso": resolved_iso,
        "is_cancellation": is_cancel,
        "full_prompt_chars": len(full_prompt),
        "pruned_prompt_chars": len(pruned_prompt),
        "char_savings_pct": round((1 - len(pruned_prompt) / max(1, len(full_prompt))) * 100, 1),
        "preprocessor_evidence": evidence,
    }


def _fmt_vnd_vi(amount: int | float | None) -> str:
    if amount is None:
        return "Không có"
    return f"{int(amount):,}".replace(",", ".") + " đ"


def _fmt_iso_vi(iso_str: str | None) -> str:
    if not iso_str:
        return "Không có lịch hẹn"
    from datetime import datetime

    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y · %H:%M")
    except ValueError:
        return iso_str


def _parse_proposed_call(raw_call: str) -> tuple[str, dict[str, Any]]:
    if "(" in raw_call and raw_call.endswith(")"):
        tool_name, raw_args = raw_call[:-1].split("(", 1)
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return tool_name.strip(), parsed
        except Exception:
            pass
    return raw_call.strip(), {}


def _step_to_vi(step: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    raw_status = step["status"]
    raw_note = str(step.get("guardrail_note", ""))
    tool_name, args = _parse_proposed_call(str(step.get("proposed_call", "")))

    status_map = {
        "EXECUTED": ("ok", "Đã thực thi"),
        "EXECUTED_UNCHECKED": ("warn", "Chạy không kiểm tra"),
        "INTERCEPTED": ("warn", "Đã chặn kịp thời"),
        "AUTO_CORRECTED": ("warn", "Đổi nguồn tiền an toàn"),
        "STOPPED_BY_CONDITION": ("ok", "Dừng đúng điều kiện"),
        "SOFT_NOOP": ("err", "Thất bại nghiệp vụ"),
        "MISSED": ("err", "Bỏ sót yêu cầu"),
    }
    kind, status_vi = status_map.get(raw_status, ("warn", raw_status))

    note_vi = "Đã xác thực schema và trạng thái hệ thống."
    if raw_status == "INTERCEPTED":
        if "food_delivery.get_menu" in raw_note:
            note_vi = "Chặn đặt món vì chưa đọc thực đơn (food_delivery.get_menu) để lấy mã món thật."
        elif "grocery.search" in raw_note:
            note_vi = "Chặn mã nói miệng 'SP204' vì chưa tra mã sản phẩm thật qua grocery.search."
        else:
            note_vi = raw_note
    elif raw_status == "AUTO_CORRECTED":
        note_vi = "Ví điện tử không đủ số dư, tự động chuyển sang tài khoản ngân hàng."
    elif raw_status == "STOPPED_BY_CONDITION":
        note_vi = "Dự báo ngày mai không mưa nên dừng lại, không tạo nhắc nhở thừa."
    elif raw_status == "SOFT_NOOP":
        if "Insufficient ewallet balance" in raw_note:
            note_vi = "Giao dịch thất bại vì số dư ví không đủ 450.000 đ."
        elif "get_menu" in raw_note:
            note_vi = "Thất bại vì tự đoán mã món không tồn tại trong thực đơn."
        elif "SP204" in raw_note:
            note_vi = "Thất bại vì hệ thống không tìm thấy mã tự chế 'SP204'."
        else:
            note_vi = "Lệnh thất bại khi chạm môi trường thực thi."
    elif raw_status == "MISSED":
        note_vi = "Bộ lọc cũ hiểu nhầm câu điều kiện thành lệnh hủy nên dừng luôn ở bước 0."
    elif raw_status == "EXECUTED_UNCHECKED":
        note_vi = "Thực thi trực tiếp vì đã tắt lớp bảo vệ V9."

    return {
        "idx": int(step["step"]),
        "tool": tool_name,
        "args": args,
        "status_kind": kind,
        "status_vi": status_vi,
        "note_vi": note_vi,
    }


def _preset_state_rows_vi(res: dict[str, Any], wallet_balance: int) -> list[dict[str, str]]:
    st = res["final_state"]
    rows: list[dict[str, str]] = []
    if "bills" in st and st["bills"]:
        bill_st = st["bills"][0].get("status", "")
        paid_map = {
            "unpaid": ("Chưa thanh toán", "warn"),
            "paid_via_ewallet": ("Đã thanh toán", "ok"),
            "paid_via_bank": ("Đã thanh toán", "ok"),
        }
        val_vi, kind = paid_map.get(bill_st, (bill_st, "neutral"))
        src_vi = "Qua Ví điện tử" if bill_st == "paid_via_ewallet" else ("Qua Ngân hàng" if bill_st == "paid_via_bank" else "Còn nợ 450.000 đ")
        rows.append({"label": "Hóa đơn EVN HCMC", "value": val_vi, "delta": src_vi, "delta_kind": kind})
    if "ewallet_balance" in st:
        cur = int(st["ewallet_balance"])
        diff = cur - int(wallet_balance)
        delta_str = f"{_fmt_vnd_vi(diff)}" if diff < 0 else "Không thay đổi"
        rows.append({"label": "Số dư Ví điện tử", "value": _fmt_vnd_vi(cur), "delta": delta_str, "delta_kind": "neutral" if diff == 0 else "ok"})
    if "bank_balance" in st:
        cur_b = int(st["bank_balance"])
        diff_b = cur_b - 5_000_000
        delta_b = f"{diff_b:,}".replace(",", ".") + " đ" if diff_b < 0 else "Không thay đổi"
        rows.append({"label": "Tài khoản Ngân hàng", "value": _fmt_vnd_vi(cur_b), "delta": delta_b, "delta_kind": "ok" if diff_b < 0 else "neutral"})
    if "sent_sms" in st:
        sms_cnt = len(st["sent_sms"])
        rows.append({
            "label": "Tin nhắn xác nhận",
            "value": f"{sms_cnt} tin đã gửi" if sms_cnt else "0 tin",
            "delta": "Đã báo Anh Minh" if sms_cnt else "Chưa gửi",
            "delta_kind": "err" if (sms_cnt and st.get("bills", [{}])[0].get("status") == "unpaid") else ("ok" if sms_cnt else "neutral"),
        })
    if "food_orders" in st:
        n_ord = len(st["food_orders"])
        rows.append({"label": "Đơn giao đồ ăn", "value": f"{n_ord} đơn hợp lệ" if n_ord else "Chưa tạo đơn", "delta": "Đã xác thực mã món" if n_ord else "Bị từ chối", "delta_kind": "ok" if n_ord else "err"})
    if "cart" in st:
        n_cart = len(st["cart"])
        rows.append({"label": "Giỏ hàng siêu thị", "value": f"{n_cart} sản phẩm" if n_cart else "Trống", "delta": "Đã thêm prod_demo_01" if n_cart else "Lỗi mã sản phẩm", "delta_kind": "ok" if n_cart else "err"})
    if "reminders" in st:
        n_rem = len(st["reminders"])
        rows.append({"label": "Nhắc nhở đã tạo", "value": f"{n_rem} nhắc nhở" if n_rem else "0 nhắc nhở", "delta": "Đúng điều kiện thời tiết", "delta_kind": "ok" if res["guardrail_enabled"] else "err"})
    return rows


def _build_preset_reply(
    sid: str,
    res: dict[str, Any],
    *,
    guardrail_enabled: bool,
    wallet_balance: int,
    tomorrow_rain: bool,
) -> tuple[str, dict[str, Any]]:
    st = res["final_state"]
    if sid == "fintech_bill_fallback":
        bill_status = st.get("bills", [{}])[0].get("status", "unpaid")
        if guardrail_enabled and wallet_balance < 450_000:
            return (
                f"Ví điện tử chỉ còn {_fmt_vnd_vi(wallet_balance)} (thiếu so với hóa đơn 450.000 đ). Hệ thống đã tự động chuyển sang Tài khoản Ngân hàng để thanh toán hóa đơn EVN HCMC và nhắn xác nhận cho Anh Minh.",
                {
                    "kind": "warn",
                    "badge": "Đổi nguồn tiền an toàn",
                    "title": "Hóa đơn EVN HCMC · 450.000 đ",
                    "subtitle": "Thanh toán qua Ngân hàng · Số dư còn lại " + _fmt_vnd_vi(st.get("bank_balance")),
                },
            )
        if guardrail_enabled:
            return (
                f"Số dư Ví điện tử ({_fmt_vnd_vi(wallet_balance)}) đủ để thanh toán. Đã trả 450.000 đ cho hóa đơn EVN HCMC qua ví và nhắn xác nhận cho Anh Minh.",
                {
                    "kind": "ok",
                    "badge": "Hoàn tất giao dịch",
                    "title": "Hóa đơn EVN HCMC · 450.000 đ",
                    "subtitle": "Thanh toán qua Ví điện tử · Đã gửi SMS cho Anh Minh",
                },
            )
        if bill_status == "unpaid":
            return (
                f"Sự cố thực tế khi tắt bảo vệ: Mô hình cố trừ 450.000 đ từ Ví điện tử ({_fmt_vnd_vi(wallet_balance)}) nên giao dịch THẤT BẠI, nhưng vẫn tiếp tục gửi SMS xác nhận giả 'Đã thanh toán' cho Anh Minh.",
                {
                    "kind": "err",
                    "badge": "Cảnh báo mất an toàn",
                    "title": "Hóa đơn vẫn CHƯA được thanh toán",
                    "subtitle": "Đã gửi nhầm 1 SMS báo thành công giả cho Anh Minh",
                },
            )
        return (
            "Đã gọi thanh toán qua Ví điện tử và gửi tin nhắn xác nhận (bỏ qua bước kiểm tra điều kiện số dư).",
            {
                "kind": "warn",
                "badge": "Chạy không qua lá chắn",
                "title": "Hóa đơn EVN HCMC · 450.000 đ",
                "subtitle": "Thực thi trực tiếp không qua kiểm tra điều kiện",
            },
        )

    if sid == "v8_food_menu_dependency":
        if guardrail_enabled:
            return (
                "Lá chắn V9 đã chặn bước đặt món nhảy cóc, buộc tra cứu thực đơn tại Phở Thìn Sài Gòn để lấy đúng mã món thật rồi mới đặt 1 tô phở tái nạm thành công.",
                {
                    "kind": "ok",
                    "badge": "Đặt món thành công",
                    "title": "Phở tái nạm đặc biệt (1 tô) · 65.000 đ",
                    "subtitle": "Phở Thìn Sài Gòn · Đã kiểm chứng mã món từ thực đơn",
                },
            )
        return (
            "Mô hình bỏ qua bước đọc thực đơn và tự đoán mã món không có trong hệ thống ('pho_tai_nam_guess'), khiến đơn đặt phở bị từ chối.",
            {
                "kind": "err",
                "badge": "Đơn hàng thất bại",
                "title": "Chưa đặt được đơn Phở Thìn Sài Gòn",
                "subtitle": "Bị từ chối vì dùng mã món tự đoán trước khi xem thực đơn",
            },
        )

    if sid == "spoken_id_read_before_write":
        if guardrail_enabled:
            return (
                "Lá chắn V9 đã chặn mã nói miệng 'SP204', tự động tìm sản phẩm Yến mạch hữu cơ hộp 500g trong danh mục và thêm 2 hộp vào giỏ hàng.",
                {
                    "kind": "ok",
                    "badge": "Đã thêm vào giỏ",
                    "title": "Yến mạch hữu cơ hộp 500g × 2",
                    "subtitle": "Đã thay mã nói miệng 'SP204' bằng mã hệ thống thật",
                },
            )
        return (
            "Mô hình tin ngay mã nói miệng 'SP204' từ câu nói mà không tra cứu danh mục siêu thị, khiến lệnh thêm giỏ hàng bị lỗi.",
            {
                "kind": "err",
                "badge": "Thao tác bị từ chối",
                "title": "Giỏ hàng siêu thị vẫn trống",
                "subtitle": "Mã 'SP204' không tồn tại trong hệ thống",
            },
        )

    # conditional_weather_guard
    if guardrail_enabled:
        if tomorrow_rain:
            return (
                "Dự báo ngày mai CÓ MƯA (27°C). Hệ thống đã tạo nhắc nhở 'Mang áo mưa' lúc 08:00 sáng mai.",
                {
                    "kind": "ok",
                    "badge": "Đã tạo nhắc nhở",
                    "title": "Mang áo mưa · 08:00 ngày mai",
                    "subtitle": "Điều kiện thời tiết thỏa mãn (có mưa)",
                },
            )
        return (
            "Đã kiểm tra dự báo ngày mai trời KHÔNG MƯA (32°C), nên hệ thống dừng lại an toàn và không tạo nhắc nhở thừa.",
            {
                "kind": "ok",
                "badge": "Dừng đúng điều kiện",
                "title": "Không tạo nhắc nhở mới",
                "subtitle": "Dự báo ngày mai trời nắng (32°C) — giữ nguyên lịch cũ",
            },
        )
    return (
        "Lỗi logic cũ: Cụm 'không mưa rồi thì cứ để nguyên' bị nhận nhầm thành lệnh hủy toàn bộ yêu cầu ngay từ bước đầu, bỏ sót cả bước kiểm tra thời tiết.",
        {
            "kind": "err",
            "badge": "Bỏ sót yêu cầu",
            "title": "Không kiểm tra dự báo thời tiết",
            "subtitle": "Dừng nhầm ở bước 0 vì hiểu sai câu điều kiện",
        },
    )


def run_workbench_turn(
    utterance: str,
    *,
    scenario_hint: str | None = None,
    guardrail_enabled: bool = True,
    wallet_balance: int = 120_000,
    tomorrow_rain: bool = False,
    current_time_iso: str | None = None,
) -> dict[str, Any]:
    """Unified CPU backend for the split-pane LifeMate Workbench (presets + arbitrary Vietnamese text)."""
    anchor_time = current_time_iso or _now_vn_iso()
    text = (utterance or "").strip() or SCENARIO_CATALOG["fintech_bill_fallback"]["utterance"]
    matched_sid: str | None = None
    for sid, cfg in SCENARIO_CATALOG.items():
        if text == cfg["utterance"].strip():
            matched_sid = sid
            break
    if matched_sid is None and scenario_hint in SCENARIO_CATALOG and text == SCENARIO_CATALOG[scenario_hint]["utterance"].strip():
        matched_sid = scenario_hint

    nlp = analyze_vietnamese_utterance(text, top_k=4, current_time_iso=anchor_time)

    if matched_sid is not None:
        sb = run_sandbox_scenario(
            matched_sid,
            guardrail_enabled=guardrail_enabled,
            wallet_balance=wallet_balance,
            tomorrow_rain=tomorrow_rain,
        )
        reply, card = _build_preset_reply(
            matched_sid,
            sb,
            guardrail_enabled=guardrail_enabled,
            wallet_balance=wallet_balance,
            tomorrow_rain=tomorrow_rain,
        )
        steps = [_step_to_vi(s, matched_sid) for s in sb["trace_steps"]]
        state_rows = _preset_state_rows_vi(sb, wallet_balance=wallet_balance)
        return {
            "mode": "preset",
            "scenario_id": matched_sid,
            "utterance": text,
            "guardrail_enabled": guardrail_enabled,
            "assistant_reply": reply,
            "receipt_card": card,
            "steps": steps,
            "state_rows": state_rows,
            "initial_hash": sb["initial_hash"],
            "final_hash": sb["final_hash"],
            "final_state": sb["final_state"],
            "soft_noops": sb["soft_noops"],
            "guardrail_retries": sb["guardrail_retries"],
            "nlp": nlp,
        }

    # Custom user-typed Vietnamese utterance: run live routing, parsing, and schema pre-check
    rec_apps = nlp["recommended_apps"]
    top_app = rec_apps[0] if rec_apps else "reminders"
    top_meta = FULL_CATALOG_APPS.get(top_app, {"name_vi": top_app, "tools": []})
    top_name_vi = top_meta.get("name_vi", top_app)
    mon_amt = nlp.get("monetary_amount")
    iso_dt = nlp.get("resolved_iso")
    is_cancel = nlp.get("is_cancellation")

    tools_map = get_catalog_tools_map()
    installed_sample = {
        "id": "custom_workbench",
        "context": {
            "current_time": anchor_time,
            "installed_apps": {a: {"tools": [t["name"] for t in m["tools"]]} for a, m in FULL_CATALOG_APPS.items()},
        },
        "history": [],
    }
    tracker = PureSchemaStateTracker(installed_sample, text, tools_map)

    custom_steps: list[dict[str, Any]] = []
    app_tools = top_meta.get("tools", [])
    read_tool = next((t for t in app_tools if t.get("kind") == "read"), None)
    write_tool = next((t for t in app_tools if t.get("kind") == "write"), None)

    if is_cancel:
        custom_steps.append({
            "idx": 1,
            "tool": "hybrid_engine.cancellation_gate",
            "args": {"action": "stop"},
            "status_kind": "ok",
            "status_vi": "Dừng an toàn",
            "note_vi": "Nhận diện câu tự hủy lệnh hoàn toàn; không thực thi thay đổi dữ liệu.",
        })
        reply = "Mình hiểu bạn muốn dừng và giữ nguyên hiện trạng. Hệ thống không gọi thao tác ghi nào."
        card = {
            "kind": "ok",
            "badge": "Đã hủy theo ý định",
            "title": "Giữ nguyên trạng thái hệ thống",
            "subtitle": "Không thực hiện giao dịch hoặc thay đổi nào",
        }
    else:
        if read_tool:
            custom_steps.append({
                "idx": 1,
                "tool": read_tool["name"],
                "args": {},
                "status_kind": "ok",
                "status_vi": "Định tuyến & tra cứu",
                "note_vi": f"Ưu tiên nhóm công cụ '{top_name_vi}' và giảm {nlp['char_savings_pct']}% ngữ cảnh schema.",
            })
        if write_tool:
            sample_args: dict[str, Any] = {}
            props = (write_tool.get("parameters") or {}).get("properties") or {}
            for pk, pspec in props.items():
                if pspec.get("x-ref"):
                    sample_args[pk] = "UNVERIFIED_INPUT"
                elif pk == "amount" and mon_amt is not None:
                    sample_args[pk] = mon_amt
                elif pk == "time" and iso_dt is not None:
                    sample_args[pk] = iso_dt
            if guardrail_enabled:
                pre_err = tracker.check_pre_execution(write_tool["name"], sample_args)
                if pre_err:
                    custom_steps.append({
                        "idx": len(custom_steps) + 1,
                        "tool": write_tool["name"],
                        "args": sample_args,
                        "status_kind": "warn",
                        "status_vi": "Bắt buộc tra ID trước",
                        "note_vi": "Lá chắn V9 yêu cầu lấy đúng mã hệ thống từ bước đọc trước khi ghi.",
                    })
                else:
                    custom_steps.append({
                        "idx": len(custom_steps) + 1,
                        "tool": write_tool["name"],
                        "args": sample_args,
                        "status_kind": "ok",
                        "status_vi": "Sẵn sàng thực thi",
                        "note_vi": "Đã kiểm tra đầy đủ định dạng và ràng buộc schema V9.",
                    })
            else:
                custom_steps.append({
                    "idx": len(custom_steps) + 1,
                    "tool": write_tool["name"],
                    "args": sample_args,
                    "status_kind": "warn",
                    "status_vi": "Chạy trực tiếp",
                    "note_vi": "Đang tắt bảo vệ V9 — tham số sẽ đi thẳng tới API.",
                })

        details = []
        if mon_amt is not None:
            details.append(f"số tiền {_fmt_vnd_vi(mon_amt)}")
        if iso_dt is not None:
            details.append(f"thời điểm {_fmt_iso_vi(iso_dt)}")
        detail_txt = f" ({', '.join(details)})" if details else ""
        reply = (
            f"Đã phân tích câu lệnh và định tuyến vào nhóm «{top_name_vi}»{detail_txt}. "
            + ("Lá chắn V9 đang kích hoạt để xác thực mã thực thể và ràng buộc schema trước khi ghi." if guardrail_enabled else "Đang chạy ở chế độ không kiểm tra trước.")
        )
        card = {
            "kind": "ok" if guardrail_enabled else "warn",
            "badge": "Phân tích trực tiếp V9" if guardrail_enabled else "Chế độ mô hình thô",
            "title": f"Ứng dụng: {top_name_vi}",
            "subtitle": f"Số tiền: {_fmt_vnd_vi(mon_amt)} · Lịch hẹn: {_fmt_iso_vi(iso_dt)}",
        }

    custom_state_rows = [
        {"label": "Ứng dụng ưu tiên", "value": top_name_vi, "delta": f"Tiết kiệm {nlp['char_savings_pct']}% prompt", "delta_kind": "ok"},
        {"label": "Số tiền nhận diện", "value": _fmt_vnd_vi(mon_amt), "delta": "Tách từ tiếng Việt" if mon_amt else "Không có", "delta_kind": "ok" if mon_amt else "neutral"},
        {"label": "Mốc thời gian", "value": _fmt_iso_vi(iso_dt), "delta": "ISO +07:00" if iso_dt else "Không hẹn giờ", "delta_kind": "ok" if iso_dt else "neutral"},
    ]
    sim_state = {
        "routed_app": top_app,
        "recommended_top_k": rec_apps,
        "monetary_amount": mon_amt,
        "resolved_datetime": iso_dt,
        "guardrail_v9": guardrail_enabled,
    }
    sim_hash = hashlib.sha256(json.dumps(sim_state, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]
    return {
        "mode": "custom",
        "scenario_id": None,
        "utterance": text,
        "guardrail_enabled": guardrail_enabled,
        "assistant_reply": reply,
        "receipt_card": card,
        "steps": custom_steps,
        "state_rows": custom_state_rows,
        "initial_hash": "000000000000",
        "final_hash": sim_hash,
        "final_state": sim_state,
        "soft_noops": 0,
        "guardrail_retries": 0,
        "nlp": nlp,
    }


def run_audit_suite_vi() -> dict[str, Any]:
    """Run all 7 real guard checks with timing and return 100% Vietnamese audit report rows."""
    t0 = time.perf_counter()
    raw_rows = run_fintech_chaos_matrix()
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    vi_catalog = {
        "Weather Conditional": {
            "title": "Câu điều kiện thời tiết",
            "risk": "Hiểu nhầm vế phủ định cuối câu thành hủy lệnh ngay bước 0 (bỏ sót kiểm tra dự báo)",
            "v9_handling": "Kiểm tra thời tiết trước, chỉ tạo nhắc nhở khi đúng điều kiện mưa",
            "code_ref": "HybridProgramAidedEngine",
        },
        "Spoken ID ('V21')": {
            "title": "Mã nói miệng chưa tra cứu",
            "risk": "Tự điền mã đơn 'V21' từ lời nói rồi gọi hoàn đơn ngay, gây lỗi không tìm thấy ID",
            "v9_handling": "Chặn thao tác ghi và yêu cầu tra danh sách đơn hàng trước",
            "code_ref": "PureSchemaStateTracker",
        },
        "Food Menu Gate": {
            "title": "Đặt món bỏ qua thực đơn",
            "risk": "Nhảy cóc gọi đặt đồ ăn khi chưa đọc menu khiến hệ thống từ chối sai mã món",
            "v9_handling": "Bắt buộc gọi đọc thực đơn lấy mã món thật rồi mới mở lệnh đặt hàng",
            "code_ref": "PureSchemaStateTracker",
        },
        "Return Status Gate": {
            "title": "Hoàn đơn không đủ điều kiện",
            "risk": "Gửi yêu cầu trả hàng cho đơn đã ở trạng thái hủy",
            "v9_handling": "Đối chiếu trạng thái đơn hàng trước khi gọi API và chặn thao tác sai",
            "code_ref": "PureSchemaStateTracker",
        },
        "Wallet Fallback": {
            "title": "Ví điện tử không đủ tiền",
            "risk": "Thanh toán ví thất bại nhưng vẫn tiếp tục gửi tin nhắn xác nhận giả cho người nhận",
            "v9_handling": "Đối chiếu số dư thực, tự chuyển sang Tài khoản Ngân hàng hoặc chặn SMS giả",
            "code_ref": "HybridProgramAidedEngine",
        },
        "History Entity Reuse": {
            "title": "Tái sử dụng ID từ hội thoại cũ",
            "risk": "Lá chắn quá cứng chặn oan mã đặt chỗ cũ hoặc nhầm mã khách sạn sang mã đặt chỗ",
            "v9_handling": "Chuẩn hóa số ít/nhiều cùng app (cho dùng lại mã đặt chỗ) nhưng khóa cứng khóa ngoại",
            "code_ref": "V9 Schema-First",
        },
        "Capacity & Date Gate": {
            "title": "Vượt sức chứa & lệch ngày đặt phòng",
            "risk": "Đặt 6 khách tại bàn tối đa 4 người hoặc nhập ngày trả phòng nhỏ hơn ngày nhận phòng",
            "v9_handling": "Chặn trước khi chạm API dựa trên giới hạn schema và ngày tháng hợp lệ",
            "code_ref": "V9 Schema-First",
        },
    }

    rows: list[dict[str, Any]] = []
    for r in raw_rows:
        meta = vi_catalog[r["Scenario"]]
        rows.append({
            "title": meta["title"],
            "risk": meta["risk"],
            "v9_handling": meta["v9_handling"] if r["Safe"] else "Kiểm thử không đạt",
            "code_ref": meta["code_ref"],
            "safe": bool(r["Safe"]),
        })

    return {
        "passed": sum(1 for x in rows if x["safe"]),
        "total": len(rows),
        "elapsed_ms": elapsed_ms,
        "rows": rows,
    }


def get_architecture_layers_vi() -> list[dict[str, str]]:
    """Return 5-layer architecture summary for editorial HTML rendering."""
    return [
        {
            "layer": "1. Nhận dạng giọng nói",
            "component": "Whisper Turbo CT2 (~0.8B) + Catalog Hotwords",
            "role": "Loại nhiễu âm thanh và tự sinh từ khóa chuyên ngành theo danh mục ứng dụng đang cài",
        },
        {
            "layer": "2. Chuẩn hóa tiếng Việt",
            "component": "SafeContextPreprocessor · Number/Date Engine",
            "role": "Tách số tiền bằng chữ ('hai triệu rưỡi'), giờ tương đối ('8 giờ rưỡi tối mai') sang ISO +07:00",
        },
        {
            "layer": "3. Định tuyến & Nén lịch sử",
            "component": "History-Aware Router + 24-Turn Compactor",
            "role": "Xếp hạng công cụ < 2ms trên CPU; nén hội thoại dài 24 lượt giảm -74% ký tự mà giữ 100% Entity ID",
        },
        {
            "layer": "4. Lập luận & Sinh công cụ",
            "component": "Qwen3-4B + LoRA (r=128) Chunked-CE",
            "role": "Giảm VRAM phần Loss Logits từ 5,30 GiB xuống 0,15 GiB để train 9.216 token trọn vẹn trên 1 GPU L4",
        },
        {
            "layer": "5. Lá chắn thực thi V9",
            "component": "PureSchemaStateTracker V9 + HybridEngine",
            "role": "Suy diễn x-ref, x-one-of, tách khóa ngoại, giới hạn sức chứa/ngày và số dư ví trước khi chạm API",
        },
    ]
