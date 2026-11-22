"""V9 Schema-First unit & regression tests (100% synthetic, zero competition data)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.benchmark_v9_suite import format_markdown_table, run_v9_benchmark_summary
from src.asr import build_asr_hotwords_prompt
from src.guardrail import PureSchemaStateTracker
from src.prompt_builder import format_history_block
from src.router import SemanticToolRouter


class TestV9SchemaGuardrails(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = {
            "stays.search": {
                "name": "stays.search",
                "kind": "read",
                "returns": "stays.stays[]",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
            "stays.book": {
                "name": "stays.book",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stay_id": {"type": "string", "x-ref": "stays.stays"},
                        "room_type": {"type": "string"},
                        "guests": {"type": "integer"},
                        "check_in": {"type": "string", "format": "date"},
                        "check_out": {"type": "string", "format": "date"},
                    },
                    "required": ["stay_id", "check_in", "check_out"],
                },
            },
            "stays.cancel": {
                "name": "stays.cancel",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"booking_id": {"type": "string", "x-ref": "stays.bookings"}},
                    "required": ["booking_id"],
                },
            },
            "clinic.book": {
                "name": "clinic.book",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doctor_id": {"type": "string", "x-ref": "clinic.doctors"},
                        "time": {"type": "string", "format": "date-time"},
                    },
                    "required": ["doctor_id", "time"],
                },
            },
            "clinic.reschedule": {
                "name": "clinic.reschedule",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {"type": "string", "x-ref": "clinic.appointments"},
                        "time": {"type": "string", "format": "date-time"},
                    },
                    "required": ["appointment_id", "time"],
                },
            },
            "shopping_list.add": {
                "name": "shopping_list.add",
                "kind": "write",
                "parameters": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
            "shopping_list.remove": {
                "name": "shopping_list.remove",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"item_id": {"type": "string", "x-ref": "shopping_list.items"}},
                    "required": ["item_id"],
                },
            },
            "grocery.add_to_cart": {
                "name": "grocery.add_to_cart",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"product_id": {"type": "string", "x-ref": "grocery.products"}, "quantity": {"type": "integer"}},
                    "required": ["product_id", "quantity"],
                },
            },
            "civic.submit": {
                "name": "civic.submit",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"application_id": {"type": "string", "x-ref": "civic.applications"}},
                    "required": ["application_id"],
                },
            },
            "fleet.dispatch": {
                "name": "fleet.dispatch",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {"vehicle_id": {"type": "string", "x-ref": "fleet.vehicles"}},
                    "required": ["vehicle_id"],
                },
            },
            "contacts.lookup_flat": {
                "name": "contacts.lookup_flat",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"phone": {"type": "string"}, "email": {"type": "string"}},
                    "x-one-of": ["phone", "email"],
                },
            },
            "contacts.lookup_nested": {
                "name": "contacts.lookup_nested",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "email": {"type": "string"},
                        "name": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "x-one-of": [["phone", "email"], ["name", "city"]],
                },
            },
            "events.export": {
                "name": "events.export",
                "kind": "read",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                    },
                },
            },
            "media.play": {
                "name": "media.play",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title_id": {"type": "string", "x-ref": "media.titles"},
                        "episode": {"type": "integer"},
                    },
                    "required": ["title_id"],
                },
            },
        }
        self.base_sample = {
            "id": "v9_unit",
            "context": {
                "current_time": "2026-12-01T08:00:00+07:00",
                "installed_apps": {
                    "stays": {"tools": ["stays.search", "stays.book", "stays.cancel"]},
                    "clinic": {"tools": ["clinic.book", "clinic.reschedule"]},
                    "shopping_list": {"tools": ["shopping_list.add", "shopping_list.remove"]},
                    "grocery": {"tools": ["grocery.add_to_cart"]},
                    "media": {"tools": ["media.play"]},
                },
            },
        }

    def test_singular_plural_and_generic_item_collection_matching(self) -> None:
        sample = {
            **self.base_sample,
            "history": [
                {"role": "user", "audio_text": "Đặt phòng và lịch khám"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "stays.book", "arguments": {}},
                        {"name": "clinic.book", "arguments": {}},
                        {"name": "shopping_list.add", "arguments": {"name": "Sữa"}},
                    ],
                    "tool_results": [
                        {"ok": True, "booking": {"id": "bk_101", "stay_id": "st_05"}},
                        {"ok": True, "appointment": {"id": "ap_202", "doctor_id": "doc_09"}},
                        {"ok": True, "item": {"id": "itm_303", "name": "Sữa"}},
                    ],
                },
            ],
        }
        tracker = PureSchemaStateTracker(sample, "Hủy đặt phòng và sửa lịch vừa tạo", self.catalog)
        self.assertIsNone(tracker.check_pre_execution("stays.cancel", {"booking_id": "bk_101"}))
        self.assertIsNone(
            tracker.check_pre_execution(
                "clinic.reschedule",
                {"appointment_id": "ap_202", "time": "2026-12-03T09:00:00+07:00"},
            )
        )
        self.assertIsNone(tracker.check_pre_execution("shopping_list.remove", {"item_id": "itm_303"}))

    def test_cross_app_lock_blocks_generic_and_plural_leaks(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Thêm đồ vào giỏ siêu thị", self.catalog)
        tracker.record_observation("pantry.list", {}, {"ok": True, "item": {"id": "pan_01", "name": "Muối"}})
        tracker.record_observation("pantry.all", {}, {"ok": True, "items": [{"id": "pan_02", "name": "Đường"}]})
        self.assertIsNotNone(tracker.check_pre_execution("grocery.add_to_cart", {"product_id": "pan_01", "quantity": 1}))
        self.assertIsNotNone(tracker.check_pre_execution("grocery.add_to_cart", {"product_id": "pan_02", "quantity": 1}))

    def test_primary_id_vs_foreign_key_separation(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Hủy đặt phòng", self.catalog)
        tracker.record_observation(
            "stays.book",
            {},
            {"ok": True, "booking": {"id": "bk_901", "stay_id": "st_05"}},
        )
        self.assertIsNone(tracker.check_pre_execution("stays.cancel", {"booking_id": "bk_901"}))
        err_fk = tracker.check_pre_execution("stays.cancel", {"booking_id": "st_05"})
        self.assertIsNotNone(err_fk)

    def test_dynamic_schema_id_discovery_for_unseen_keys(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Nộp hồ sơ và điều xe", self.catalog)
        tracker.record_observation("civic.list", {}, {"ok": True, "applications": [{"application_id": "app_77"}]})
        tracker.record_observation("fleet.list", {}, {"ok": True, "vehicles": [{"vehicle_id": "veh_88"}]})
        self.assertIsNone(tracker.check_pre_execution("civic.submit", {"application_id": "app_77"}))
        self.assertIsNone(tracker.check_pre_execution("fleet.dispatch", {"vehicle_id": "veh_88"}))
        # Cross-collection mismatch should still be caught
        self.assertIsNotNone(tracker.check_pre_execution("civic.submit", {"application_id": "veh_88"}))

    def test_generic_x_one_of_precheck_flat_and_nested(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Tìm liên hệ", self.catalog)
        self.assertIsNotNone(tracker.check_pre_execution("contacts.lookup_flat", {}))
        self.assertIsNone(tracker.check_pre_execution("contacts.lookup_flat", {"email": "a@b.com"}))

        # Nested requires at least one from each group
        self.assertIsNotNone(tracker.check_pre_execution("contacts.lookup_nested", {"phone": "090"}))
        self.assertIsNone(tracker.check_pre_execution("contacts.lookup_nested", {"phone": "090", "city": "HCM"}))

    def test_guest_capacity_max_guests_gate(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Đặt phòng 6 người", self.catalog)
        tracker.record_observation(
            "stays.search",
            {},
            {"ok": True, "stays": [{"id": "st_01", "available": True, "max_guests": 4}]},
        )
        err = tracker.check_pre_execution(
            "stays.book",
            {"stay_id": "st_01", "guests": 6, "check_in": "2026-12-10", "check_out": "2026-12-12"},
        )
        self.assertIsNotNone(err)
        self.assertIsNone(
            tracker.check_pre_execution(
                "stays.book",
                {"stay_id": "st_01", "guests": 4, "check_in": "2026-12-10", "check_out": "2026-12-12"},
            )
        )

    def test_temporal_date_ranges_and_iso_timezone_gate(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Kiểm tra ngày và múi giờ", self.catalog)
        tracker.record_observation("stays.search", {}, {"ok": True, "stays": [{"id": "st_01"}]})
        tracker.record_observation("clinic.list", {}, {"ok": True, "doctors": [{"id": "doc_01"}]})

        # check_in >= check_out blocked
        self.assertIsNotNone(
            tracker.check_pre_execution(
                "stays.book",
                {"stay_id": "st_01", "check_in": "2026-12-12", "check_out": "2026-12-10"},
            )
        )
        self.assertIsNotNone(
            tracker.check_pre_execution(
                "stays.book",
                {"stay_id": "st_01", "check_in": "2026-12-12", "check_out": "2026-12-12"},
            )
        )
        # start_date > end_date blocked
        self.assertIsNotNone(
            tracker.check_pre_execution(
                "events.export",
                {"start_date": "2026-12-20", "end_date": "2026-12-10"},
            )
        )
        # Missing timezone offset on date-time blocked
        self.assertIsNotNone(
            tracker.check_pre_execution(
                "clinic.book",
                {"doctor_id": "doc_01", "time": "2026-12-12T09:00:00"},
            )
        )
        self.assertIsNone(
            tracker.check_pre_execution(
                "clinic.book",
                {"doctor_id": "doc_01", "time": "2026-12-12T09:00:00+07:00"},
            )
        )

    def test_availability_room_type_slots_hours_and_episode_gates(self) -> None:
        tracker = PureSchemaStateTracker(self.base_sample, "Kiểm tra ràng buộc entity", self.catalog)
        tracker.record_observation(
            "stays.search",
            {},
            {
                "ok": True,
                "stays": [
                    {"id": "st_off", "available": False},
                    {"id": "st_rooms", "available": True, "room_types": ["Standard", "Deluxe"]},
                ],
            },
        )
        tracker.record_observation(
            "clinic.list",
            {},
            {
                "ok": True,
                "doctors": [{"id": "doc_slots", "slots": ["09:00", "14:30"], "open": "08:00", "close": "17:00"}],
            },
        )
        tracker.record_observation(
            "media.search",
            {},
            {"ok": True, "titles": [{"id": "ser_01", "episodes": 12}]},
        )

        self.assertIsNotNone(
            tracker.check_pre_execution(
                "stays.book", {"stay_id": "st_off", "check_in": "2026-12-01", "check_out": "2026-12-02"}
            )
        )
        self.assertIsNotNone(
            tracker.check_pre_execution(
                "stays.book",
                {"stay_id": "st_rooms", "room_type": "Penthouse", "check_in": "2026-12-01", "check_out": "2026-12-02"},
            )
        )
        self.assertIsNone(
            tracker.check_pre_execution(
                "stays.book",
                {"stay_id": "st_rooms", "room_type": "deluxe", "check_in": "2026-12-01", "check_out": "2026-12-02"},
            )
        )
        # Slot & open/close hours
        self.assertIsNotNone(
            tracker.check_pre_execution("clinic.book", {"doctor_id": "doc_slots", "time": "2026-12-02T11:00:00+07:00"})
        )
        self.assertIsNotNone(
            tracker.check_pre_execution("clinic.book", {"doctor_id": "doc_slots", "time": "2026-12-02T20:00:00+07:00"})
        )
        self.assertIsNone(
            tracker.check_pre_execution("clinic.book", {"doctor_id": "doc_slots", "time": "2026-12-02T14:30:00+07:00"})
        )
        # Episode bounds
        self.assertIsNotNone(tracker.check_pre_execution("media.play", {"title_id": "ser_01", "episode": 15}))
        self.assertIsNone(tracker.check_pre_execution("media.play", {"title_id": "ser_01", "episode": 12}))

    def test_catalog_driven_dynamic_asr_hotwords_zero_shot(self) -> None:
        unseen_apps = {
            "pet_care": {
                "id": "pet_care",
                "name_vi": "Chăm sóc thú cưng",
                "description": "Đặt lịch spa tắm tỉa, khám thú y",
                "tools": ["pet_care.book"],
            }
        }
        unseen_tools = {
            "pet_care.book": {
                "name": "pet_care.book",
                "kind": "write",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "service": {"type": "string", "enum": ["cắt tỉa lông", "internal_raw_code"]},
                    },
                },
            }
        }
        sample = {
            "id": "hw_unseen",
            "context": {
                "user_profile": {"city": "Đà Nẵng"},
                "installed_apps": {"pet_care": {"tools": ["pet_care.book"]}},
            },
            "history": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "pet_care.list", "arguments": {}}],
                    "tool_results": [{"ok": True, "pet": {"id": "p_1", "name": "Mèo Mướp"}}],
                }
            ],
        }
        prompt = build_asr_hotwords_prompt(sample, catalog_apps_map=unseen_apps, catalog_tools_map=unseen_tools)
        self.assertIn("Chăm sóc thú cưng", prompt)
        self.assertIn("cắt tỉa lông", prompt)
        self.assertIn("Mèo Mướp", prompt)
        self.assertNotIn("internal_raw_code", prompt)

    def test_history_aware_routing_elliptic_vs_explicit_switch(self) -> None:
        router = SemanticToolRouter(top_k_recommended=2)
        installed = {
            "stays": {"tools": ["stays.search", "stays.book", "stays.cancel"]},
            "clinic": {"tools": ["clinic.book", "clinic.reschedule"]},
            "grocery": {"tools": ["grocery.add_to_cart"]},
            "media": {"tools": ["media.play"]},
        }
        descs = {
            "stays": "Đặt phòng khách sạn nghỉ dưỡng",
            "clinic": "Đặt lịch khám bác sĩ phòng khám",
            "grocery": "Đi chợ mua thực phẩm siêu thị",
            "media": "Xem phim video trực tuyến",
        }
        sample_hist = {
            "id": "route_hist",
            "context": {"installed_apps": installed},
            "history": [
                {"role": "user", "audio_text": "Tìm khách sạn"},
                {"role": "assistant", "tool_calls": [{"name": "stays.book", "arguments": {}}], "tool_results": [{"ok": True}]},
            ],
        }
        # Elliptic follow-up promotes recent active app `stays`
        rec_elliptic, _ = router.route_sample_apps(
            "Hủy cái vừa đặt lúc nãy giúp mình", sample_hist, self.catalog, installed_app_descriptions=descs
        )
        self.assertEqual(rec_elliptic[0], "stays")

        # Explicit topic switch to grocery/clinic still wins cleanly
        rec_switch, _ = router.route_sample_apps(
            "Đi chợ mua thực phẩm siêu thị nhé", sample_hist, self.catalog, installed_app_descriptions=descs
        )
        self.assertEqual(rec_switch[0], "grocery")

    def test_24_turn_compaction_preserves_100_pct_ids_and_recent_verbatim(self) -> None:
        history = []
        all_ids = []
        for t in range(1, 25):
            history.append({"role": "user", "audio_text": f"Truy vấn lượt {t}"})
            items = [
                {
                    "id": f"ent_{t:02d}_{i:02d}",
                    "name": f"Entity {i}",
                    "status": "confirmed",
                    "description": "Đoạn mô tả rất dài để kiểm tra khả năng nén ngữ cảnh của hệ thống V9.",
                    "reviews": [{"text": "Đánh giá chi tiết dài dòng không cần giữ ở các lượt cũ"}],
                }
                for i in range(10)
            ]
            all_ids.extend(it["id"] for it in items)
            history.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "stays.search", "arguments": {"turn": t}}],
                    "tool_results": [{"ok": True, "stays": items}],
                }
            )

        raw = format_history_block(history, compact_older_than=0)
        compacted = format_history_block(history)
        reduction = 1.0 - (len(compacted) / len(raw))
        self.assertGreaterEqual(reduction, 0.35)
        for eid in all_ids:
            self.assertIn(eid, compacted)
        # Recent turn 24 retains verbatim details
        self.assertIn("Đánh giá chi tiết dài dòng", compacted.split("[Lượt 24 - Người dùng]")[-1])

    def test_benchmark_v9_suite_smoke(self) -> None:
        summary = run_v9_benchmark_summary()
        self.assertEqual(summary["unsafe_interception"]["v9_pct"], 100.0)
        self.assertEqual(summary["false_positive_history_blocks"]["v9_fp_pct"], 0.0)
        self.assertEqual(summary["compaction_24_turn"]["entity_id_retention_pct"], 100.0)
        self.assertEqual(summary["zero_shot_hotwords"]["v9_pct"], 100.0)
        table = format_markdown_table(summary)
        self.assertIn("V9 Schema-First", table)


if __name__ == "__main__":
    unittest.main()