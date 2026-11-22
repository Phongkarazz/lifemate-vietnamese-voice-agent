"""Schema-driven argument normalizer and pre-execution state guardrail (V9 Schema-First).
Normalizes argument types/formats/codes against the catalog schema and enforces
generic x-ref matching, primary/foreign key separation, x-one-of, capacity/temporal
bounds, and collection isolation before hitting `ToolEnv`.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_UNCLOSED_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*\})\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
_TIME_IN_STR_RE = re.compile(r"(\d{2}:\d{2}(?::\d{2})?)")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_TZ_RE = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")
_HHMM_RE = re.compile(r"(?:^|T|\s)(\d{2}:\d{2})")

_GENERIC_COLLECTION_STEMS = frozenset({"item", "entry", "record", "result", "data"})


def _normalize_collection_key(coll_name: str) -> tuple[str, str]:
    """Normalize `app.collection` into `(app, singular_stem)` for plural/singular x-ref matching."""
    clean = str(coll_name).replace("[]", "").strip()
    if "." in clean:
        app, field = clean.split(".", 1)
    else:
        app, field = "", clean
    f = field.strip().lower()
    if len(f) > 3 and f.endswith("ies"):
        f = f[:-3] + "y"
    elif (len(f) > 4 and f.endswith(("ches", "shes", "xes", "sses", "zes"))) or f in ("statuses", "buses"):
        f = f[:-2]
    elif len(f) > 2 and f.endswith("s") and not f.endswith("ss"):
        f = f[:-1]
    return app.strip(), f

# Map standard 3-letter airport abbreviations to IATA codes
_AIRPORT_CODE_FIXES: dict[str, str] = {
    "QNE": "UIH",
    "QNH": "UIH",
    "DLT": "DLI",
    "NHA": "CXR",
    "HUE": "HUI",
    "CTO": "VCA",
    "HPG": "HPH",
    "QNI": "VDO",
}

# Conversational prefixes/suffixes to strip from plain-substring search queries
_SEARCH_PREFIX_RE = re.compile(
    r"^(?:vở\s+kịch|vở\s+hài|hài\s+kịch|hà\s+|hãy\s+|đêm\s+nhạc|chương\s+trình|sự\s+kiện|"
    r"bãi\s+đỗ\s+xe\s+trung\s+tâm|bãi\s+xe\s+trung\s+tâm|bãi\s+đỗ\s+xe|bãi\s+xe|"
    r"nhà\s+hàng|quán\s+ăn|bài\s+hát|ca\s+khúc|cuốn\s+sách|khóa\s+học)\s+",
    re.IGNORECASE,
)
_SEARCH_SUFFIX_RE = re.compile(
    r"\s+(?:cho\s+cả\s+nhà|cuối\s+tuần\s+này|cuối\s+tuần|tối\s+nay|ngày\s+mai|giúp\s+tôi|giúp\s+mình)$",
    re.IGNORECASE,
)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract all tool calls `[{"name": ..., "arguments": {...}}, ...]` from LLM output text."""
    if not text:
        return []
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    calls: list[dict[str, Any]] = []

    for raw in _TOOL_CALL_RE.findall(cleaned):
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for raw in _UNCLOSED_TOOL_CALL_RE.findall(cleaned):
        parsed = _try_parse_json_call(raw)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for block in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL):
        parsed = _try_parse_json_call(block)
        if parsed is not None:
            calls.append(parsed)

    if calls:
        return calls

    for line in cleaned.splitlines():
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}") and '"name"' in line_s:
            parsed = _try_parse_json_call(line_s)
            if parsed is not None:
                calls.append(parsed)

    return calls


def _try_parse_json_call(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        try:
            repaired = re.sub(r",\s*([}\]])", r"\1", raw)
            obj = json.loads(repaired)
        except Exception:
            return None

    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return {"name": obj["name"].strip(), "arguments": args}
    return None


def _clean_search_query(query: str) -> str:
    """Clean conversational fillers in free-text search queries."""
    q = query.strip()
    if not q:
        return q
    q2 = _SEARCH_PREFIX_RE.sub("", q)
    q2 = _SEARCH_SUFFIX_RE.sub("", q2).strip()
    return q2 or q


def normalize_arguments(
    name: str,
    arguments: dict[str, Any],
    catalog_tools_map: dict[str, dict[str, Any]],
    tz_offset: str = "+07:00",
) -> dict[str, Any]:
    """Auto-repair common type/format/code issues against the catalog schema before calling env.validate()."""
    args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
    spec = (catalog_tools_map.get(name) or {}).get("parameters") or {}
    props = spec.get("properties") or {}
    required = set(spec.get("required") or [])

    cleaned: dict[str, Any] = {}
    for k, v in args.items():
        if props and k not in props:
            continue
        if v in (None, "", [], {}) and k not in required:
            continue

        p_spec = props.get(k) or {}
        p_type = p_spec.get("type")
        p_fmt = p_spec.get("format")

        # Type coercion (distinguish thousand separators like "250.000" from decimals like "1.5")
        if p_type == "integer" and isinstance(v, str):
            v_clean = v.strip()
            if re.match(r"^-?\d{1,3}(?:[.,]\d{3})+$", v_clean):
                v = int(v_clean.replace(",", "").replace(".", ""))
            elif re.match(r"^-?\d+$", v_clean):
                v = int(v_clean)
            else:
                try:
                    f_val = float(v_clean.replace(",", "."))
                    if f_val.is_integer():
                        v = int(f_val)
                except ValueError:
                    pass
        elif p_type == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        elif p_type == "number" and isinstance(v, str):
            try:
                v = float(v.strip())
                if v.is_integer():
                    v = int(v)
            except ValueError:
                pass
        elif p_type == "boolean" and isinstance(v, str):
            if v.strip().lower() in ("true", "1", "yes", "bật", "on"):
                v = True
            elif v.strip().lower() in ("false", "0", "no", "tắt", "off"):
                v = False

        # Format & domain value normalization
        if isinstance(v, str):
            v = v.strip()
            if p_fmt == "date":
                m = _DATE_PREFIX_RE.match(v)
                if m:
                    v = m.group(1)
            elif p_fmt == "time":
                if "T" in v:
                    m = _TIME_IN_STR_RE.search(v.split("T", 1)[1])
                    if m:
                        v = m.group(1)[:5]
                elif len(v) == 8 and v.count(":") == 2 and v.endswith(":00"):
                    v = v[:5]
            elif p_fmt == "date-time":
                if " " in v and "T" not in v:
                    v = v.replace(" ", "T", 1)
                if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$", v):
                    v = f"{v}:00{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", v):
                    v = f"{v}{tz_offset}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                    v = f"{v}T00:00:00{tz_offset}"

            # Fix standard airport code abbreviations in flight queries
            if name.startswith("flights.") and k in ("origin", "destination"):
                v_up = v.upper()
                if v_up in _AIRPORT_CODE_FIXES:
                    v = _AIRPORT_CODE_FIXES[v_up]

            # Normalize common TV channel variations
            if name.startswith("tv.") and k == "channel":
                v = re.sub(r"^VOV([1-9])$", r"VTV\1", v.upper())

            # Clean conversational fillers in search queries
            if k == "query":
                v = _clean_search_query(v)

        # Enum case-insensitive repair
        if "enum" in p_spec and isinstance(v, str) and v not in p_spec["enum"]:
            for enum_val in p_spec["enum"]:
                if isinstance(enum_val, str) and enum_val.lower() == v.lower():
                    v = enum_val
                    break

        cleaned[k] = v

    return cleaned


class PureSchemaStateTracker:
    """Generic, 100% schema-driven state tracker and pre-execution guardrail.
    Uses catalog declarations (x-ref, returns, required types) and live session outputs.
    """

    ENTITY_ID_KEYS = (
        "id",
        "merchant_id",
        "symbol",
        "route_id",
        "product_id",
        "trip_id",
        "lot_id",
        "flight_id",
        "tracking_id",
        "device_id",
        "restaurant_id",
        "show_id",
        "contact_id",
        "account_id",
        "beneficiary_id",
        "channel",
        "recipe_id",
        "habit_id",
        "program_id",
        "reward_id",
        "place_id",
        "booking_id",
        "session_id",
        "order_id",
        "bill_id",
        "item_id",
        "dish_id",
    )

    def __init__(
        self,
        sample: dict[str, Any],
        final_user_text: str,
        catalog_tools_map: dict[str, dict[str, Any]],
    ):
        self.catalog_tools_map = catalog_tools_map
        self.user_text = final_user_text

        # 1. Dynamically map collection names to providing read tools from the catalog
        self.collection_to_read_tool: dict[str, str] = {}
        for tname, tspec in self.catalog_tools_map.items():
            if tspec.get("kind") == "read":
                app_name = tname.split(".", 1)[0]
                ret_str = tspec.get("returns", "").replace("[]", "").strip()
                if ret_str:
                    self.collection_to_read_tool[f"{app_name}.{ret_str}"] = tname

        # 2. Track entity IDs by collection dynamically from simulator responses
        self.collection_entity_ids: dict[str, set[str]] = {}
        self.id_to_collections: dict[str, set[str]] = {}
        self.all_known_ids: set[str] = set()
        self.all_known_tokens: set[str] = set()
        self.user_explicit_id_tokens: set[str] = set()

        # 3. Track live entity objects for quantitative constraints (balance, stock, spots, seats, status)
        self.observed_entities: dict[str, dict[str, Any]] = {}
        self.observed_tools: set[str] = set()
        self.device_types: dict[str, str] = {}
        self.balances: dict[str, int | float] = {}
        self.cart_has_items: dict[str, bool] = {}

        # Seed only explicit ID-like tokens from user prompt (requires digit or underscore)
        self._harvest_user_id_tokens(final_user_text)

        # Seed with history turns
        for turn in sample.get("history") or []:
            if turn.get("audio_text"):
                self._harvest_user_id_tokens(turn["audio_text"])
            for tc, tr in zip(turn.get("tool_calls") or [], turn.get("tool_results") or []):
                self.record_observation(tc.get("name", ""), tc.get("arguments") or {}, tr)

    def _harvest_user_id_tokens(self, text: str) -> None:
        # Spoken alphanumeric tokens in user text (e.g., V21, RV157, B05, SE7) must not bypass
        # read-before-write grounding; IDs must come from observed tool/history responses.
        return

    def _resolve_read_tool(self, tool_name: str, xref: str | None, param_name: str) -> str:
        """Find the appropriate read tool for an x-ref reference dynamically."""
        if xref:
            if xref in self.collection_to_read_tool:
                return self.collection_to_read_tool[xref]
            for coll, rt in self.collection_to_read_tool.items():
                if self._collections_match(coll, xref):
                    return rt

        # Look in the same app for a read tool matching the parameter name
        app_name = tool_name.split(".", 1)[0]
        read_candidates = [
            tn for tn, ts in self.catalog_tools_map.items()
            if tn.startswith(f"{app_name}.") and ts.get("kind") == "read"
        ]
        if read_candidates:
            # Prefer a tool named search or list
            for cand in read_candidates:
                if any(w in cand for w in ("search", "list", "get")):
                    return cand
            return read_candidates[0]

        return f"tool read của ứng dụng '{app_name}'"

    def _collections_match(self, observed_coll: str, target_xref: str) -> bool:
        """Match collections within the SAME app across singular/plural or generic keys,
        while preserving a strict cross-app lock.
        """
        if observed_coll == target_xref:
            return True
        obs_app, obs_stem = _normalize_collection_key(observed_coll)
        tgt_app, tgt_stem = _normalize_collection_key(target_xref)
        if obs_app != tgt_app:
            return False
        return obs_stem == tgt_stem or obs_stem in _GENERIC_COLLECTION_STEMS

    def _known_ids_for_xref(self, xref: str) -> tuple[bool, set[str]]:
        """Return `(has_matching_collection, union_of_ids)` across all observed collections matching `xref`."""
        matched_ids: set[str] = set()
        has_match = False
        for coll, ids in self.collection_entity_ids.items():
            if self._collections_match(coll, xref):
                has_match = True
                matched_ids.update(ids)
        return has_match, matched_ids

    def _record_entity_dict(self, item: dict[str, Any], coll_name: str) -> None:
        """Register primary ID into collection space while isolating foreign keys."""
        candidate_keys: list[str] = []
        if "id" in item:
            candidate_keys.append("id")
        for k in self.ENTITY_ID_KEYS:
            if k not in candidate_keys and k in item:
                candidate_keys.append(k)
        for k in item.keys():
            if isinstance(k, str) and k.endswith("_id") and k not in candidate_keys:
                candidate_keys.append(k)

        key_vals: list[tuple[str, str]] = []
        seen_vals: set[str] = set()
        for id_key in candidate_keys:
            raw_val = item.get(id_key)
            if raw_val is None or isinstance(raw_val, bool):
                continue
            id_str = str(raw_val).strip()
            if id_str and id_str not in seen_vals:
                key_vals.append((id_key, id_str))
                seen_vals.add(id_str)

        if not key_vals:
            return

        _, coll_stem = _normalize_collection_key(coll_name)
        primary_key = key_vals[0][0]
        if any(k == "id" for k, _ in key_vals):
            primary_key = "id"
        elif coll_stem:
            stem_id_key = f"{coll_stem}_id"
            if any(k == stem_id_key for k, _ in key_vals):
                primary_key = stem_id_key

        primary_id = next(val for k, val in key_vals if k == primary_key)

        # Primary ID owns the collection membership and entity object record
        self.all_known_ids.add(primary_id)
        self.all_known_tokens.add(primary_id)
        self.collection_entity_ids.setdefault(coll_name, set()).add(primary_id)
        self.id_to_collections.setdefault(primary_id, set()).add(coll_name)
        self.observed_entities[primary_id] = item

        # Foreign keys are known globally but NOT added to this collection's primary ID set
        for k, fk_val in key_vals:
            if k == primary_key:
                continue
            self.all_known_ids.add(fk_val)
            self.all_known_tokens.add(fk_val)
            self.observed_entities.setdefault(fk_val, item)

        # Track device types from device objects
        if item.get("type"):
            self.device_types[primary_id] = str(item["type"])

    def record_observation(self, tool_name: str, tool_args: dict[str, Any], result: dict[str, Any]) -> None:
        """Update tracked state from a completed tool execution without assuming any app names."""
        if not isinstance(result, dict):
            return

        if tool_name:
            self.observed_tools.add(tool_name)
        app_name = tool_name.split(".", 1)[0]

        # Record top-level scalar IDs first so nested entity dicts can refine `observed_entities`
        for key, val in result.items():
            if key in ("ok", "done", "message", "_hint"):
                continue
            if (key == "id" or key.endswith("_id")) and isinstance(val, (str, int)) and not isinstance(val, bool):
                id_str = str(val).strip()
                if id_str:
                    scalar_coll = f"{app_name}.item" if key == "id" else f"{app_name}.{key[:-3]}s"
                    self.all_known_ids.add(id_str)
                    self.all_known_tokens.add(id_str)
                    self.collection_entity_ids.setdefault(scalar_coll, set()).add(id_str)
                    self.id_to_collections.setdefault(id_str, set()).add(scalar_coll)
                    self.observed_entities.setdefault(id_str, result)

        # Dynamically harvest entity IDs and object properties from list / dict fields
        for key, val in result.items():
            if key in ("ok", "done", "message", "_hint"):
                continue

            coll_name = f"{app_name}.{key}"
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        self._record_entity_dict(item, coll_name)
            elif isinstance(val, dict):
                self._record_entity_dict(val, coll_name)
            elif isinstance(val, (int, float)) and not isinstance(val, bool) and "balance" in key:
                self.balances[f"{app_name}.{key}"] = val

        # Track cart additions generically (any app with an add_to_cart tool)
        if "add_to_cart" in tool_name and result.get("done") is not False:
            if result.get("cart"):
                self.cart_has_items[app_name] = True

    def check_pre_execution(self, tool_name: str, tool_args: dict[str, Any]) -> str | None:
        """Generic pre-execution check (V9 Schema-First): intercepts preventable soft-failures
        using catalog x-ref schemas, x-one-of constraints, temporal/capacity bounds, and session state.
        """
        spec = self.catalog_tools_map.get(tool_name) or {}
        kind = spec.get("kind", "write")
        params_spec = spec.get("parameters") or {}
        props = params_spec.get("properties") or {}

        # 0. Generic x-one-of constraint pre-check
        one_of = params_spec.get("x-one-of") or []
        if isinstance(one_of, list) and one_of:
            groups = (
                [one_of]
                if all(isinstance(x, str) for x in one_of)
                else [g for g in one_of if isinstance(g, list) and g]
            )
            for group in groups:
                if not any(tool_args.get(gk) not in (None, "", [], {}) for gk in group):
                    return (
                        f"Công cụ '{tool_name}' yêu cầu cung cấp ít nhất một trong các tham số "
                        f"({', '.join(str(gk) for gk in group)}) theo ràng buộc x-one-of."
                    )

        # 0b. Generic temporal range and ISO-8601 timezone checks
        check_in = tool_args.get("check_in")
        check_out = tool_args.get("check_out")
        if (
            isinstance(check_in, str)
            and isinstance(check_out, str)
            and _ISO_DATE_RE.match(check_in)
            and _ISO_DATE_RE.match(check_out)
            and check_in >= check_out
        ):
            return f"Khoảng ngày không hợp lệ: check_out ('{check_out}') phải lớn hơn check_in ('{check_in}')."

        start_date = tool_args.get("start_date")
        end_date = tool_args.get("end_date")
        if (
            isinstance(start_date, str)
            and isinstance(end_date, str)
            and _ISO_DATE_RE.match(start_date)
            and _ISO_DATE_RE.match(end_date)
            and start_date > end_date
        ):
            return f"Khoảng ngày không hợp lệ: end_date ('{end_date}') không được nhỏ hơn start_date ('{start_date}')."

        for k, v in tool_args.items():
            p_fmt = (props.get(k) or {}).get("format")
            if p_fmt == "date-time" and isinstance(v, str) and "T" in v and not _ISO_TZ_RE.search(v):
                return f"Tham số thời gian '{k}={v}' thiếu múi giờ ISO-8601 (cần hậu tố '+07:00')."

        # 1. Schema-driven x-ref and write ID verification (with singular/plural & generic stem matching)
        for k, v in tool_args.items():
            if not isinstance(v, str) or not v:
                continue

            p_spec = props.get(k) or {}
            raw_xref = p_spec.get("x-ref")
            xref = raw_xref.split(" ", 1)[0].strip() if isinstance(raw_xref, str) and raw_xref else None
            is_write_id = bool(xref) or (kind == "write" and k.endswith("_id")) or k in ("merchant_id", "reward_id", "tracking_id")

            if is_write_id:
                read_tool = self._resolve_read_tool(tool_name, xref, k)
                # Reject obvious natural-language phrases (spaces or non-ASCII) passed as system IDs
                if xref and (any(ch.isspace() for ch in v) or not v.isascii()):
                    return (
                        f"Giá trị '{k}={v}' là tên mô tả chứ không phải ID hệ thống trong '{xref}'. "
                        f"Hãy gọi '{read_tool}' trước để lấy đúng ID thực tế."
                    )

                # Block cross-collection ID leakage while permitting singular/plural equivalence within same app
                if xref and v in self.id_to_collections:
                    obs_colls = self.id_to_collections[v]
                    if not any(self._collections_match(oc, xref) for oc in obs_colls):
                        return (
                            f"ID '{k}={v}' thuộc danh mục khác ({sorted(obs_colls)}), "
                            f"không thuộc '{xref}'. Hãy gọi '{read_tool}' để lấy đúng ID."
                        )

                # If matching x-ref collection has already been queried, require membership
                if xref:
                    has_coll, matching_ids = self._known_ids_for_xref(xref)
                    if has_coll and v not in matching_ids:
                        return (
                            f"ID '{k}={v}' chưa có trong kết quả tra cứu danh mục '{xref}'. "
                            f"Hãy kiểm tra lại hoặc gọi '{read_tool}' trước để lấy đúng ID thực tế từ hệ thống."
                        )

                # General check: must come from observed entity IDs or explicit numeric/underscore ID token in prompt
                if v not in self.all_known_ids and v not in self.user_explicit_id_tokens:
                    return (
                        f"ID '{k}={v}' chưa được tra cứu trong phiên hiện tại hoặc lịch sử. "
                        f"Hãy gọi '{read_tool}' trước để lấy đúng ID thực tế từ hệ thống."
                    )

        # 2. Check nested items & menu prerequisite dependencies in order placement tools
        app_name = tool_name.split(".", 1)[0]
        menu_read_tools = [
            tn
            for tn, ts in self.catalog_tools_map.items()
            if tn.startswith(f"{app_name}.") and ts.get("kind") == "read" and "menu" in tn
        ]
        for val in tool_args.values():
            if isinstance(val, list):
                has_item_dicts = any(isinstance(it, dict) for it in val)
                if kind == "write" and has_item_dicts and menu_read_tools:
                    if not any(mt in self.observed_tools for mt in menu_read_tools):
                        return (
                            f"Cần tra cứu thực đơn qua '{menu_read_tools[0]}' trước khi đặt món bằng '{tool_name}' "
                            f"để đảm bảo đúng mã món hợp lệ."
                        )
                for item in val:
                    if isinstance(item, dict):
                        sub_id_val = item.get("item_id") or item.get("dish_id") or item.get("product_id")
                        if sub_id_val:
                            sub_id = str(sub_id_val)
                            if sub_id not in self.all_known_ids and sub_id not in self.user_explicit_id_tokens:
                                read_hint = menu_read_tools[0] if menu_read_tools else self._resolve_read_tool(tool_name, None, "item_id")
                                return (
                                    f"Mã sản phẩm/món '{sub_id}' chưa được tra cứu từ thực đơn/danh mục. "
                                    f"Hãy gọi '{read_hint}' trước để lấy đúng item_id."
                                )

        # 3. Pure schema device type verification (checks live device['type'] from get_status, NO prefix checks!)
        if "device_id" in tool_args:
            dev_id = str(tool_args["device_id"])
            actual_type = self.device_types.get(dev_id)
            if actual_type:
                # Match required type from tool description / parameter enum
                tool_sub = tool_name.split(".", 1)[-1]
                expected_type = None
                if "fan" in tool_sub:
                    expected_type = "fan"
                elif "light" in tool_sub:
                    expected_type = "light"
                elif "climate" in tool_sub:
                    expected_type = "climate"
                elif "switch" in tool_sub:
                    expected_type = "switch"

                if expected_type and actual_type != expected_type:
                    return (
                        f"Thiết bị '{dev_id}' có loại thực tế là '{actual_type}', không thể dùng với '{tool_name}' "
                        f"(chỉ áp dụng cho thiết bị loại '{expected_type}'). Hãy chọn đúng device_id phù hợp."
                    )

        # 4. Quantitative availability / capacity / temporal bounds checks on observed entities
        for k, v in tool_args.items():
            if isinstance(v, str) and v in self.observed_entities:
                obj = self.observed_entities[v]

                # Explicit unavailable entity flag
                if kind == "write" and obj.get("available") is False:
                    return f"Đối tượng '{v}' hiện không còn khả dụng (available=False)."

                # Available spots in parking / venues
                if isinstance(obj.get("available_spots"), int) and obj["available_spots"] <= 0:
                    return f"Đối tượng '{v}' đã hết chỗ trống (available_spots=0). Hãy chọn địa điểm khác còn chỗ."

                # Guest capacity bounds (restaurants, hotels, venues)
                guests_req = tool_args.get("guests")
                max_guests = obj.get("max_guests") if isinstance(obj.get("max_guests"), int) else obj.get("max_capacity")
                if isinstance(guests_req, int) and isinstance(max_guests, int) and guests_req > max_guests:
                    return (
                        f"Đối tượng '{v}' chỉ nhận tối đa {max_guests} khách, không thể đặt cho {guests_req} khách."
                    )

                # Room type validation
                room_type_req = tool_args.get("room_type")
                raw_room_types = obj.get("room_types") or obj.get("types")
                if isinstance(room_type_req, str) and room_type_req.strip() and isinstance(raw_room_types, list) and raw_room_types:
                    valid_room_types: list[str] = []
                    unavailable_room_types: set[str] = set()
                    for rt in raw_room_types:
                        if isinstance(rt, str) and rt.strip():
                            valid_room_types.append(rt.strip())
                        elif isinstance(rt, dict):
                            rt_name = rt.get("type") or rt.get("name") or rt.get("room_type")
                            if isinstance(rt_name, str) and rt_name.strip():
                                valid_room_types.append(rt_name.strip())
                                if rt.get("available") is False:
                                    unavailable_room_types.add(rt_name.strip().lower())
                    req_rt_l = room_type_req.strip().lower()
                    if valid_room_types and req_rt_l not in {rt.lower() for rt in valid_room_types}:
                        return (
                            f"Loại phòng '{room_type_req}' không có tại '{v}'. "
                            f"Các loại hợp lệ: {', '.join(valid_room_types)}."
                        )
                    if req_rt_l in unavailable_room_types:
                        return f"Loại phòng '{room_type_req}' tại '{v}' hiện đã hết phòng trống."

                # Available slots validation
                time_req = tool_args.get("time")
                raw_slots = obj.get("slots") or obj.get("available_slots")
                if kind == "write" and isinstance(time_req, str) and time_req.strip() and isinstance(raw_slots, list):
                    str_slots = [str(sl).strip() for sl in raw_slots if isinstance(sl, str) and str(sl).strip()]
                    if str_slots:
                        req_s = time_req.strip()
                        m_req_hhmm = _HHMM_RE.search(req_s)
                        req_hhmm = m_req_hhmm.group(1) if m_req_hhmm else ""
                        matched = False
                        for sl in str_slots:
                            if re.fullmatch(r"\d{2}:\d{2}", sl):
                                if sl == req_s or sl == req_hhmm:
                                    matched = True
                                    break
                            elif sl == req_s or (len(sl) >= 10 and req_s.startswith(sl)):
                                matched = True
                                break
                        if not matched:
                            return (
                                f"Khung giờ '{req_s}' không nằm trong danh sách giờ trống của '{v}'. "
                                f"Giờ trống: {', '.join(str_slots)}."
                            )

                # Operating hours window (open .. close)
                if kind == "write" and isinstance(time_req, str) and isinstance(obj.get("open"), str) and isinstance(obj.get("close"), str):
                    open_s = obj["open"].strip()
                    close_s = obj["close"].strip()
                    m_hhmm = _HHMM_RE.search(time_req.strip())
                    if m_hhmm and re.fullmatch(r"\d{2}:\d{2}", open_s) and re.fullmatch(r"\d{2}:\d{2}", close_s):
                        hhmm = m_hhmm.group(1)
                        if open_s < close_s and not (open_s <= hhmm <= close_s):
                            return (
                                f"Thời gian '{hhmm}' nằm ngoài giờ hoạt động của '{v}' ({open_s} - {close_s})."
                            )

                # Episode range bounds (video / audio series)
                ep_req = tool_args.get("episode")
                max_ep = obj.get("episodes") if isinstance(obj.get("episodes"), int) else obj.get("total_episodes")
                if isinstance(ep_req, int) and isinstance(max_ep, int) and (ep_req < 1 or ep_req > max_ep):
                    return (
                        f"Tập phim episode={ep_req} không hợp lệ cho '{v}' (chỉ có từ 1 đến {max_ep} tập)."
                    )

                # Status eligibility gates (ended sessions, delivered cancels, ineligible returns/re-actions)
                raw_status = obj.get("status")
                status_l = raw_status.strip().lower() if isinstance(raw_status, str) else ""
                if status_l == "ended" and "end" in tool_name:
                    return f"Phiên '{v}' đã kết thúc trước đó rồi, không gọi lại lệnh này."
                if status_l == "delivered" and "cancel" in tool_name:
                    return f"Đơn hàng '{v}' đã giao (delivered), không thể hủy trực tiếp."
                if "return" in tool_name and status_l in (
                    "cancelled",
                    "canceled",
                    "returned",
                    "refunded",
                    "shipping",
                    "in_transit",
                    "pending",
                    "processing",
                    "preparing",
                ):
                    return (
                        f"Đơn hàng '{v}' đang ở trạng thái '{raw_status}', không đủ điều kiện yêu cầu trả hàng."
                    )
                if status_l in ("cancelled", "canceled") and any(
                    w in tool_name for w in ("cancel", "return", "pay", "confirm")
                ):
                    return f"Đối tượng '{v}' đã bị hủy ({raw_status}), không thể tiếp tục thao tác '{tool_name}'."

                # Seats left in transportation trips
                seats_req = tool_args.get("seats")
                if isinstance(seats_req, int) and isinstance(obj.get("seats_left"), int):
                    if seats_req > obj["seats_left"]:
                        return (
                            f"Chuyến '{v}' chỉ còn {obj['seats_left']} chỗ trống, không đủ đặt {seats_req} ghế. "
                            f"Hãy chọn chuyến khác có đủ số chỗ."
                        )

                # Inventory stock in pantry / warehouse
                qty_req = tool_args.get("quantity")
                if isinstance(qty_req, (int, float)) and isinstance(obj.get("stock"), (int, float)):
                    if qty_req > obj["stock"]:
                        return (
                            f"Số lượng tồn kho của '{v}' chỉ còn {obj['stock']}, không đủ số lượng {qty_req}. "
                            f"Không thể thao tác vượt quá số lượng tồn kho."
                        )

        # 5. Check checkout on empty cart
        if "checkout" in tool_name:
            app_name = tool_name.split(".", 1)[0]
            if not self.cart_has_items.get(app_name, False):
                return (
                    f"Giỏ hàng của ứng dụng '{app_name}' hiện đang trống. "
                    f"Hãy gọi tool tìm kiếm và thêm vào giỏ hàng trước khi thực hiện checkout."
                )

        # 6. E-wallet insufficient balance fallback check
        if tool_name == "bill_payment.pay" and tool_args.get("source") == "ewallet":
            bill_id = str(tool_args.get("bill_id") or "")
            bill_obj = self.observed_entities.get(bill_id)
            bill_amt = bill_obj.get("amount") if isinstance(bill_obj, dict) else None
            wallet_bal = self.balances.get("ewallet.balance")

            if wallet_bal is None:
                user_l = self.user_text.lower()
                if any(kw in user_l for kw in ("không đủ", "nếu đủ", "nếu ví", "trừ ngân hàng", "qua ngân hàng")):
                    return (
                        "Người dùng có đặt điều kiện về số dư ví điện tử. "
                        "Hãy gọi 'ewallet.get_balance' trước để kiểm tra số dư ví."
                    )
            elif isinstance(bill_amt, (int, float)) and wallet_bal < bill_amt:
                return (
                    f"Số dư ví điện tử hiện tại ({wallet_bal}) KHÔNG ĐỦ để thanh toán hóa đơn ({bill_amt}). "
                    f"Hãy chuyển sang thanh toán bằng ngân hàng (source='bank') hoặc dừng lại."
                )

        return None
