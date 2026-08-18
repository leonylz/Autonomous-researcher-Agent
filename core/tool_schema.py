"""
Function Calling JSON 容错 — 四层防御。

Layer 1: json_repair      → 修语法（未转义换行、尾逗号、单引号）
Layer 2: jsonschema       → 验结构（字段类型、必填、枚举值）
Layer 3: Schema-Coerce    → 缺省填充 + 多余字段裁剪 + 类型强转
Layer 4: Retry-with-error → 把具体校验错误反馈给 LLM 重新生成（最多 2 次）

面试价值：LLM 输出不可靠，工业级容错是 Function Calling 的及格线。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

logger = logging.getLogger("autoresearcher.tool_schema")

# ── 可选依赖 ──
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False


# ═══════════════════════════════════════════════════════════════════
# 类型强转
# ═══════════════════════════════════════════════════════════════════

def coerce_type(value, target_type: str):
    """类型强转。LLM 经常把数字/布尔写成字符串。"""
    if target_type == "integer":
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            s = value.strip()
            try:
                return int(s)
            except ValueError:
                pass
        return value

    if target_type == "number":
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        return value

    if target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            s = value.strip().lower()
            if s in ("true", "1", "yes", "y"):
                return True
            if s in ("false", "0", "no", "n", ""):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return value

    if target_type == "string":
        if value is None:
            return ""
        if not isinstance(value, str):
            return str(value)
        return value

    if target_type == "array":
        if isinstance(value, str):
            # LLM 可能把数组写成逗号分隔的字符串
            s = value.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    pass
            return [x.strip() for x in s.split(",") if x.strip()]
        if isinstance(value, list):
            return value
        return [value] if value is not None else []

    if target_type == "object":
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        if isinstance(value, dict):
            return value
        return {}

    return value


# ═══════════════════════════════════════════════════════════════════
# Schema 校验 + 缺省填充
# ═══════════════════════════════════════════════════════════════════

def validate_and_coerce(args: dict, schema: dict) -> tuple[dict, list[str]]:
    """
    校验 + 缺省填充 + 多余裁剪 + 类型强转。

    Returns
    -------
    (cleaned_args, errors)
        errors 为空列表表示校验通过。
        若有 errors，cleaned_args 仍然可用（非致命错误时不阻塞）。
    """
    if not isinstance(args, dict):
        return {}, ["args must be a JSON object"]

    cleaned: dict = {}
    errors: list[str] = []
    props = schema.get("properties", {}) or {}
    required: set = set(schema.get("required", []) or [])

    # 1. 检查必填字段
    for key in required:
        if key not in args or args[key] is None or (isinstance(args[key], str) and not args[key].strip()):
            errors.append(f"missing required field: {key}")

    # 2. 逐字段校验 + 类型强转
    #     schema 未声明任何 properties 时，透传所有 args（工具可能是自由参数）。
    #     只有当 schema 明确声明了字段，才裁剪多余/幻觉字段。
    has_schema_props = bool(props)
    for key, value in args.items():
        if not has_schema_props:
            cleaned[key] = value
            continue
        if key not in props:
            # 多余字段 —— LLM 经常幻觉出额外字段，裁剪掉
            logger.debug("Dropping hallucinated field: %s", key)
            continue

        pspec = props[key]
        expected_type = pspec.get("type", "string")

        coerced = coerce_type(value, expected_type)
        if type(coerced).__name__ != type(value).__name__:
            logger.debug("Coerced %s: %s → %s", key, type(value).__name__, expected_type)

        # 枚举校验
        enum_values = pspec.get("enum")
        if enum_values and coerced not in enum_values:
            errors.append(
                f"field '{key}' value '{coerced}' not in allowed values: {enum_values}"
            )

        cleaned[key] = coerced

    # 3. 缺失可选字段 → 填默认值
    for key, pspec in props.items():
        if key not in cleaned:
            if "default" in pspec:
                cleaned[key] = pspec["default"]
            elif key not in required:
                # 可选字段给类型零值
                cleaned[key] = _type_default(pspec.get("type", "string"))

    # 4. 正式 jsonschema 校验（如果安装了）
    if HAS_JSONSCHEMA and not errors:
        try:
            jsonschema.validate(instance=cleaned, schema=schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"schema validation: {exc.message}")

    return cleaned, errors


def _type_default(type_name: str):
    """类型的零值。"""
    return {
        "string": "",
        "integer": 0,
        "number": 0.0,
        "boolean": False,
        "array": [],
        "object": {},
    }.get(type_name, "")


# ═══════════════════════════════════════════════════════════════════
# 四层容错入口
# ═══════════════════════════════════════════════════════════════════

def robust_parse_tool_call(
    raw_body: str,
    tool_schemas: dict[str, dict],
    max_retries: int = 2,
    retry_callback: Optional[Callable[[str], str]] = None,
) -> Optional[dict]:
    """
    四层容错解析一个 <tool_call> JSON body。

    Parameters
    ----------
    raw_body : str
        <tool_call> 标签内的原始 JSON 文本。
    tool_schemas : dict
        {tool_name: input_schema} 映射。
    max_retries : int
        最大重试次数（每次重试会将错误信息反馈给 retry_callback）。
    retry_callback : callable or None
        接收错误描述，返回修正后的 JSON 字符串。使用 LLM 重新生成。

    Returns
    -------
    {"name": str, "args": dict} 或 None（不可恢复）。
    """
    body = raw_body.strip()

    # ── Layer 1: json_repair 语法修复 ──
    if HAS_JSON_REPAIR:
        try:
            body = repair_json(body)
        except Exception:
            logger.debug("json_repair failed, using raw body")

    # ── Layer 2-4: 解析 → Schema 校验 → 重试 ──
    for attempt in range(1 + max_retries):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            error_msg = f"JSON syntax error at line {exc.lineno}, col {exc.colno}: {exc.msg}"
            logger.warning("Tool call parse attempt %d: %s", attempt + 1, error_msg)
            if attempt < max_retries and retry_callback:
                body = retry_callback(error_msg)
                continue
            return None

        if not isinstance(parsed, dict):
            return None

        name = parsed.get("name", "")
        if not name or not isinstance(name, str):
            logger.warning("Tool call missing 'name' field")
            return None

        # LLM 常见变异：args vs arguments vs parameters
        raw_args = parsed.get("args") or parsed.get("arguments") or parsed.get("parameters") or {}

        if not isinstance(raw_args, dict):
            logger.warning("Tool '%s' args is not a dict: %s", name, type(raw_args).__name__)
            return None  # args 非法 → 拒绝该调用（而非 coerce 成 {}）

        # ── Layer 3: Schema 校验 ──
        schema = tool_schemas.get(name)
        if schema:
            cleaned_args, errors = validate_and_coerce(raw_args, schema)

            if errors:
                error_msg = "; ".join(errors)
                logger.warning(
                    "Tool '%s' validation failed (attempt %d): %s",
                    name, attempt + 1, error_msg,
                )

                if attempt < max_retries and retry_callback:
                    retry_prompt = (
                        f"Tool '{name}' argument validation failed:\n"
                        f"  Errors: {error_msg}\n"
                        f"  Expected schema: {json.dumps(schema, ensure_ascii=False)}\n"
                        f"  You provided: {json.dumps(raw_args, ensure_ascii=False)}\n"
                        f"Please fix the errors and re-emit the corrected <tool_call> block."
                    )
                    body = retry_callback(retry_prompt)
                    continue

                # 最后一次尝试：非致命错误降级处理
                fatal = any("missing required field" in e for e in errors)
                if fatal:
                    logger.error(
                        "Tool '%s' has fatal schema errors after %d retries: %s",
                        name, max_retries, error_msg,
                    )
                    return None

                logger.info(
                    "Using coerced args for '%s' despite non-fatal errors: %s",
                    name, error_msg,
                )
                return {"name": name, "args": cleaned_args}

            return {"name": name, "args": cleaned_args}
        else:
            # 未知工具：宽松模式
            logger.debug("Unknown tool '%s', using raw args", name)
            return {"name": name, "args": raw_args}

    return None


# ═══════════════════════════════════════════════════════════════════
# Tool schema 索引构建
# ═══════════════════════════════════════════════════════════════════

def build_schema_index(tool_defs: list[dict]) -> dict[str, dict]:
    """从 ToolRegistry.get_tools_for() 的输出构建 {name: input_schema} 索引。"""
    index: dict[str, dict] = {}
    for td in tool_defs:
        name = td.get("name", "")
        schema = td.get("input_schema", {})
        if name and schema:
            index[name] = schema
    return index
