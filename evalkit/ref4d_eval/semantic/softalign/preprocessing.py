# softalign/preprocessing.py
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import PreprocConfig
from .types import Entity

__all__ = [
    "normalize_text",
    "normalize_name",
    "normalize_key",
    "normalize_value",
    "normalize_attr_map",
    "coerce_entities_from_raw",
]

# ---- 常量：哪些 key 允许非字母值（如纯数字/混合符号），与“无词表”不冲突 ----
# 典型场景：
# - number-or-id: 耳标/球衣号/牌照号 等
# - printed-text: 画面中文字/号码
# - signature: 提取阶段生成的稳态签名（可能含数字/短码）
_ALLOW_NONALPHA_KEYS = {"number-or-id", "printed-text", "signature"}


# ------------------------------
# 基础规整工具（不涉词表/同义映射）
# ------------------------------

def _strip_decorations(s: str) -> str:
    # 移除两端常见装饰字符；内部保留（如 a-b、C&F）
    return s.strip(" \t\r\n'\"`.,;:!?()[]{}<>")

def _canonical_hyphen(s: str) -> str:
    # 统一下划线/多连字符；保留单个连字符，避免断词
    s = s.replace("_", "-")
    s = re.sub(r"\s*-\s*", "-", s)         # a - b -> a-b
    s = re.sub(r"-{2,}", "-", s)           # --- -> -
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_unicode(s: str) -> str:
    # NFC 规约 & 去除兼容性异常空白；不过度去重音（保留可读性）
    s = unicodedata.normalize("NFC", s)
    # 将各类不可见空白归一为普通空格
    s = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", s)
    return s

def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    strip: bool = True,
    canonical_hyphen: bool = True,
) -> str:
    """
    极简可复现文本规整：
      - 统一 Unicode & 空白
      - 可选小写
      - 可选去两端装饰符
      - 可选下划线→连字符、连字符规范化
    不做任何词表/同义替换。
    """
    if not isinstance(text, str):
        return ""
    s = _normalize_unicode(text)
    if lowercase:
        s = s.lower()
    if strip:
        s = _strip_decorations(s)
    if canonical_hyphen:
        s = _canonical_hyphen(s)
    # 再次清理多余空格
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------------
# 令牌级过滤（不维护词表）
# ------------------------------

def _has_alpha(s: str) -> bool:
    return bool(re.search(r"[a-z]", s))

def _passes_length_gate(token: str, drop_short_token_len: int) -> bool:
    # 统计字母字符长度；若过短且整体长度也极短，则丢弃
    alpha_len = len(re.sub(r"[^a-z]", "", token))
    if alpha_len <= drop_short_token_len and len(token) <= drop_short_token_len + 1:
        return False
    return True


# ------------------------------
# 名称/键/值 规整
# ------------------------------

def normalize_name(name: str, cfg: PreprocConfig) -> str:
    """
    实体类名规整：小写、去装饰、连字符统一、空白合并。
    不进行任何同义词/复数化/词表操作。
    """
    return normalize_text(
        name or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )

def normalize_key(key: str, cfg: PreprocConfig) -> str:
    """
    属性键规整：与 name 同一规则，强调字符串相等。
    """
    return normalize_text(
        key or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )

def normalize_value(key: str, val: str, cfg: PreprocConfig) -> Optional[str]:
    """
    属性值规整（单值）：
      - 轻规整：小写/修饰去除/连字符统一
      - 非字母值过滤：默认丢弃，但若 key 在 _ALLOW_NONALPHA_KEYS 列表中则保留
      - 极短 token 过滤：基于字母长度阈值
    返回 None 表示该值被过滤。
    """
    s = normalize_text(
        val or "",
        lowercase=cfg.lowercase,
        strip=cfg.strip,
        canonical_hyphen=cfg.canonical_hyphen,
    )
    if not s:
        return None

    # 允许纯数字/弱字母的 key 单独放行
    allow_nonalpha = normalize_key(key, cfg) in _ALLOW_NONALPHA_KEYS
    if cfg.drop_nonalpha and (not allow_nonalpha) and (not _has_alpha(s)):
        return None

    if not _passes_length_gate(s, cfg.drop_short_token_len):
        return None

    return s


# ------------------------------
# 属性映射规整（键值去重、稳定排序）
# ------------------------------

def _dedup_stable(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_attr_map(attrs: Dict[str, List[str]], cfg: PreprocConfig) -> Dict[str, List[str]]:
    """
    对实体的属性映射做规整：
      - 键、值逐一规整
      - 空键/空值过滤
      - 值去重（保持首次出现顺序）
      - 不做词表/同义处理，维持开放集合
    """
    if not isinstance(attrs, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for raw_k, raw_vals in attrs.items():
        k = normalize_key(str(raw_k), cfg)
        if not k:
            continue

        vals: List[str] = []
        for v in (raw_vals or []):
            nv = normalize_value(k, v, cfg)
            if nv:
                vals.append(nv)

        vals = _dedup_stable(vals)
        if vals:
            out[k] = vals

    return out


# ------------------------------
# 从原始 JSON（抽取结果）稳健构造实体列表
# ------------------------------

def coerce_entities_from_raw(
    raw_doc: Dict,
    cfg: PreprocConfig,
    *,
    expect_path: Tuple[str, ...] = ("fine", "entities"),
) -> List[Entity]:
    """
    从抽取 JSON 中稳健拿到实体列表并做轻规整：
      - 路径默认 raw["fine"]["entities"]
      - name 使用 normalize_name
      - attrs 使用 normalize_attr_map
      - id 以字符串保留
    """
    obj = raw_doc or {}
    for k in expect_path:
        obj = obj.get(k, {}) if isinstance(obj, dict) else {}
    entities_raw = obj if isinstance(obj, list) else []

    out: List[Entity] = []
    for e in entities_raw:
        if not isinstance(e, dict):
            continue
        name = normalize_name(e.get("name", "") or "", cfg)
        attrs = e.get("attributes", {}) or {}
        attrs_norm = normalize_attr_map(attrs, cfg)
        out.append(Entity(id=str(e.get("id", "") or ""), name=name, attrs=attrs_norm))
    return out
