# /root/autodl-tmp/event_eval/src/embed/e5_encoder.py
# -*- coding: utf-8 -*-
"""
E5 encoder for segment-level (and micro-event level) text embeddings.

I/O:
- Input : data/vlm/{ref|gen}/{sample_id}.vlm.json
          Each item: {"id","s_abs","e_abs","s","e","text": "clause1 | clause2 | ..."}
- Output: data/embeds/{ref|gen}/{sample_id}.emb.json
          Expanded list. Each item (one micro-event):
            {
              "id": "e0001#1",
              "parent_id": "e0001",
              "s_abs","e_abs","s","e",
              "text": "one clause",
              "emb": [float, ...],   # L2-normalized when normalize=True
              "norm": true           # ← 新增，标注该向量是否已单位化
            }
- Optional split export (embed.microevent.split_export = true):
  data/embeds/{ref|gen}/{sample_id}.parts/e0001__1.emb.json
"""

from __future__ import annotations
import os
import math
import re
import argparse
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from ..common.io import read_json, write_json, read_yaml, ensure_dir

LOGGER = logging.getLogger("event_eval.embed.e5")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    LOGGER.addHandler(h)
    LOGGER.setLevel(logging.INFO)


# --------------------------
# Utils
# --------------------------

def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "fp32").lower()
    if s in ("fp16", "float16"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32

def _expand_placeholders(s: str, cfg: Dict[str, Any]) -> str:
    data_root = cfg.get("paths", {}).get("data_root", "/root/autodl-tmp/event_eval/data")
    return s.replace("${paths.data_root}", str(data_root))

def _infer_side(path: str) -> str:
    p = str(path)
    if "/vlm/ref/" in p or "/events/ref/" in p: return "ref"
    if "/vlm/gen/" in p or "/events/gen/" in p: return "gen"
    return "ref"

def _infer_sample_id(vlm_path: str) -> str:
    name = Path(vlm_path).name
    if name.endswith(".vlm.json"):
        return name[:-len(".vlm.json")]
    if name.endswith(".events.json"):
        return name[:-len(".events.json")]
    return Path(vlm_path).stem

def _split_microevents(
    text: str,
    delims: Optional[List[str]] = None,
    sep: Optional[str] = None,
    strip_punct: bool = True,
    max_parts: int = 8,
) -> List[str]:
    """
    按多分隔符切分微事件句子集合。优先 delims；若未提供，则使用 sep；
    两者都未提供则采用默认分隔符：["|","。","；",";","."]
    """
    if text is None:
        return []
    t0 = text.strip()
    if not t0:
        return []

    # 确定分隔符集合
    if delims and isinstance(delims, list) and any(delims):
        _delims = [str(d) for d in delims if d is not None and str(d) != ""]
    elif sep:
        _delims = [str(sep)]
    else:
        _delims = ["|", "。", "；", ";", "."]

    # 构造 regex 并切分
    pattern = "|".join([re.escape(d) for d in _delims])
    parts = re.split(pattern, t0)

    out: List[str] = []
    for t in parts:
        t = t.strip()
        if strip_punct:
            t = t.strip(" .;、，。！？!?:：-—|")
        if t:
            out.append(t)
        if max_parts and len(out) >= max_parts:
            break
    return out or [""]

def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
    summed = (last_hidden * mask).sum(dim=1)                  # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-9)                   # [B, 1]
    return summed / denom

def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)

def _device_from_cfg(device: str) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------
# E5 Model Loader
# --------------------------

def load_e5(cfg: Dict[str, Any]):
    ecfg = cfg.get("embed", {}) or {}
    model_name = ecfg.get("model", "/root/autodl-tmp/models/e5-large-v2")
    device = _device_from_cfg(ecfg.get("device", "auto"))
    dtype = _dtype_from_str(ecfg.get("dtype", "fp32"))

    # ★ CPU 上强制 fp32，避免半精度算子不支持/不稳定
    if device == "cpu" and dtype is not torch.float32:
        LOGGER.info(f"[E5] Forcing dtype=float32 on CPU (was {dtype}).")
        dtype = torch.float32

    LOGGER.info(f"Loading E5 model: {model_name} (device={device}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    return tokenizer, model, device, dtype


# --------------------------
# Encoding
# --------------------------

@torch.inference_mode()
def encode_texts(texts: List[str],
                 tokenizer: AutoTokenizer,
                 model: AutoModel,
                 device: str,
                 dtype: torch.dtype,
                 batch_size: int,
                 max_length: int,
                 text_prefix: str,
                 normalize: bool) -> torch.Tensor:
    """
    Return: torch.Tensor [N, D] on CPU
    """
    all_out: List[torch.Tensor] = []
    n = len(texts)
    for st in range(0, n, batch_size):
        ed = min(n, st + batch_size)
        batch = [f"{text_prefix}{t}" for t in texts[st:ed]]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        last_hidden = out.last_hidden_state  # [B, L, D]
        pooled = _mean_pool(last_hidden, enc["attention_mask"])
        if normalize:
            pooled = _normalize(pooled)
        all_out.append(pooled.detach().to("cpu"))
    return torch.cat(all_out, dim=0) if all_out else torch.empty(0, model.config.hidden_size)


# --------------------------
# Similarity helpers
# --------------------------

def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(a.size(0), b.size(0))
    a_n = _normalize(a) if not torch.allclose(a.norm(dim=1), torch.ones(a.size(0)), atol=1e-3) else a
    b_n = _normalize(b) if not torch.allclose(b.norm(dim=1), torch.ones(b.size(0)), atol=1e-3) else b
    return a_n @ b_n.t()

def pairwise_sim_sem(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (pairwise_cosine(a, b) + 1.0) / 2.0


# --------------------------
# Runner
# --------------------------

def _derive_out_paths(cfg: Dict[str, Any], side: str, sample_id: str) -> Tuple[Path, Optional[Path]]:
    ecfg = cfg.get("embed", {}) or {}
    exp = ecfg.get("export", {}) or {}

    out_dir = _expand_placeholders(exp.get("out_dir", "${paths.data_root}/embeds"), cfg)
    fname = exp.get("fname_ref" if side == "ref" else "fname_gen", "{sample_id}.emb.json")
    out_path = Path(out_dir) / fname.format(sample_id=sample_id)
    ensure_dir(out_path.parent)

    parts_dir_pat = exp.get("parts_dir", None)
    parts_dir = None
    if parts_dir_pat:
        parts_rel = parts_dir_pat.format(side=side, sample_id=sample_id)
        parts_dir = Path(out_dir) / parts_rel
        ensure_dir(parts_dir)
    return out_path, parts_dir

def run(vlm_json_path: str, out_json_path: Optional[str], cfg_path: str) -> Dict[str, Any]:
    cfg = read_yaml(cfg_path)
    ecfg = cfg.get("embed", {}) or {}

    # load model
    tokenizer, model, device, dtype = load_e5(cfg)

    # params
    bs = int(ecfg.get("batch_size", 64))
    max_len = int(ecfg.get("max_length", 128))
    text_prefix = str(ecfg.get("text_prefix", "query: "))
    norm = bool(ecfg.get("normalize", True))

    mcfg = ecfg.get("microevent", {}) or {}
    # 新增：支持 delims（优先生效），兼容原 sep
    delims_cfg = mcfg.get("delims", None)
    delims = [str(d) for d in delims_cfg] if isinstance(delims_cfg, list) else None
    sep = str(mcfg.get("sep", "")).strip() or None
    strip_punct = bool(mcfg.get("strip_punct", True))
    max_parts = int(mcfg.get("max_parts", 8))
    split_export = bool(mcfg.get("split_export", True))

    # read vlm
    vlm = read_json(vlm_json_path)
    if not isinstance(vlm, list):
        raise ValueError(f"vlm json must be list: {vlm_json_path}")

    side = _infer_side(vlm_json_path)
    sample_id = _infer_sample_id(vlm_json_path)

    # derive outputs
    if out_json_path:
        out_path = Path(out_json_path)
        ensure_dir(out_path.parent)
        parts_dir = None
    else:
        out_path, parts_dir = _derive_out_paths(cfg, side, sample_id)
        if not split_export:
            parts_dir = None

    # expand micro-events
    expanded: List[Dict[str, Any]] = []
    for seg in vlm:
        base = {k: seg[k] for k in seg.keys() if k not in ("text",)}
        parent_id = str(seg.get("id", ""))
        texts = _split_microevents(
            seg.get("text", ""),
            delims=delims,
            sep=sep,
            strip_punct=strip_punct,
            max_parts=max_parts,
        )
        if not texts:
            texts = [""]  # placeholder
        for idx, t in enumerate(texts, start=1):
            rec = dict(base)
            rec["parent_id"] = parent_id
            rec["id"] = f"{parent_id}#{idx}"
            rec["text"] = t
            expanded.append(rec)

    # encode
    texts = [r["text"] for r in expanded]
    embs = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
        batch_size=bs,
        max_length=max_len,
        text_prefix=text_prefix,
        normalize=norm,
    )
    dim = embs.size(-1)

    # attach & export aggregate
    for rec, vec in zip(expanded, embs):
        rec["emb"] = vec.tolist()
        rec["norm"] = bool(norm)  # ← 新增：记录是否已单位化
    write_json(expanded, out_path, indent=2)
    LOGGER.info(f"Wrote aggregate embeddings: {out_path} (items={len(expanded)}, dim={dim})")

    # optional split export
    if parts_dir is not None:
        for rec in expanded:
            fname = f"{rec['id'].replace('#','__')}.emb.json"
            write_json([rec], parts_dir / fname, indent=2)
        LOGGER.info(f"Wrote per-micro-event files under: {parts_dir}")

    return {"n_items": len(expanded), "dim": dim, "out": str(out_path), "parts_dir": str(parts_dir) if parts_dir else None}


# --------------------------
# CLI
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="E5 embedding for VLM micro-events")
    ap.add_argument("--vlm", type=str, required=True, help="Path to vlm json, e.g., data/vlm/ref/<id>.vlm.json")
    ap.add_argument("--config", type=str, required=True, help="Path to configs/model_embed.yaml")
    ap.add_argument("--out", type=str, default=None, help="Optional: override aggregate output path")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        vlm_json_path=args.vlm,
        out_json_path=args.out,
        cfg_path=args.config,
    )
