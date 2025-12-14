# softalign/api.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import Config, load_config
from .encoder import TextEncoder, build_text_encoder
from .scoring import SampleScorer
from .types import ScoreReport


class SoftAlignAPI:
    """
    对外友好入口：
      - from_yaml(): 读取 YAML、构建 Config 与 Encoder
      - score_pair[_from_files](): 单样本评分（含非空校验与可解释报告）
    """

    def __init__(self, cfg: Config, encoder: TextEncoder):
        self.cfg = cfg
        self.encoder = encoder
        self._scorer = SampleScorer(cfg, encoder)

    # --------- 工厂 ---------
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SoftAlignAPI":
        cfg: Config = load_config(yaml_path)
        enc: TextEncoder = build_text_encoder(cfg)
        return cls(cfg, enc)

    # --------- 评分接口 ---------
    def score_pair(
        self,
        ref_doc_or_path: Dict | str | Path,
        gen_doc_or_path: Dict | str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self._scorer.score_pair(ref_doc_or_path, gen_doc_or_path, sample_id=sample_id)

    def score_pair_from_files(
        self,
        ref_json_path: str | Path,
        gen_json_path: str | Path,
        sample_id: Optional[str] = None,
    ) -> ScoreReport:
        return self._scorer.score_pair_from_files(ref_json_path, gen_json_path, sample_id=sample_id)


# 便捷方法
def build_api(yaml_path: str | Path) -> SoftAlignAPI:
    """
    一行构建：
      api = build_api("softalign.yaml")
      report = api.score_pair_from_files("ref.json", "gen.json", sample_id="vid_0001")
    """
    return SoftAlignAPI.from_yaml(yaml_path)
