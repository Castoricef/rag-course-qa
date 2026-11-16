"""配置读取：用标准库 tomllib 读 config.toml，省掉一个 PyYAML 依赖。"""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULTS = {
    "max_chars": 500,
    "overlap": 80,
    "top_k": 3,
    "min_score": 0.35,
    "min_score_offline": 0.2,
    "chat_model": "gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "temperature": 0.2,
    "offline": False,
}


def load(path: str | Path = "config.toml") -> dict:
    cfg = dict(DEFAULTS)
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            cfg.update(tomllib.load(f))
    return cfg
