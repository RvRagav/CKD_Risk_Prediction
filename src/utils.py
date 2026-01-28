from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def to_list(x: Iterable[str]) -> list[str]:
    return list(x)


def clip_to_clinical_bounds(df, bounds):
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = out[col].clip(lo, hi)
    return out
