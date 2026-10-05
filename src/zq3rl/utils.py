"""Shared utilities: seeding, JSON/CSV IO, run directories, metric helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def environment_info() -> Dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git_revision(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        info["torch"] = "missing"
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
#  Run directories
# --------------------------------------------------------------------------- #
@dataclass
class RunDir:
    root: Path
    name: str

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def make(self) -> "RunDir":
        for p in (self.figures, self.data, self.models, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def make_run_dir(root: str | Path, name: Optional[str] = None) -> RunDir:
    root = Path(root)
    if name is None:
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
    rd = RunDir(root=root / name, name=name)
    return rd.make()


# --------------------------------------------------------------------------- #
#  IO
# --------------------------------------------------------------------------- #
def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=default),
                    encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Minimal CSV writer (no pandas dependency in the hot path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def summarise(values: Iterable[float]) -> Dict[str, float]:
    a = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal
    approximation at small n, which matters for the 100-episode evaluations)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(centre - half, 0.0)), float(min(centre + half, 1.0)))


# --------------------------------------------------------------------------- #
#  Console
# --------------------------------------------------------------------------- #
class Tee:
    """Duplicate writes to the console and a log file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, s: str) -> int:
        self._stdout.write(s)
        self._fh.write(s)
        self._fh.flush()
        return len(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def banner(text: str, width: int = 74, ch: str = "=") -> str:
    line = ch * width
    return f"{line}\n{text}\n{line}"
