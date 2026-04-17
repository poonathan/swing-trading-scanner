import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / ".cache"
TTL_PRICE = 4 * 3600      # 4 hours for price/technical data
TTL_FUNDAMENTAL = 24 * 3600  # 24 hours for fundamentals


def _key(symbol: str, data_type: str) -> str:
    date_str = time.strftime("%Y-%m-%d")
    raw = f"{symbol}:{data_type}:{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def get(symbol: str, data_type: str, ttl: int = TTL_PRICE) -> Optional[Any]:
    key = _key(symbol, data_type)
    p = _path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(p) as f:
            payload = json.load(f)
        dtype = payload.get("dtype")
        if dtype == "dataframe":
            return pd.read_json(payload["data"], orient="split")
        return payload.get("data")
    except Exception:
        return None


def set(symbol: str, data_type: str, value: Any) -> None:
    key = _key(symbol, data_type)
    p = _path(key)
    try:
        if isinstance(value, pd.DataFrame):
            payload = {"dtype": "dataframe", "data": value.to_json(orient="split")}
        else:
            payload = {"dtype": "json", "data": value}
        with open(p, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def clear() -> int:
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        count += 1
    return count


def stats() -> dict:
    if not CACHE_DIR.exists():
        return {"files": 0, "size_mb": 0}
    files = list(CACHE_DIR.glob("*.json"))
    size = sum(f.stat().st_size for f in files)
    return {"files": len(files), "size_mb": round(size / 1_048_576, 2)}
