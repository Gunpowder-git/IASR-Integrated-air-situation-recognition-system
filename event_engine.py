from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import time

# 统一事件模型（Event Schema）
@dataclass
class Event:
    event_id: str
    time_s: float
    category: str                 # bird_flock / visibility_low / vehicle_stopped / intrusion
    severity: int                 # 0-3
    confidence: float             # 0-1
    target: str                   # 通知对象：运行/安防/应急/运维/农业...
    message: str                  # 人类可读说明
    evidence_path: Optional[str] = None
    extras: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

def make_event_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time()*1000)}"

class Cooldown:
    """简单防抖：同类事件在cooldown秒内不重复发"""
    def __init__(self):
        self.last_emit: Dict[str, float] = {}

    def allow(self, key: str, now_s: float, cooldown_s: float) -> bool:
        last = self.last_emit.get(key, -1e9)
        if now_s - last >= cooldown_s:
            self.last_emit[key] = now_s
            return True
        return False

def severity_by_threshold(value: float, t1: float, t2: float, t3: float) -> int:
    """把连续值映射到 0/1/2/3"""
    if value < t1: return 0
    if value < t2: return 1
    if value < t3: return 2
    return 3
