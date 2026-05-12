from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import time
import uuid
import json

import pandas as pd


EVENT_LABELS = {
    "bird_flock": "鸟群风险",
    "intrusion": "敏感区域入侵",
    "visibility_low": "低能见度风险",
    "vehicle_stopped": "疑似异常停车",
    "crop_disease_suspected": "疑似作物病害",
}

SEVERITY_LABELS = {
    0: "提示",
    1: "一般",
    2: "较高",
    3: "紧急",
}

DEFAULT_ACTIONS = {
    "bird_flock": "建议运行/飞行管控人员复核，必要时调整航线或降低运行强度。",
    "intrusion": "建议安防/监管人员复核目标身份与位置，必要时启动处置流程。",
    "visibility_low": "建议应急/运行调度复核气象与能见度，必要时延误或提高运行间隔。",
    "vehicle_stopped": "建议交管/运营人员复核现场，判断是否存在事故、故障或拥堵源。",
    "crop_disease_suspected": "建议农业运维人员复核样本，必要时安排人工巡检或采样检测。",
}


@dataclass
class Event:
    """统一事件模型。既能保存算法结果，也方便模拟跨部门联动。"""

    event_id: str
    time_s: float
    category: str
    severity: int
    confidence: float
    target: str
    message: str
    evidence_path: Optional[str] = None
    extras: Optional[dict] = None
    source: str = "video"
    source_id: str = "default_source"
    status: str = "待人工复核"
    recommended_action: str = ""
    dispatch_channel: str = "模拟平台队列"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not self.recommended_action:
            self.recommended_action = DEFAULT_ACTIONS.get(self.category, "建议人工复核并记录处置结果。")

    def to_dict(self) -> dict:
        return asdict(self)


def make_event_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"


class Cooldown:
    """简单防抖：同类事件在 cooldown 秒内不重复发。"""

    def __init__(self) -> None:
        self.last_emit: Dict[str, float] = {}

    def allow(self, key: str, now_s: float, cooldown_s: float) -> bool:
        last = self.last_emit.get(key, -1e9)
        if now_s - last >= cooldown_s:
            self.last_emit[key] = now_s
            return True
        return False


def severity_by_threshold(value: float, t1: float, t2: float, t3: float) -> int:
    """把连续值映射到 0/1/2/3。"""
    if value < t1:
        return 0
    if value < t2:
        return 1
    if value < t3:
        return 2
    return 3


def event_to_zh_dict(event: Event | dict) -> dict:
    e = event.to_dict() if isinstance(event, Event) else dict(event)
    severity = int(e.get("severity", 0))
    category = e.get("category", "")
    return {
        "事件ID": e.get("event_id", ""),
        "时间(s)": round(float(e.get("time_s", 0.0)), 2),
        "事件类型": EVENT_LABELS.get(category, category),
        "英文类型": category,
        "严重等级": SEVERITY_LABELS.get(severity, str(severity)),
        "严重等级数值": severity,
        "置信度": round(float(e.get("confidence", 0.0)), 3),
        "通知对象": e.get("target", ""),
        "处置状态": e.get("status", ""),
        "建议动作": e.get("recommended_action", ""),
        "事件说明": e.get("message", ""),
        "证据路径": e.get("evidence_path", ""),
        "来源": e.get("source", ""),
        "来源ID": e.get("source_id", ""),
        "生成时间": e.get("created_at", ""),
    }


def events_to_zh_dataframe(events: List[Event | dict]) -> pd.DataFrame:
    return pd.DataFrame([event_to_zh_dict(e) for e in events])


def events_to_raw_dataframe(events: List[Event | dict]) -> pd.DataFrame:
    rows = [e.to_dict() if isinstance(e, Event) else dict(e) for e in events]
    return pd.DataFrame(rows)


def make_dispatch_rows(events: List[Event | dict], run_id: str = "") -> List[dict]:
    rows: List[dict] = []
    for i, event in enumerate(events, start=1):
        e = event.to_dict() if isinstance(event, Event) else dict(event)
        category = e.get("category", "")
        severity = int(e.get("severity", 0))
        dispatch_id = f"DISP-{run_id}-{i:03d}" if run_id else f"DISP-{i:03d}"
        rows.append({
            "dispatch_id": dispatch_id,
            "event_id": e.get("event_id", ""),
            "event_category": category,
            "event_label": EVENT_LABELS.get(category, category),
            "severity": severity,
            "severity_label": SEVERITY_LABELS.get(severity, str(severity)),
            "target": e.get("target", ""),
            "channel": e.get("dispatch_channel", "模拟平台队列"),
            "status": e.get("status", "待人工复核"),
            "recommended_action": e.get("recommended_action", DEFAULT_ACTIONS.get(category, "建议人工复核。")),
            "created_at": e.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "message": e.get("message", ""),
            "evidence_path": e.get("evidence_path", ""),
        })
    return rows


def dispatch_to_zh_dataframe(events: List[Event | dict], run_id: str = "") -> pd.DataFrame:
    rows = make_dispatch_rows(events, run_id=run_id)
    zh_rows = []
    for r in rows:
        zh_rows.append({
            "派单ID": r["dispatch_id"],
            "关联事件ID": r["event_id"],
            "事件类型": r["event_label"],
            "严重等级": r["severity_label"],
            "通知对象": r["target"],
            "联动通道": r["channel"],
            "处置状态": r["status"],
            "建议动作": r["recommended_action"],
            "生成时间": r["created_at"],
            "事件说明": r["message"],
            "证据路径": r["evidence_path"],
        })
    return pd.DataFrame(zh_rows)


def write_event_outputs(events: List[Event | dict], out_dir: str | Path, run_id: str = "", prefix: str = "") -> dict:
    """保存原始事件、中文事件表和模拟联动派单表。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name_prefix = f"{prefix}_" if prefix else ""

    raw_df = events_to_raw_dataframe(events)
    zh_df = events_to_zh_dataframe(events)
    dispatch_df = pd.DataFrame(make_dispatch_rows(events, run_id=run_id))
    dispatch_zh_df = dispatch_to_zh_dataframe(events, run_id=run_id)

    events_json = out_dir / f"{name_prefix}events.json"
    events_csv = out_dir / f"{name_prefix}events.csv"
    events_zh_csv = out_dir / f"{name_prefix}events_zh.csv"
    dispatch_csv = out_dir / f"{name_prefix}dispatch_log.csv"
    dispatch_zh_csv = out_dir / f"{name_prefix}dispatch_log_zh.csv"

    raw_records = [e.to_dict() if isinstance(e, Event) else dict(e) for e in events]
    events_json.write_text(json.dumps(raw_records, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_df.to_csv(events_csv, index=False, encoding="utf-8-sig")
    zh_df.to_csv(events_zh_csv, index=False, encoding="utf-8-sig")
    dispatch_df.to_csv(dispatch_csv, index=False, encoding="utf-8-sig")
    dispatch_zh_df.to_csv(dispatch_zh_csv, index=False, encoding="utf-8-sig")

    return {
        "events_json": str(events_json),
        "events_csv": str(events_csv),
        "events_zh_csv": str(events_zh_csv),
        "dispatch_csv": str(dispatch_csv),
        "dispatch_zh_csv": str(dispatch_zh_csv),
    }
