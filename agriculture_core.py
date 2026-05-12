from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

from event_engine import Event, make_event_id, severity_by_threshold, write_event_outputs


def _ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _resize_if_needed(img: np.ndarray, max_width: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def analyze_crop_image(
    image_path: str,
    out_dir: str,
    disease_threshold: float = 0.08,
    min_leaf_area_ratio: float = 0.05,
    routing: Optional[Dict[str, str]] = None,
    run_id: str = "",
    source_id: str = "agriculture_image",
) -> tuple[dict, list[Event], dict]:
    """
    农业扩展接口 MVP：基于颜色/纹理启发式判断“疑似病害”。

    说明：这不是严肃病害分类模型，而是一个可插拔农业事件接口。
    它用于把作物图像转成“疑似病害事件”，方便接入现有预警/联动系统。
    """
    _ensure_dir(out_dir)
    out_dir_path = Path(out_dir)
    evidence_dir = out_dir_path / "agriculture_evidence"
    _ensure_dir(evidence_dir)

    if routing is None:
        routing = {"crop_disease_suspected": "农业运维/巡检人员"}

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("图片读取失败：请确认上传的是 jpg/png/jpeg 等常见图片格式。")

    img = _resize_if_needed(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 绿色叶片区域：粗略检测叶面主体。
    leaf_mask = cv2.inRange(hsv, np.array([35, 35, 35]), np.array([95, 255, 255]))
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)

    # 疑似病斑/黄化/枯斑区域：棕黄、暗斑、低亮度区域。
    brown_yellow = cv2.inRange(hsv, np.array([8, 45, 35]), np.array([42, 255, 230]))
    dark_spots = cv2.inRange(hsv, np.array([0, 25, 0]), np.array([179, 255, 75]))
    suspected_mask = cv2.bitwise_or(brown_yellow, dark_spots)

    # 如果检测到较明确叶片，就只在叶片邻近区域内判断；否则使用全图比例作为兜底。
    leaf_area = int(np.count_nonzero(leaf_mask))
    total_area = int(img.shape[0] * img.shape[1])
    leaf_area_ratio = leaf_area / max(total_area, 1)

    if leaf_area_ratio >= min_leaf_area_ratio:
        leaf_dilated = cv2.dilate(leaf_mask, np.ones((13, 13), np.uint8), iterations=1)
        suspected_mask = cv2.bitwise_and(suspected_mask, leaf_dilated)
        denominator = max(leaf_area, 1)
    else:
        denominator = max(total_area, 1)

    suspected_mask = cv2.morphologyEx(suspected_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    suspected_area = int(np.count_nonzero(suspected_mask))
    suspected_ratio = suspected_area / denominator
    disease_score = min(100.0, suspected_ratio / max(disease_threshold, 1e-6) * 60.0)

    severity = severity_by_threshold(suspected_ratio, disease_threshold, disease_threshold * 2.0, disease_threshold * 3.5)
    is_suspected = suspected_ratio >= disease_threshold and (leaf_area_ratio >= min_leaf_area_ratio or suspected_area > 500)

    annotated = img.copy()
    red_overlay = np.zeros_like(annotated)
    red_overlay[:, :, 2] = 255
    mask_bool = suspected_mask > 0
    annotated[mask_bool] = cv2.addWeighted(annotated, 0.45, red_overlay, 0.55, 0)[mask_bool]

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    status_text = "疑似病害/黄化/枯斑" if is_suspected else "未触发疑似病害事件"
    cv2.putText(
        annotated,
        f"Agriculture MVP: {status_text}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )

    annotated_path = evidence_dir / "crop_disease_overlay.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    metrics = {
        "source_id": source_id,
        "leaf_area_ratio": leaf_area_ratio,
        "suspected_area_ratio": suspected_ratio,
        "disease_score": disease_score,
        "threshold": disease_threshold,
        "is_suspected": bool(is_suspected),
        "severity": int(severity),
        "status": status_text,
        "annotated_image": str(annotated_path),
        "note": "MVP启发式判断，不等同于专业病害诊断。",
    }

    events: list[Event] = []
    if is_suspected:
        events.append(Event(
            event_id=make_event_id("crop_disease"),
            time_s=0.0,
            category="crop_disease_suspected",
            severity=int(severity),
            confidence=float(min(0.95, max(0.35, disease_score / 100.0))),
            target=routing.get("crop_disease_suspected", "农业运维/巡检人员"),
            message=(
                f"疑似作物病害/黄化/枯斑：异常区域占比约 {suspected_ratio * 100:.2f}% ，"
                "建议人工复核。"
            ),
            evidence_path=str(annotated_path),
            extras={
                "leaf_area_ratio": leaf_area_ratio,
                "suspected_area_ratio": suspected_ratio,
                "disease_score": disease_score,
                "threshold": disease_threshold,
            },
            source="agriculture_image",
            source_id=source_id,
            status="待农业人员复核",
        ))

    metrics_json = out_dir_path / "agriculture_metrics.json"
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_zh = pd.DataFrame([{
        "来源ID": source_id,
        "叶片区域占比(%)": round(leaf_area_ratio * 100, 2),
        "疑似异常区域占比(%)": round(suspected_ratio * 100, 2),
        "疑似病害分数(0-100)": round(disease_score, 2),
        "触发阈值(%)": round(disease_threshold * 100, 2),
        "是否触发疑似病害": "是" if is_suspected else "否",
        "严重等级数值": int(severity),
        "状态说明": status_text,
        "说明": "MVP启发式判断，不等同于专业病害诊断。",
    }])
    metrics_zh_csv = out_dir_path / "agriculture_metrics_zh.csv"
    metrics_zh.to_csv(metrics_zh_csv, index=False, encoding="utf-8-sig")

    event_paths = write_event_outputs(events, out_dir_path, run_id=run_id, prefix="agriculture")

    paths = {
        "annotated_image": str(annotated_path),
        "agriculture_metrics_json": str(metrics_json),
        "agriculture_metrics_zh_csv": str(metrics_zh_csv),
        **event_paths,
    }
    return metrics, events, paths
