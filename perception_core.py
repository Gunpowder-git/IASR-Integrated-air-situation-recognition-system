from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from event_engine import Event, Cooldown, make_event_id, severity_by_threshold, write_event_outputs

# COCO 常见类：person=0, car=2, motorcycle=3, airplane=4, bus=5, truck=7, bird=14
VEHICLE_CLS = {2, 3, 5, 7}
AIRBORNE_CLS = {4, 14}


METRIC_COLUMN_ZH = {
    "time_s": "时间(s)",
    "frame_index": "处理帧序号",
    "raw_frame_index": "原始帧序号",
    "flow_vpm": "流量(辆/分钟)",
    "vehicle_count_roi": "ROI内车辆数",
    "occ_ratio": "占有率",
    "occ_ratio_pct": "占有率(%)",
    "speed_rel_pxs": "平均相对速度(px/s)",
    "bird_count": "鸟类目标数",
    "visibility_score": "能见度分数",
    "airborne_in_forbidden": "敏感区内空中目标数",
    "congestion_index": "拥堵指数(0-100)",
    "congestion_level": "拥堵等级",
}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def point_in_polygon(x: float, y: float, poly: List[Tuple[int, int]]) -> bool:
    """Ray casting 算法，判断点是否在多边形内。"""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside


def laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def contrast_std(gray: np.ndarray) -> float:
    return float(np.std(gray))


def norm_series(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-6:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def level_ci(x: float) -> str:
    if x < 40:
        return "畅通"
    if x < 70:
        return "缓行"
    return "拥堵"


def add_congestion_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["congestion_index"] = []
        df["congestion_level"] = []
        return df
    flow_n = norm_series(df["flow_vpm"])
    occ_n = norm_series(df["occ_ratio"])
    spd_n = norm_series(df["speed_rel_pxs"])
    df = df.copy()
    df["congestion_index"] = 100.0 * (0.35 * flow_n + 0.45 * occ_n + 0.20 * (1.0 - spd_n))
    df["congestion_level"] = df["congestion_index"].apply(level_ci)
    return df


def metrics_to_zh_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    display = df.copy()
    if "occ_ratio" in display.columns:
        display["occ_ratio_pct"] = (display["occ_ratio"] * 100).round(2)
    numeric_round_cols = ["time_s", "speed_rel_pxs", "visibility_score", "congestion_index", "occ_ratio"]
    for col in numeric_round_cols:
        if col in display.columns:
            display[col] = display[col].astype(float).round(2)
    keep_order = [
        "time_s", "frame_index", "raw_frame_index", "flow_vpm", "vehicle_count_roi",
        "occ_ratio_pct", "speed_rel_pxs", "bird_count", "visibility_score",
        "airborne_in_forbidden", "congestion_index", "congestion_level",
    ]
    keep_cols = [c for c in keep_order if c in display.columns]
    return display[keep_cols].rename(columns=METRIC_COLUMN_ZH)


def resolve_model_path(model_name: str = "yolov8n.pt", allow_download: bool = False) -> str:
    """优先使用本地模型，避免现场因网络失败而报错。"""
    candidates = [
        Path(model_name),
        Path.cwd() / model_name,
        Path(__file__).resolve().parent / model_name,
        Path(__file__).resolve().parent / "models" / model_name,
        Path.cwd() / "models" / model_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    if allow_download:
        return model_name

    raise FileNotFoundError(
        "未找到 YOLO 模型权重文件。请将 yolov8n.pt 放到项目根目录或 models/ 文件夹，"
        "或在侧边栏勾选允许自动下载模型。"
    )


def transcode_to_h264(src: str, dst: str) -> None:
    """转码为浏览器兼容的 H.264。失败时给出明确错误。"""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("缺少 imageio-ffmpeg，无法转码视频。请运行：python -m pip install imageio-ffmpeg") from exc

    cmd = [
        ffmpeg, "-y", "-i", src, "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError("视频转码失败：请检查 ffmpeg/imageio-ffmpeg 是否可用，或尝试缩短视频。")


def _resize_frame(frame: np.ndarray, resize_width: Optional[int]) -> np.ndarray:
    if not resize_width or resize_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame
    scale = resize_width / float(w)
    return cv2.resize(frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _sanitize_roi(roi: Optional[Tuple[int, int, int, int]], w: int, h: int) -> Tuple[int, int, int, int]:
    if roi is None:
        return (0, 0, w, h)
    x1, y1, x2, y2 = roi
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(1, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI 区域无效：请使用 (x1,y1,x2,y2)，且 x2>x1、y2>y1。")
    return (x1, y1, x2, y2)


def _default_count_line(w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (int(w * 0.20), int(h * 0.65)), (int(w * 0.80), int(h * 0.65))


def process_video(
    video_path: str,
    out_dir: str,
    conf: float = 0.35,
    model_name: str = "yolov8n.pt",
    allow_model_download: bool = False,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    resize_width: Optional[int] = None,
    count_line_a: Optional[Tuple[int, int]] = None,
    count_line_b: Optional[Tuple[int, int]] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    forbidden_poly: Optional[List[Tuple[int, int]]] = None,
    bird_thresh: int = 6,
    bird_sustain_s: float = 1.5,
    visibility_low_thresh: float = 120.0,
    visibility_sustain_s: float = 2.0,
    stop_speed_pxs: float = 8.0,
    stop_sustain_s: float = 2.0,
    cooldown_s: float = 8.0,
    routing: Optional[Dict[str, str]] = None,
    run_id: str = "",
    source_id: str = "video_01",
) -> tuple[pd.DataFrame, List[Event], str, dict]:
    """视频态势识别主流程。"""
    out_dir_path = Path(out_dir)
    ensure_dir(out_dir_path)
    evidence_dir = out_dir_path / "evidence"
    ensure_dir(evidence_dir)

    if frame_skip < 1:
        frame_skip = 1

    if routing is None:
        routing = {
            "bird_flock": "运行/飞行管控",
            "intrusion": "安防/监管",
            "visibility_low": "应急/运行调度",
            "vehicle_stopped": "交管/运营",
        }

    if not Path(video_path).exists():
        raise FileNotFoundError(f"未找到视频文件：{video_path}")

    model_path = resolve_model_path(model_name, allow_download=allow_model_download)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("视频打开失败：请确认格式为 mp4/mov/mkv，且文件未损坏。")

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("视频没有可读取帧：请换一个更短或格式更标准的视频。")
    first_frame = _resize_frame(first_frame, resize_width)
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    processed_fps = max(raw_fps / frame_skip, 1.0)
    out_raw = out_dir_path / "annotated_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_raw), fourcc, processed_fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter 打不开：当前环境缺少可用 mp4 编码器。")

    roi = _sanitize_roi(roi, w, h)
    rx1, ry1, rx2, ry2 = roi

    if count_line_a is None or count_line_b is None:
        count_line_a, count_line_b = _default_count_line(w, h)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("未安装 ultralytics，无法运行 YOLO 检测。请运行：python -m pip install ultralytics") from exc

    model = YOLO(model_path)

    def side_of_line(p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.sign((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))

    last_side: Dict[int, float] = {}
    counted_ids = set()
    crossings_ts: List[float] = []
    last_pos: Dict[int, Tuple[int, int, float]] = {}
    stopped_time: Dict[int, float] = {}
    stop_emitted: set[int] = set()
    bird_high_since: Optional[float] = None
    vis_low_since: Optional[float] = None
    intrusion_emitted: set[int] = set()
    cooldown = Cooldown()
    events: List[Event] = []
    rows = []

    raw_frame_idx = 0
    processed_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_frame_idx % frame_skip != 0:
            raw_frame_idx += 1
            continue

        frame = _resize_frame(frame, resize_width)
        t = raw_frame_idx / raw_fps
        annotated = frame.copy()

        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        cv2.line(annotated, count_line_a, count_line_b, (255, 255, 255), 2)
        if forbidden_poly and len(forbidden_poly) >= 3:
            cv2.polylines(annotated, [np.array(forbidden_poly, dtype=np.int32)], True, (255, 255, 255), 2)

        results = model.track(frame, conf=conf, iou=0.5, persist=True, verbose=False)

        veh_count_in_roi = 0
        occ_area_sum = 0.0
        veh_speeds = []
        bird_count = 0
        airborne_in_forbidden = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for b in boxes:
                cls = int(b.cls[0].item())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                tid = int(b.id[0].item()) if b.id is not None else None
                obj_conf = float(b.conf[0].item()) if b.conf is not None else 0.6

                if cls == 14:
                    bird_count += 1

                if forbidden_poly and cls in AIRBORNE_CLS and point_in_polygon(cx, cy, forbidden_poly):
                    airborne_in_forbidden += 1
                    if tid is not None and tid not in intrusion_emitted:
                        if cooldown.allow("intrusion", t, cooldown_s):
                            ev_path = evidence_dir / f"intrusion_{processed_frame_idx}.jpg"
                            cv2.imwrite(str(ev_path), frame)
                            events.append(Event(
                                event_id=make_event_id("intrusion"),
                                time_s=t,
                                category="intrusion",
                                severity=2,
                                confidence=obj_conf,
                                target=routing.get("intrusion", "安防/监管"),
                                message="检测到空中目标进入禁飞/敏感区域（需人工复核）。",
                                evidence_path=str(ev_path),
                                extras={"cls": cls, "tid": tid, "center": [cx, cy]},
                                source="video",
                                source_id=source_id,
                                status="待安防/监管复核",
                            ))
                        intrusion_emitted.add(tid)

                if cls in VEHICLE_CLS:
                    in_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
                    if in_roi:
                        veh_count_in_roi += 1
                        occ_area_sum += float(max(0, x2 - x1) * max(0, y2 - y1))

                    if tid is not None:
                        if tid in last_pos:
                            px, py, pt = last_pos[tid]
                            dt = max(t - pt, 1e-6)
                            v_rel = math.hypot(cx - px, cy - py) / dt
                            veh_speeds.append(v_rel)

                            if in_roi and v_rel < stop_speed_pxs:
                                stopped_time[tid] = stopped_time.get(tid, 0.0) + dt
                            else:
                                stopped_time[tid] = 0.0

                            if (tid not in stop_emitted) and stopped_time.get(tid, 0.0) >= stop_sustain_s:
                                if cooldown.allow("vehicle_stopped", t, cooldown_s):
                                    ev_path = evidence_dir / f"vehicle_stop_{processed_frame_idx}.jpg"
                                    cv2.imwrite(str(ev_path), frame)
                                    sev = severity_by_threshold(stopped_time[tid], 1.0, 2.0, 4.0)
                                    events.append(Event(
                                        event_id=make_event_id("vehicle_stopped"),
                                        time_s=t,
                                        category="vehicle_stopped",
                                        severity=sev,
                                        confidence=0.75,
                                        target=routing.get("vehicle_stopped", "交管/运营"),
                                        message=f"疑似异常停车：车辆ID {tid} 已低速/停止约 {stopped_time[tid]:.1f}s（需人工复核）。",
                                        evidence_path=str(ev_path),
                                        extras={"tid": tid, "stopped_s": stopped_time[tid]},
                                        source="video",
                                        source_id=source_id,
                                        status="待交管/运营复核",
                                    ))
                                stop_emitted.add(tid)

                        last_pos[tid] = (cx, cy, t)

                        side_now = side_of_line((cx, cy), count_line_a, count_line_b)
                        side_prev = last_side.get(tid, side_now)
                        last_side[tid] = side_now
                        if tid not in counted_ids and side_prev != 0 and side_now != 0 and side_prev != side_now:
                            counted_ids.add(tid)
                            crossings_ts.append(t)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                if tid is not None:
                    cv2.putText(annotated, f"ID {tid}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        crossings_last_60 = [ts for ts in crossings_ts if t - ts <= 60.0]
        flow_vpm = len(crossings_last_60)
        roi_area = float((rx2 - rx1) * (ry2 - ry1))
        occ_ratio = min(occ_area_sum / max(roi_area, 1.0), 1.0)
        speed_mean = float(np.mean(veh_speeds)) if veh_speeds else 0.0

        roi_frame = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        vis_lap = laplacian_var(gray)
        vis_ctr = contrast_std(gray)
        visibility_score = 0.7 * vis_lap + 0.3 * vis_ctr

        if bird_count >= bird_thresh:
            bird_high_since = bird_high_since or t
            if (t - bird_high_since) >= bird_sustain_s:
                if cooldown.allow("bird_flock", t, cooldown_s):
                    ev_path = evidence_dir / f"bird_{processed_frame_idx}.jpg"
                    cv2.imwrite(str(ev_path), frame)
                    sev = severity_by_threshold(bird_count, 3, 6, 12)
                    events.append(Event(
                        event_id=make_event_id("bird_flock"),
                        time_s=t,
                        category="bird_flock",
                        severity=sev,
                        confidence=0.65,
                        target=routing.get("bird_flock", "运行/飞行管控"),
                        message=f"疑似鸟群风险：检测到 bird 数量={bird_count}（建议减速/绕行/复核）。",
                        evidence_path=str(ev_path),
                        extras={"bird_count": bird_count},
                        source="video",
                        source_id=source_id,
                        status="待运行/飞行管控复核",
                    ))
                bird_high_since = None
        else:
            bird_high_since = None

        if visibility_score <= visibility_low_thresh:
            vis_low_since = vis_low_since or t
            if (t - vis_low_since) >= visibility_sustain_s:
                if cooldown.allow("visibility_low", t, cooldown_s):
                    ev_path = evidence_dir / f"visibility_{processed_frame_idx}.jpg"
                    cv2.imwrite(str(ev_path), frame)
                    sev = severity_by_threshold(visibility_low_thresh - visibility_score, 20, 60, 120)
                    events.append(Event(
                        event_id=make_event_id("visibility_low"),
                        time_s=t,
                        category="visibility_low",
                        severity=sev,
                        confidence=0.7,
                        target=routing.get("visibility_low", "应急/运行调度"),
                        message=f"能见度下降：visibility_score={visibility_score:.1f}（可能雾/霾/烟，建议谨慎运行）。",
                        evidence_path=str(ev_path),
                        extras={"visibility_score": visibility_score},
                        source="video",
                        source_id=source_id,
                        status="待应急/运行调度复核",
                    ))
                vis_low_since = None
        else:
            vis_low_since = None

        cv2.putText(annotated, f"flow(vpm): {flow_vpm}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"occ_ratio: {occ_ratio:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"speed_rel(pxs): {speed_mean:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"birds: {bird_count}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"vis_score: {visibility_score:.1f}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out.write(annotated)
        rows.append({
            "time_s": t,
            "frame_index": processed_frame_idx,
            "raw_frame_index": raw_frame_idx,
            "flow_vpm": flow_vpm,
            "vehicle_count_roi": veh_count_in_roi,
            "occ_ratio": occ_ratio,
            "speed_rel_pxs": speed_mean,
            "bird_count": bird_count,
            "visibility_score": visibility_score,
            "airborne_in_forbidden": airborne_in_forbidden,
        })

        processed_frame_idx += 1
        raw_frame_idx += 1
        if max_frames and processed_frame_idx >= max_frames:
            break

    cap.release()
    out.release()

    if not rows:
        raise RuntimeError("没有处理到有效帧：请检查 frame_skip/max_frames 设置，或更换视频。")

    df = pd.DataFrame(rows)
    df = add_congestion_columns(df)

    out_h264 = out_dir_path / "annotated_h264.mp4"
    try:
        transcode_to_h264(str(out_raw), str(out_h264))
    except RuntimeError:
        # 兜底：如果转码失败，仍保留 raw 视频路径，避免整个流程白跑。
        shutil.copyfile(out_raw, out_h264)
        raise

    metrics_csv = out_dir_path / "metrics.csv"
    metrics_zh_csv = out_dir_path / "metrics_zh.csv"
    df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    metrics_to_zh_dataframe(df).to_csv(metrics_zh_csv, index=False, encoding="utf-8-sig")

    event_paths = write_event_outputs(events, out_dir_path, run_id=run_id)

    run_config = {
        "run_id": run_id,
        "source_id": source_id,
        "video_path": video_path,
        "model_name": model_name,
        "model_path": model_path,
        "conf": conf,
        "max_frames": max_frames,
        "frame_skip": frame_skip,
        "resize_width": resize_width,
        "roi": roi,
        "count_line_a": count_line_a,
        "count_line_b": count_line_b,
        "forbidden_poly": forbidden_poly,
        "thresholds": {
            "bird_thresh": bird_thresh,
            "bird_sustain_s": bird_sustain_s,
            "visibility_low_thresh": visibility_low_thresh,
            "visibility_sustain_s": visibility_sustain_s,
            "stop_speed_pxs": stop_speed_pxs,
            "stop_sustain_s": stop_sustain_s,
            "cooldown_s": cooldown_s,
        },
        "routing": routing,
    }
    run_config_json = out_dir_path / "run_config.json"
    run_config_json.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = {
        "out_raw": str(out_raw),
        "out_h264": str(out_h264),
        "metrics_csv": str(metrics_csv),
        "metrics_zh_csv": str(metrics_zh_csv),
        "evidence_dir": str(evidence_dir),
        "run_config_json": str(run_config_json),
        **event_paths,
    }
    return df, events, str(out_h264), paths
