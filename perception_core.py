from __future__ import annotations
import os
import cv2
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO
from event_engine import Event, make_event_id, Cooldown, severity_by_threshold

# COCO 常见类：car=2, motorcycle=3, bus=5, truck=7, bird=14, airplane=4
VEHICLE_CLS = {2, 3, 5, 7}
AIRBORNE_CLS = {4, 14}  # airplane + bird

def point_in_polygon(x: float, y: float, poly: List[Tuple[int, int]]) -> bool:
    """Ray casting 算法，判断点是否在多边形内"""
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
    """清晰度指标：方差越大越清晰；雾霾/虚焦会下降"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def contrast_std(gray: np.ndarray) -> float:
    """对比度指标：标准差越大越清晰；雾霾会下降"""
    return float(np.std(gray))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def transcode_to_h264(src: str, dst: str):
    """用 imageio-ffmpeg 内置 ffmpeg 做 H.264 转码，保证浏览器可播"""
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", src,
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst
    ]
    subprocess.run(cmd, check=True)

def process_video(
    video_path: str,
    out_dir: str,
    conf: float = 0.35,
    model_name: str = "yolov8n.pt",
    max_frames: Optional[int] = None,
    # 交通计数线（默认一条水平线）
    count_line_a: Tuple[int, int] = (120, 360),
    count_line_b: Tuple[int, int] = (640, 360),
    # ROI（默认全画面）
    roi: Optional[Tuple[int, int, int, int]] = None,
    # 禁飞区多边形（用于“入侵预警”）
    forbidden_poly: Optional[List[Tuple[int, int]]] = None,
    # 事件阈值
    bird_thresh: int = 6,              # 鸟数量阈值（>=触发）
    bird_sustain_s: float = 1.5,       # 持续时间
    visibility_low_thresh: float = 120.0,  # 清晰度阈值（越低越“雾/霾/烟”）
    visibility_sustain_s: float = 2.0,
    stop_speed_pxs: float = 8.0,       # 低于这个像素速度认为“近似停止”
    stop_sustain_s: float = 2.0,
    cooldown_s: float = 8.0,
    # 路由映射（你可在 app.py 里改）
    routing: Optional[Dict[str, str]] = None,
):
    """
    返回：
      df_metrics: 指标表（交通+鸟+能见度）
      events: 事件列表（统一schema）
      out_video_h264: 浏览器可播放的标注视频
      paths: 一些输出文件路径
    """
    ensure_dir(out_dir)
    evidence_dir = os.path.join(out_dir, "evidence")
    ensure_dir(evidence_dir)

    if routing is None:
        routing = {
            "bird_flock": "运行/飞行管控",
            "intrusion": "安防/监管",
            "visibility_low": "应急/运行调度",
            "vehicle_stopped": "交管/运营"
        }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if roi is None:
        roi = (0, 0, w, h)
    rx1, ry1, rx2, ry2 = roi

    # 输出 raw mp4（OpenCV写，可能浏览器不认，后面转码）
    out_raw = os.path.join(out_dir, "annotated_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_raw, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter 打不开：当前环境缺少可用mp4编码器。")

    model = YOLO(model_name)

    # 交通过线计数：track_id -> last_side
    def side_of_line(p, a, b):
        return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))

    last_side: Dict[int, float] = {}
    counted_ids = set()
    crossings_ts: List[float] = []

    # 速度/停止检测：track_id -> (x,y,t), stopped_time
    last_pos: Dict[int, Tuple[int, int, float]] = {}
    stopped_time: Dict[int, float] = {}
    stop_emitted: set[int] = set()

    # 鸟群持续检测
    bird_high_since: Optional[float] = None

    # 能见度持续检测
    vis_low_since: Optional[float] = None

    # 入侵检测
    intrusion_emitted: set[int] = set()

    cooldown = Cooldown()
    events: List[Event] = []
    rows = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        annotated = frame.copy()

        # ROI、计数线、禁飞区可视化
        cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        cv2.line(annotated, count_line_a, count_line_b, (255, 255, 255), 2)
        if forbidden_poly and len(forbidden_poly) >= 3:
            cv2.polylines(annotated, [np.array(forbidden_poly, dtype=np.int32)], True, (255, 255, 255), 2)

        # YOLO 跟踪
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

                # bird统计（不要求在ROI）
                if cls == 14:
                    bird_count += 1

                # 禁飞区入侵（只看空中目标：bird/airplane）
                if forbidden_poly and cls in AIRBORNE_CLS and point_in_polygon(cx, cy, forbidden_poly):
                    airborne_in_forbidden += 1
                    if tid is not None and tid not in intrusion_emitted:
                        # 防抖：同类事件 cooldown
                        if cooldown.allow("intrusion", t, cooldown_s):
                            ev_path = os.path.join(evidence_dir, f"intrusion_{frame_idx}.jpg")
                            cv2.imwrite(ev_path, frame)
                            e = Event(
                                event_id=make_event_id("intrusion"),
                                time_s=t,
                                category="intrusion",
                                severity=2,
                                confidence=float(b.conf[0].item()) if b.conf is not None else 0.6,
                                target=routing["intrusion"],
                                message="检测到空中目标进入禁飞/敏感区域（需人工复核）",
                                evidence_path=ev_path,
                                extras={"cls": cls, "tid": tid}
                            )
                            events.append(e)
                        intrusion_emitted.add(tid)

                # 交通指标（车辆类 + ROI内）
                if cls in VEHICLE_CLS:
                    in_roi = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
                    if in_roi:
                        veh_count_in_roi += 1
                        occ_area_sum += float((x2 - x1) * (y2 - y1))

                    # track速度 & 停止异常
                    if tid is not None:
                        if tid in last_pos:
                            px, py, pt = last_pos[tid]
                            dt = max(t - pt, 1e-6)
                            v_rel = math.hypot(cx - px, cy - py) / dt  # px/s
                            veh_speeds.append(v_rel)

                            # 停止累计（只统计在ROI附近的）
                            if in_roi and v_rel < stop_speed_pxs:
                                stopped_time[tid] = stopped_time.get(tid, 0.0) + dt
                            else:
                                stopped_time[tid] = 0.0

                            if (tid not in stop_emitted) and stopped_time.get(tid, 0.0) >= stop_sustain_s:
                                if cooldown.allow("vehicle_stopped", t, cooldown_s):
                                    ev_path = os.path.join(evidence_dir, f"vehicle_stop_{frame_idx}.jpg")
                                    cv2.imwrite(ev_path, frame)
                                    sev = severity_by_threshold(stopped_time[tid], 1.0, 2.0, 4.0)
                                    e = Event(
                                        event_id=make_event_id("vehicle_stopped"),
                                        time_s=t,
                                        category="vehicle_stopped",
                                        severity=sev,
                                        confidence=0.75,
                                        target=routing["vehicle_stopped"],
                                        message=f"疑似异常停车：车辆ID {tid} 已低速/停止约 {stopped_time[tid]:.1f}s（需人工复核）",
                                        evidence_path=ev_path,
                                        extras={"tid": tid, "stopped_s": stopped_time[tid]}
                                    )
                                    events.append(e)
                                stop_emitted.add(tid)

                        last_pos[tid] = (cx, cy, t)

                        # 过线计数（同ID只计一次）
                        side_now = side_of_line((cx, cy), count_line_a, count_line_b)
                        side_prev = last_side.get(tid, side_now)
                        last_side[tid] = side_now
                        if tid not in counted_ids and side_prev != 0 and side_now != 0 and side_prev != side_now:
                            counted_ids.add(tid)
                            crossings_ts.append(t)

                # 画框（简洁）
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
                if tid is not None:
                    cv2.putText(annotated, f"ID {tid}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 流量（过去60秒过线）
        crossings_last_60 = [ts for ts in crossings_ts if t - ts <= 60.0]
        flow_vpm = len(crossings_last_60)

        # 占有率（bbox面积/ROI面积）
        roi_area = float((rx2 - rx1) * (ry2 - ry1))
        occ_ratio = min(occ_area_sum / max(roi_area, 1.0), 1.0)

        speed_mean = float(np.mean(veh_speeds)) if veh_speeds else 0.0

        # 天气/能见度指标（取ROI内）
        roi_frame = frame[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        vis_lap = laplacian_var(gray)
        vis_ctr = contrast_std(gray)
        # 合成一个可用“能见度分数”（数值越大越清晰）
        visibility_score = 0.7 * vis_lap + 0.3 * vis_ctr

        # 鸟群预警（持续触发）
        if bird_count >= bird_thresh:
            bird_high_since = bird_high_since or t
            if (t - bird_high_since) >= bird_sustain_s:
                if cooldown.allow("bird_flock", t, cooldown_s):
                    ev_path = os.path.join(evidence_dir, f"bird_{frame_idx}.jpg")
                    cv2.imwrite(ev_path, frame)
                    sev = severity_by_threshold(bird_count, 3, 6, 12)
                    events.append(Event(
                        event_id=make_event_id("bird_flock"),
                        time_s=t,
                        category="bird_flock",
                        severity=sev,
                        confidence=0.65,
                        target=routing["bird_flock"],
                        message=f"疑似鸟群风险：检测到 bird 数量={bird_count}（建议减速/绕行/复核）",
                        evidence_path=ev_path,
                        extras={"bird_count": bird_count}
                    ))
                bird_high_since = None
        else:
            bird_high_since = None

        # 能见度/雾霾预警（持续触发）
        if visibility_score <= visibility_low_thresh:
            vis_low_since = vis_low_since or t
            if (t - vis_low_since) >= visibility_sustain_s:
                if cooldown.allow("visibility_low", t, cooldown_s):
                    ev_path = os.path.join(evidence_dir, f"visibility_{frame_idx}.jpg")
                    cv2.imwrite(ev_path, frame)
                    sev = severity_by_threshold(visibility_low_thresh - visibility_score, 20, 60, 120)
                    events.append(Event(
                        event_id=make_event_id("visibility_low"),
                        time_s=t,
                        category="visibility_low",
                        severity=sev,
                        confidence=0.7,
                        target=routing["visibility_low"],
                        message=f"能见度下降：visibility_score={visibility_score:.1f}（可能雾/霾/烟，建议谨慎运行）",
                        evidence_path=ev_path,
                        extras={"visibility_score": visibility_score}
                    ))
                vis_low_since = None
        else:
            vis_low_since = None

        # HUD
        cv2.putText(annotated, f"flow(vpm): {flow_vpm}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"occ_ratio: {occ_ratio:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"speed_rel(pxs): {speed_mean:.1f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"birds: {bird_count}", (10, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"vis_score: {visibility_score:.1f}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        out.write(annotated)

        rows.append({
            "time_s": t,
            "flow_vpm": flow_vpm,
            "occ_ratio": occ_ratio,
            "speed_rel_pxs": speed_mean,
            "bird_count": bird_count,
            "visibility_score": visibility_score,
            "airborne_in_forbidden": airborne_in_forbidden
        })

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()

    df = pd.DataFrame(rows)

    # 转码为浏览器可播放的 H264
    out_h264 = os.path.join(out_dir, "annotated_h264.mp4")
    transcode_to_h264(out_raw, out_h264)

    # 保存 metrics + events
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    df.to_csv(metrics_csv, index=False)

    events_json = os.path.join(out_dir, "events.json")
    with open(events_json, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in events], f, ensure_ascii=False, indent=2)

    paths = {
        "out_raw": out_raw,
        "out_h264": out_h264,
        "metrics_csv": metrics_csv,
        "events_json": events_json,
        "evidence_dir": evidence_dir
    }
    return df, events, out_h264, paths
