from __future__ import annotations

import ast
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from agriculture_core import analyze_crop_image
from event_engine import events_to_zh_dataframe, dispatch_to_zh_dataframe
from perception_core import metrics_to_zh_dataframe, process_video


st.set_page_config(page_title="IASR空中态势识别集成系统", layout="wide")
st.title("IASR 空中态势识别集成系统")
st.caption("态势识别 · 预警路由 · 接口扩展 · 结果导出")

OUTPUT_ROOT = Path("outputs") / "runs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def create_run_dir(prefix: str) -> tuple[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{prefix}_{ts}"
    out_dir = OUTPUT_ROOT / base
    suffix = 1
    while out_dir.exists():
        out_dir = OUTPUT_ROOT / f"{base}_{suffix}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir.name, out_dir


def parse_tuple(text: str, expected_len: int, name: str) -> Optional[tuple[int, ...]]:
    if not text.strip():
        return None
    try:
        value = ast.literal_eval(text)
        if not isinstance(value, (tuple, list)) or len(value) != expected_len:
            raise ValueError
        return tuple(int(v) for v in value)
    except Exception as exc:
        raise ValueError(f"{name} 格式错误，请填写类似 {('(0,0,100,100)' if expected_len == 4 else '(100,300)')} 的格式。") from exc


def parse_line(text: str) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    if not text.strip():
        return None, None
    try:
        value = ast.literal_eval(text)
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError
        a, b = value
        if len(a) != 2 or len(b) != 2:
            raise ValueError
        return (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
    except Exception as exc:
        raise ValueError("计数线格式错误，请填写 [(x1,y1),(x2,y2)]，例如 [(120,360),(640,360)]。") from exc


def parse_polygon(text: str) -> Optional[list[tuple[int, int]]]:
    if not text.strip():
        return None
    try:
        value = ast.literal_eval(text)
        if not isinstance(value, (tuple, list)) or len(value) < 3:
            raise ValueError
        poly = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError
            poly.append((int(item[0]), int(item[1])))
        return poly
    except Exception as exc:
        raise ValueError("敏感区多边形格式错误，请填写 [(x1,y1),(x2,y2),(x3,y3)]。") from exc


def safe_read_bytes(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def show_download(path: str | Path, label: str, file_name: Optional[str] = None, mime: Optional[str] = None) -> None:
    path = Path(path)
    if path.exists():
        st.download_button(label, data=safe_read_bytes(path), file_name=file_name or path.name, mime=mime)


def display_error(exc: Exception) -> None:
    st.error(str(exc))
    if isinstance(exc, FileNotFoundError) and "yolov8n.pt" in str(exc):
        st.info("解决方法：把 yolov8n.pt 放到项目根目录或 models/ 文件夹；如果网络稳定，也可在侧边栏勾选“允许自动下载模型”。")
        st.code("IASR_system/\n  app.py\n  perception_core.py\n  yolov8n.pt   # 放这里\n", language="text")
    else:
        with st.expander("查看调试信息"):
            st.exception(exc)


SCENE_PRESETS = {
    "通用模式": {
        "conf": 0.35,
        "bird_thresh": 6,
        "visibility": 120.0,
        "stop_speed": 8.0,
        "stop_sustain": 2.0,
        "cooldown": 8.0,
    },
    "交通态势优先": {
        "conf": 0.30,
        "bird_thresh": 8,
        "visibility": 100.0,
        "stop_speed": 10.0,
        "stop_sustain": 1.5,
        "cooldown": 6.0,
    },
    "低能见度预警优先": {
        "conf": 0.35,
        "bird_thresh": 8,
        "visibility": 180.0,
        "stop_speed": 8.0,
        "stop_sustain": 2.0,
        "cooldown": 6.0,
    },
    "敏感区入侵演示": {
        "conf": 0.25,
        "bird_thresh": 6,
        "visibility": 120.0,
        "stop_speed": 8.0,
        "stop_sustain": 2.0,
        "cooldown": 5.0,
    },
}


with st.sidebar:
    st.header("输入与运行")
    uploaded = st.file_uploader("上传视频（mp4 / mov / mkv）", type=["mp4", "mov", "mkv"])

    preset_name = st.selectbox("场景预设", list(SCENE_PRESETS.keys()), index=0)
    preset = SCENE_PRESETS[preset_name]

    with st.expander("运行模式", expanded=True):
        model_name = st.text_input("模型文件名/路径", value="yolov8n.pt")
        allow_download = st.checkbox("允许自动下载模型（网络不稳定时不建议）", value=False)
        conf = st.slider("检测置信度（高置信度可能导致处理时间加长）", 0.10, 0.80, float(preset["conf"]), 0.05)
        max_frames = st.number_input("最多处理帧数（0=全视频）", min_value=0, max_value=20000, value=600, step=50)
        frame_skip = st.slider("跳帧处理（1=不跳帧，2=每2帧取1帧）", 1, 10, 2, 1)
        resize_width = st.number_input("最大处理宽度(px，0=不缩放)", min_value=0, max_value=3840, value=960, step=80)
        source_id = st.text_input("视频ID（会记录在表单中）", value="camera_or_video_01")

    with st.expander("区域与空间配置", expanded=False):
        st.caption("坐标基于处理后画面。如果设置了最大处理宽度，坐标也按缩放后画面填写。留空则使用系统默认。")
        roi_text = st.text_input("ROI区域 (x1,y1,x2,y2)", value="")
        line_text = st.text_input("车辆计数线 [(x1,y1),(x2,y2)]", value="")
        poly_text = st.text_input("敏感/禁飞区多边形", value="")
        st.caption("多边形示例：[(100,100),(300,120),(280,300),(120,280)]")

    with st.expander("事件阈值", expanded=True):
        bird_thresh = st.slider("鸟群数量阈值", 1, 50, int(preset["bird_thresh"]), 1)
        bird_sustain = st.slider("鸟群持续时间(s)", 0.5, 10.0, 1.5, 0.5)
        visibility_low_thresh = st.slider("低能见度阈值（越高越敏感）", 20.0, 500.0, float(preset["visibility"]), 5.0)
        visibility_sustain = st.slider("低能见度持续时间(s)", 0.5, 10.0, 2.0, 0.5)
        stop_speed = st.slider("异常停车速度阈值(px/s)", 1.0, 40.0, float(preset["stop_speed"]), 1.0)
        stop_sustain = st.slider("异常停车持续时间(s)", 0.5, 10.0, float(preset["stop_sustain"]), 0.5)
        cooldown = st.slider("同类事件冷却时间(s)", 1.0, 30.0, float(preset["cooldown"]), 1.0)

    with st.expander("预警路由（可配置）", expanded=False):
        routing = {
            "bird_flock": st.text_input("鸟群风险 ->", "运行/飞行管控"),
            "intrusion": st.text_input("敏感区域入侵 ->", "安防/监管"),
            "visibility_low": st.text_input("低能见度 ->", "应急/运行调度"),
            "vehicle_stopped": st.text_input("异常停车 ->", "交管/运营"),
            "crop_disease_suspected": st.text_input("疑似作物病害 ->", "农业运维/巡检人员"),
        }

    run_btn = st.button("开始视频分析", use_container_width=True)


if run_btn:
    if not uploaded:
        st.warning("请先上传视频。")
        st.stop()

    try:
        roi = parse_tuple(roi_text, 4, "ROI")
        count_line_a, count_line_b = parse_line(line_text)
        forbidden_poly = parse_polygon(poly_text)

        run_id, out_dir = create_run_dir("video")
        input_suffix = Path(uploaded.name).suffix or ".mp4"
        in_path = out_dir / f"input{input_suffix}"
        in_path.write_bytes(uploaded.getbuffer())

        with st.spinner("正在分析：检测/跟踪/指标计算/事件预警/视频转码..."):
            df, events, out_video, paths = process_video(
                video_path=str(in_path),
                out_dir=str(out_dir),
                conf=conf,
                model_name=model_name,
                allow_model_download=allow_download,
                max_frames=(max_frames if max_frames > 0 else None),
                frame_skip=frame_skip,
                resize_width=(resize_width if resize_width > 0 else None),
                count_line_a=count_line_a,
                count_line_b=count_line_b,
                roi=roi,
                forbidden_poly=forbidden_poly,
                bird_thresh=bird_thresh,
                bird_sustain_s=bird_sustain,
                visibility_low_thresh=visibility_low_thresh,
                visibility_sustain_s=visibility_sustain,
                stop_speed_pxs=stop_speed,
                stop_sustain_s=stop_sustain,
                cooldown_s=cooldown,
                routing=routing,
                run_id=run_id,
                source_id=source_id,
            )

        st.session_state["video_result"] = {
            "run_id": run_id,
            "out_dir": str(out_dir),
            "df": df,
            "events": [e.to_dict() for e in events],
            "out_video": out_video,
            "paths": paths,
        }
        st.success(f"视频分析完成。本次输出目录：{out_dir}")
    except Exception as exc:
        display_error(exc)


video_result = st.session_state.get("video_result")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["交通态势", "预警中心", "天气态势", "农业扩展", "输出文件"])

with tab1:
    st.subheader("交通态势识别结果")
    if not video_result:
        st.info("请先上传视频并点击“开始视频分析”。")
    else:
        df: pd.DataFrame = video_result["df"]
        out_video = video_result["out_video"]
        paths = video_result["paths"]

        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.write("**标注后视频**")
            if out_video and Path(out_video).exists():
                st.video(out_video)
            else:
                st.warning("标注视频不存在，请检查输出目录。")
        with c2:
            st.write("**态势摘要**")
            st.metric("末帧拥堵指数", f"{df['congestion_index'].iloc[-1]:.1f}")
            st.metric("末帧拥堵等级", str(df["congestion_level"].iloc[-1]))
            st.metric("事件数量", len(video_result["events"]))
            st.caption("拥堵指数综合流量↑、占有率↑、速度↓，用于把现场状态转成可比较的数字信号。")

        st.write("**指标曲线**")
        chart_df = pd.DataFrame({
            "时间(s)": df["time_s"].round(2),
            "流量(辆/分钟)": df["flow_vpm"],
            "占有率(%)": (df["occ_ratio"] * 100).round(2),
            "相对速度(px/s)": df["speed_rel_pxs"].round(2),
            "拥堵指数(0-100)": df["congestion_index"].round(2),
        }).set_index("时间(s)")
        st.line_chart(chart_df)

        st.write("**中文化指标表（最近20行）**")
        df_zh = metrics_to_zh_dataframe(df)
        st.dataframe(df_zh.tail(20), use_container_width=True)
        show_download(paths["metrics_zh_csv"], "下载中文指标表 metrics_zh.csv", "metrics_zh.csv", "text/csv")

with tab2:
    st.subheader("预警中心：事件输出与联动派单")
    if not video_result:
        st.info("暂无事件。请先运行视频分析。")
    else:
        events = video_result["events"]
        paths = video_result["paths"]
        if not events:
            st.info("本次运行未触发事件。可以适当降低阈值或设置敏感区多边形。")
        else:
            ev_zh = events_to_zh_dataframe(events)
            st.write("**事件列表（中文）**")
            st.dataframe(ev_zh, use_container_width=True)

            dispatch_zh = dispatch_to_zh_dataframe(events, run_id=video_result["run_id"])
            st.write("**模拟联动派单日志**")
            st.dataframe(dispatch_zh, use_container_width=True)

            selected = st.selectbox("查看事件证据", ev_zh["事件ID"].tolist())
            selected_row = ev_zh[ev_zh["事件ID"] == selected].iloc[0]
            st.markdown(f"**事件说明：** {selected_row['事件说明']}")
            st.markdown(f"**建议动作：** {selected_row['建议动作']}")
            evidence = selected_row.get("证据路径", "")
            if evidence and Path(evidence).exists():
                st.image(evidence, caption="事件证据关键帧")

        c1, c2, c3 = st.columns(3)
        with c1:
            show_download(paths["events_json"], "下载 events.json", "events.json", "application/json")
        with c2:
            show_download(paths["events_zh_csv"], "下载中文事件表", "events_zh.csv", "text/csv")
        with c3:
            show_download(paths["dispatch_zh_csv"], "下载中文派单日志", "dispatch_log_zh.csv", "text/csv")

with tab3:
    st.subheader("天气态势：能见度估计")
    if not video_result:
        st.info("请先运行视频分析。")
    else:
        df = video_result["df"]
        weather_df = pd.DataFrame({
            "时间(s)": df["time_s"].round(2),
            "能见度分数": df["visibility_score"].round(2),
        }).set_index("时间(s)")
        st.line_chart(weather_df)

        vis_last = float(df["visibility_score"].iloc[-1])
        if vis_last < 80:
            level = "较差"
            advice = "建议低空运行谨慎，提高间隔；必要时延误或转入人工复核。"
        elif vis_last < 140:
            level = "一般"
            advice = "注意局部雾霾/烟尘影响，建议结合人工观察与气象数据复核。"
        else:
            level = "良好"
            advice = "能见度状态较好，可作为低风险参考。"
        st.markdown(f"""
**天气态势报告（MVP）**
- 能见度等级：**{level}**
- 当前能见度分数：**{vis_last:.1f}**
- 运行建议：{advice}

> 注：这是基于视频清晰度/对比度的态势估计，不等同于专业气象预报。
""")

with tab4:
    st.subheader("农业扩展接口：疑似作物病害识别（MVP）")
    st.caption("先实现“疑似病害/黄化/枯斑”事件，不做专业病害分类。输出会接入同一套事件与联动派单机制。")

    crop_img = st.file_uploader("上传作物/叶片图片（jpg/png/jpeg）", type=["jpg", "jpeg", "png"], key="crop_img")
    c1, c2, c3 = st.columns(3)
    with c1:
        disease_threshold = st.slider("疑似异常区域阈值(%)", 1.0, 30.0, 8.0, 0.5) / 100.0
    with c2:
        min_leaf_area_ratio = st.slider("最小叶片区域占比(%)", 1.0, 30.0, 5.0, 0.5) / 100.0
    with c3:
        crop_source_id = st.text_input("农业样本ID", "crop_image_01")

    if st.button("开始农业扩展分析", use_container_width=True):
        if not crop_img:
            st.warning("请先上传作物/叶片图片。")
        else:
            try:
                run_id, out_dir = create_run_dir("agriculture")
                img_suffix = Path(crop_img.name).suffix or ".jpg"
                img_path = out_dir / f"crop_input{img_suffix}"
                img_path.write_bytes(crop_img.getbuffer())
                with st.spinner("正在分析作物图像..."):
                    metrics, crop_events, crop_paths = analyze_crop_image(
                        image_path=str(img_path),
                        out_dir=str(out_dir),
                        disease_threshold=disease_threshold,
                        min_leaf_area_ratio=min_leaf_area_ratio,
                        routing=routing,
                        run_id=run_id,
                        source_id=crop_source_id,
                    )
                st.session_state["crop_result"] = {
                    "run_id": run_id,
                    "out_dir": str(out_dir),
                    "metrics": metrics,
                    "events": [e.to_dict() for e in crop_events],
                    "paths": crop_paths,
                }
                st.success(f"农业扩展分析完成。本次输出目录：{out_dir}")
            except Exception as exc:
                display_error(exc)

    crop_result = st.session_state.get("crop_result")
    if crop_result:
        paths = crop_result["paths"]
        metrics = crop_result["metrics"]
        st.write("**分析结果**")
        c1, c2, c3 = st.columns(3)
        c1.metric("是否触发疑似病害", "是" if metrics["is_suspected"] else "否")
        c2.metric("疑似异常区域占比", f"{metrics['suspected_area_ratio'] * 100:.2f}%")
        c3.metric("疑似病害分数", f"{metrics['disease_score']:.1f}")

        if Path(paths["annotated_image"]).exists():
            st.image(paths["annotated_image"], caption="农业扩展识别结果：绿色为叶片轮廓，红色为疑似异常区域")

        crop_events = crop_result["events"]
        if crop_events:
            st.write("**农业事件与联动派单**")
            st.dataframe(events_to_zh_dataframe(crop_events), use_container_width=True)
            st.dataframe(dispatch_to_zh_dataframe(crop_events, run_id=crop_result["run_id"]), use_container_width=True)
        else:
            st.info("未触发疑似病害事件；仍可下载指标表用于记录。")

        c1, c2, c3 = st.columns(3)
        with c1:
            show_download(paths["agriculture_metrics_zh_csv"], "下载农业中文指标表", "agriculture_metrics_zh.csv", "text/csv")
        with c2:
            show_download(paths["events_zh_csv"], "下载农业事件表", "agriculture_events_zh.csv", "text/csv")
        with c3:
            show_download(paths["dispatch_zh_csv"], "下载农业派单日志", "agriculture_dispatch_log_zh.csv", "text/csv")

with tab5:
    st.subheader("输出文件与运行记录")
    if not video_result and not st.session_state.get("crop_result"):
        st.info("运行后会在这里显示本次输出目录。所有输出都写入 outputs/runs/ 下的独立文件夹，不会覆盖历史结果。")
    if video_result:
        st.write("**最近一次视频分析输出**")
        st.code(video_result["out_dir"], language="text")
        paths = video_result["paths"]
        for key, value in paths.items():
            st.write(f"- `{key}`: `{value}`")
        show_download(paths["run_config_json"], "下载运行配置 run_config.json", "run_config.json", "application/json")
    if st.session_state.get("crop_result"):
        crop_result = st.session_state["crop_result"]
        st.write("**最近一次农业扩展输出**")
        st.code(crop_result["out_dir"], language="text")
        for key, value in crop_result["paths"].items():
            st.write(f"- `{key}`: `{value}`")
