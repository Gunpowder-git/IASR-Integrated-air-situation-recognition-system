import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from perception_core import process_video

def norm_series(s: pd.Series):
    if len(s) == 0:
        return s
    mn, mx = float(s.min()), float(s.max())
    if abs(mx - mn) < 1e-6:
        return pd.Series(np.zeros(len(s)))
    return (s - mn) / (mx - mn)

def congestion_index(df: pd.DataFrame):
    # 拥堵指数：流量↑、占有率↑、速度↓
    flow_n = norm_series(df["flow_vpm"])
    occ_n = norm_series(df["occ_ratio"])
    spd_n = norm_series(df["speed_rel_pxs"])
    ci = 100.0 * (0.35*flow_n + 0.45*occ_n + 0.20*(1.0 - spd_n))
    return ci

def level_ci(x: float) -> str:
    if x < 40: return "畅通"
    if x < 70: return "缓行"
    return "拥堵"

st.set_page_config(page_title="IASR空中态势识别集成系统", layout="wide")
st.title("IASR空中态势识别集成系统")

os.makedirs("outputs", exist_ok=True)

with st.sidebar:
    st.header("输入与参数")
    uploaded = st.file_uploader("上传视频（mp4/mov）", type=["mp4", "mov", "mkv"])
    conf = st.slider("检测置信度", 0.1, 0.8, 0.35, 0.05)

    st.subheader("事件阈值")
    bird_thresh = st.slider("鸟数量阈值", 1, 30, 6, 1)
    visibility_low_thresh = st.slider("能见度阈值", 20.0, 400.0, 120.0, 5.0)
    st.caption("越低表示雾/霾/烟越多")
    stop_speed = st.slider("停止判定速度(px/s)", 1.0, 30.0, 8.0, 1.0)
    stop_sustain = st.slider("停止持续触发(s)", 0.5, 6.0, 2.0, 0.5)
    max_frames = st.number_input("最多处理帧数（0=全视频，建议500）", min_value=0, max_value=5000, value=0, step=50)

    st.subheader("禁飞区多边形（可选）")
    st.caption("示例：[(100,100),(300,120),(280,300),(120,280)]")
    poly_text = st.text_input("forbidden_poly", value="")
    forbidden_poly = None
    if poly_text.strip():
        try:
            forbidden_poly = eval(poly_text.strip(), {"__builtins__": {}})
        except Exception:
            st.warning("禁飞区格式解析失败：请用 [(x,y),...] 格式")

    st.subheader("预警路由")
    routing = {
        "bird_flock": st.text_input("鸟群风险 ->", "运行/飞行管控"),
        "intrusion": st.text_input("侵入敏感区域 ->", "安防/监管"),
        "visibility_low": st.text_input("低能见度 ->", "应急/运行调度"),
        "vehicle_stopped": st.text_input("异常停车 ->", "交管/运营"),
    }

    run_btn = st.button("开始分析")

if run_btn:
    if not uploaded:
        st.error("请先上传视频。")
        st.stop()

    in_path = os.path.join("outputs", "input.mp4")
    with open(in_path, "wb") as f:
        f.write(uploaded.getbuffer())

    out_dir = os.path.join("outputs", "run")
    os.makedirs(out_dir, exist_ok=True)

    st.info("正在分析：检测/跟踪/指标/事件预警/转码输出…")
    df, events, out_video, paths = process_video(
        video_path=in_path,
        out_dir=out_dir,
        conf=conf,
        max_frames=(max_frames if max_frames > 0 else None),
        forbidden_poly=forbidden_poly,
        bird_thresh=bird_thresh,
        visibility_low_thresh=visibility_low_thresh,
        stop_speed_pxs=stop_speed,
        stop_sustain_s=stop_sustain,
        routing=routing
    )

    df["congestion_index"] = congestion_index(df)
    df["congestion_level"] = df["congestion_index"].apply(level_ci)

    st.session_state["df"] = df
    st.session_state["events"] = [e.to_dict() for e in events]
    st.session_state["out_video"] = out_video
    st.session_state["paths"] = paths
    st.success("分析完成 ✅")

# 读取状态
df = st.session_state.get("df")
events = st.session_state.get("events", [])
out_video = st.session_state.get("out_video")
paths = st.session_state.get("paths", {})

tab1, tab2, tab3, tab4 = st.tabs(["交通态势", "预警中心", "天气态势", "农业/扩展接口"])

with tab1:
    st.subheader("标注后视频")
    if out_video and os.path.exists(out_video):
        st.video(out_video)
    else:
        st.write("请先上传视频并点击【开始分析】")

    if df is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("指标曲线")
            fig = plt.figure()
            plt.plot(df["time_s"], df["flow_vpm"], label="flow(vpm)")
            plt.plot(df["time_s"], df["occ_ratio"], label="occ_ratio")
            plt.plot(df["time_s"], df["speed_rel_pxs"], label="speed_rel(pxs)")
            plt.plot(df["time_s"], df["congestion_index"], label="congestion_index")
            plt.xlabel("time(s)")
            plt.legend()
            st.pyplot(fig)
        with c2:
            st.subheader("当前拥堵等级（末帧）")
            st.metric("拥堵指数", f"{df['congestion_index'].iloc[-1]:.1f}")
            st.write("等级：", df["congestion_level"].iloc[-1])
            st.caption("说明：指数综合流量、占有率、速度得出，用于态势量化信号。")

        st.subheader("导出数据")
        st.dataframe(df.tail(20), use_container_width=True)
        if paths:
            st.download_button("下载 metrics.csv", data=open(paths["metrics_csv"], "rb").read(),
                               file_name="metrics.csv")

with tab2:
    st.subheader("事件时间线")
    if not events:
        st.write("暂无事件")
    else:
        ev_df = pd.DataFrame(events)
        # 排序
        ev_df = ev_df.sort_values("time_s")
        st.dataframe(ev_df[["time_s","category","severity","confidence","target","message","evidence_path"]], use_container_width=True)

        st.subheader("事件详情查看")
        idx = st.number_input("输入事件序号（从0开始）", min_value=0, max_value=len(events)-1, value=0, step=1)
        ev = events[int(idx)]
        st.write(ev["message"])
        st.write(f"Time: {ev['time_s']:.2f}s  |  Category: {ev['category']}  |  Severity: {ev['severity']}  |  Target: {ev['target']}")
        if ev.get("evidence_path") and os.path.exists(ev["evidence_path"]):
            st.image(ev["evidence_path"], caption="Evidence (关键帧)")

        if paths:
            st.download_button("下载 events.json", data=open(paths["events_json"], "rb").read(),
                               file_name="events.json")

with tab3:
    st.subheader("天气态势")
    if df is None:
        st.write("请先运行分析")
    else:
        fig = plt.figure()
        plt.plot(df["time_s"], df["visibility_score"], label="visibility_score")
        plt.xlabel("time(s)")
        plt.legend()
        st.pyplot(fig)

        # 简单报告（可解释）
        vis_last = float(df["visibility_score"].iloc[-1])
        if vis_last < 80:
            level = "较差"
            advice = "建议低空运行谨慎/提高间隔/必要时延误"
        elif vis_last < 140:
            level = "一般"
            advice = "注意局部雾霾/烟尘影响，建议人工复核"
        else:
            level = "良好"
            advice = "能见度良好"
        st.markdown(f"""
**天气报告（MVP）**  
- 能见度等级：**{level}**（visibility_score={vis_last:.1f}）  
- 建议：{advice}  
> 注：这是基于视频清晰度/对比度的“态势估计”，可在未来接入风速/云底高/PM2.5/水位等传感器。
""")

with tab4:
    st.subheader("农业 / 扩展接口（WIP）")
    st.write("""
    请添加农业或其他插件
""")
