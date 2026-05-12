# IASR System：空中态势识别集成系统
**面向低空经济的“感知—指标—事件—预警路由”一体化原型系统**

IASR System是一个可扩展的低空态势识别与预警集成系统。系统以**视频**为输入，进行**目标检测+跟踪**，计算**交通态势指标**（流量/占有率/速度）与**拥堵指数**，同时给出基于视频清晰度的**能见度态势估计**（雾/霾/烟的代理指标），最终输出统一格式的**事件（Event）**，并根据配置将事件路由给不同对象（运行、安防、应急、交管等）。

> ⚠️ 说明：本项目为课程/科研演示性质的 MVP，用于“态势感知与联动流程”展示。  
> 所有预警均为**辅助提示**，应由人工复核后再用于实际运行决策。

---

## 功能概览

### 感知与指标
- 基于 YOLO 的视频目标检测与跟踪
- 交通态势指标：
  - **流量 Flow（veh/min）**：通过“过线计数”统计
  - **占有率 Occupancy**：ROI 内车辆框面积/ROI 面积
  - **相对速度 Speed（px/s）**：基于跟踪轨迹的像素位移
  - **拥堵指数 Congestion Index（0–100）**：综合流量↑、占有率↑、速度↓得到可量化信号

### 天气/能见度态势（MVP 代理指标）
- 基于视频清晰度/对比度计算 **visibility_score**
- 自动生成简要“天气态势报告”（良好/一般/较差）

### 统一事件系统 + 预警路由
系统把识别结果转成统一事件模型（Event Schema），并按事件类型分发给对应对象：
- `bird_flock` → 运行/飞行管控
- `intrusion` → 安防/监管（提示需人工复核）
- `visibility_low` → 应急/运行调度
- `vehicle_stopped` → 交管/运营

事件支持：
- 时间、严重等级、置信度
- 文字说明（可解释）
- **证据关键帧截图**
- 导出 JSON

### Web 仪表盘（Streamlit）
- 多 Tab：交通态势 / 预警中心 / 天气态势 / 扩展接口
- 标注后视频展示（输出 H.264，浏览器兼容）
- 导出：`metrics.csv`、`events.json`

---

## 项目结构

```
IASR_system/
  app.py               # Streamlit 仪表盘
  perception_core.py   # 检测/跟踪 + 指标 + 事件触发 + 视频输出
  event_engine.py      # 统一事件模型 + 冷却/防抖工具
  requirements.txt
  outputs/             # 运行输出
```

---

## 环境要求
- 推荐 Python 3.9+
- Windows / macOS / Linux 均可运行
- GPU 可选（CPU 也能跑，但速度较慢）

---

## 安装与运行

### 1）创建并激活虚拟环境（推荐）

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### 2）模型权重
本项目默认使用 Ultralytics YOLO 权重（例如 `yolov8n.pt`）。

#### 推荐方案：本地放置权重
将 `yolov8n.pt` 放到项目根目录（与 `app.py` 同级）：

```
IASR_system/yolov8n.pt
```

### 3）启动仪表盘

```powershell
python -m streamlit run app.py
```

终端会显示访问地址（一般是 `http://localhost:8501`），用浏览器打开即可。

---

## 使用说明（快速上手）
1. 上传本地视频（mp4/mov/mkv）
2. 在侧边栏调整参数：
   - `conf`：检测置信度
   - 鸟群阈值、能见度阈值、异常停车阈值等
3. （可选）配置禁飞区多边形 `forbidden_poly`：
   - 示例：
     ```
     [(100,100),(300,120),(280,300),(120,280)]
     ```
4. 点击 **开始分析**
5. 查看结果：
   - **交通态势**：标注视频 + 指标曲线 + 拥堵等级
   - **预警中心**：事件列表 + 证据截图
   - **天气态势**：能见度曲线 + 天气报告

---

## 输出文件说明
每次运行会在 `outputs/run/` 下生成：

- `annotated_raw.mp4`：OpenCV 直接写出（浏览器可能不兼容）
- `annotated_h264.mp4`：H.264 转码后输出（浏览器可播放）✅
- `metrics.csv`：指标时间序列（交通/鸟/能见度等）
- `events.json`：统一事件输出
- `evidence/`：事件证据关键帧截图

---

## 事件定义

### `bird_flock`（鸟群风险）
当鸟类数量超过阈值并持续一段时间触发。

### `intrusion`（入侵/敏感区目标）
当空中目标进入禁飞/敏感区域触发。  
> 提示用途：辅助预警，需人工复核。

### `visibility_low`（能见度下降）
当 `visibility_score` 低于阈值并持续一段时间触发。

### `vehicle_stopped`（疑似异常停车）
当某车辆轨迹速度长期低于阈值触发（用于交通异常/事故风险提示）。

---

## 常见问题

### 1）标注后视频在网页端显示 0:00 / 黑屏
请使用 `annotated_h264.mp4`（本项目默认会生成并展示 H.264 输出）。

### 2）`yolov8n.pt` 自动下载失败（SSL/校园网限制）
使用“本地放置权重”方案，将 `yolov8n.pt` 放到项目根目录。

### 3）运行速度慢
- 用短视频（10–20 秒）进行演示
- 在 UI 中设置“最多处理帧数”（例如 300–800）
- 选择轻量模型 `yolov8n.pt`

---

## 可扩展方向
- 多路摄像头融合与区域热力图
- 更多事件插件：人群聚集、逆行、烟羽追踪、起降点态势评估等
- 对接学校平台 API：工单派发/消息推送/联动处置
- 农业/基础设施场景的专用模型插件化接入

---

## 许可证
- 本项目采用 MIT 许可证。 This project is licensed under the MIT.

---

## 致谢（Acknowledgements）
- Ultralytics YOLO
- Streamlit
- OpenCV / FFmpeg
- iS3 Lab, Tongji University
- XAI Lab, Tongji University

Author : He Jiale @Tongji University
