# IASR 空中态势识别集成系统

> **Integrated Air Situation Recognition System**  
> 面向低空经济场景的“视频感知 → 指标量化 → 事件预警 → 联动派单”原型系统。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

IASR 是一个面向低空经济、飞行汽车与城市低空运行场景的态势识别 MVP。它可以从视频中识别人、车辆、飞行器、鸟群等目标，计算交通与能见度等态势指标，并把复杂现场信息转化为可追踪、可导出、可联动的标准化事件。

本项目不追求替代专业监管或人工判断，而是尝试验证一个核心思路：

> **低空经济落地的关键不只是“能不能飞”，而是能不能安全、稳定、规模化地运营。**

## ✨ Features

### 1. 视频态势识别
- 支持上传本地视频：`mp4 / mov / mkv`
- 基于 YOLOv8 进行目标检测与跟踪
- 可识别并统计车辆、行人、鸟类、飞行器等目标
- 输出浏览器可播放的标注后视频

### 2. 交通指标与拥堵指数
系统会自动计算多类交通态势指标：

| 指标 | 含义 |
|---|---|
| 流量（辆/分钟） | 基于计数线统计车辆通过数量 |
| 占有率（%） | ROI 区域内目标框面积占比 |
| 相对速度（px/s） | 基于跟踪轨迹估计目标运动速度 |
| 拥堵指数（0-100） | 综合流量、占有率、速度得到的态势量化结果 |

### 3. 天气 / 能见度态势估计
- 基于视频清晰度与对比度估计 `visibility_score`
- 自动生成“良好 / 一般 / 较差”的能见度状态说明
- 可触发低能见度事件，用于模拟运行调度预警

> 说明：这里的能见度估计是 MVP 代理指标，不等同于专业气象预报。

### 4. 统一事件系统
系统将识别结果转化为标准化事件，当前支持：

| 事件类型 | 中文含义 | 默认通知对象 |
|---|---|---|
| `bird_flock` | 鸟群风险 | 运行 / 飞行管控 |
| `intrusion` | 敏感区域入侵 | 安防 / 监管 |
| `visibility_low` | 低能见度风险 | 应急 / 运行调度 |
| `vehicle_stopped` | 疑似异常停车 | 交管 / 运营 |
| `crop_disease_suspected` | 疑似作物病害 | 农业运维 / 巡检人员 |

每个事件包含：
- 事件 ID
- 事件类型
- 严重等级
- 置信度
- 通知对象
- 建议动作
- 证据关键帧
- 处置状态

### 5. 联动派单模拟
除事件表外，系统会自动生成模拟派单日志：

- `dispatch_log.csv`：原始派单日志
- `dispatch_log_zh.csv`：中文派单日志

用于模拟“事件触发 → 路由分发 → 人工复核 / 处置”的低空运行联动流程。

### 6. 农业扩展接口
系统新增农业图像分析入口，可上传作物或叶片图片，进行“疑似作物病害”检测。

当前版本采用启发式图像分析方法，重点验证插件化接口与事件联动流程：

- 识别叶片主体区域
- 标出疑似异常区域
- 输出疑似病害分数
- 生成 `crop_disease_suspected` 事件
- 接入统一事件表与派单日志

> 说明：农业模块目前是 MVP 版本，不等同于专业病害诊断模型。

### 7. 更友好的结果导出
每次运行都会生成独立输出目录，不会覆盖历史结果。系统支持导出：

- 中文指标表：`metrics_zh.csv`
- 原始指标表：`metrics.csv`
- 中文事件表：`events_zh.csv`
- 原始事件表：`events.json / events.csv`
- 中文派单日志：`dispatch_log_zh.csv`
- 标注后视频：`annotated_h264.mp4`
- 运行配置：`run_config.json`
- 证据关键帧：`evidence/`

## 🚀 Quick Start

### 1. 克隆项目

```bash
git clone https://github.com/Gunpowder-git/IASR-Integrated-air-situation-recognition-system.git
cd IASR-Integrated-air-situation-recognition-system
```

### 2. 创建虚拟环境

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. 准备 YOLO 权重

本项目默认使用 `yolov8n.pt`。

推荐将权重文件放在项目根目录或 `models/` 文件夹：

```text
IASR-Integrated-air-situation-recognition-system/
  app.py
  perception_core.py
  yolov8n.pt          # 可以放这里
  models/
    yolov8n.pt        # 或者放这里
```

如果没有本地权重，也可以在页面侧边栏勾选“允许自动下载模型”，但网络不稳定时不推荐。

### 5. 启动系统

```bash
python -m streamlit run app.py
```

启动后，终端会显示访问地址，一般是：

```text
http://localhost:8501
```

## 🧭 How to Use

### 视频分析流程

1. 打开 Streamlit 页面
2. 上传视频文件（建议先使用 10-30 秒短视频）
3. 在侧边栏选择场景预设：
   - 通用模式
   - 交通态势优先
   - 低能见度预警优先
   - 敏感区入侵演示
4. 根据需要调整参数：
   - 检测置信度
   - 最多处理帧数
   - 跳帧处理
   - 最大处理宽度
   - ROI 区域
   - 车辆计数线
   - 敏感区 / 禁飞区多边形
   - 事件阈值
   - 预警路由对象
5. 点击 **开始视频分析**
6. 在不同 Tab 中查看结果：
   - 交通态势
   - 预警中心
   - 天气态势
   - 输出文件

### 农业扩展流程

1. 进入 **农业扩展** Tab
2. 上传作物或叶片图片（`jpg / jpeg / png`）
3. 调整疑似异常区域阈值
4. 点击 **开始农业扩展分析**
5. 查看标注图片、疑似病害分数、事件表与派单日志

## ⚙️ 参数说明

### 运行模式

| 参数 | 建议说明 |
|---|---|
| 检测置信度 | 值越高，检测越严格；值越低，更容易触发目标 |
| 最多处理帧数 | 演示时建议 300-800，避免处理时间过长 |
| 跳帧处理 | 值越大速度越快，但结果更粗略 |
| 最大处理宽度 | 建议 960 或 1280，兼顾速度与效果 |

### 区域配置

| 参数 | 示例 |
|---|---|
| ROI 区域 | `(0,0,960,540)` |
| 车辆计数线 | `[(120,360),(640,360)]` |
| 敏感区多边形 | `[(100,100),(300,120),(280,300),(120,280)]` |

### 事件阈值

| 参数 | 作用 |
|---|---|
| 鸟群数量阈值 | 超过一定数量后触发鸟群风险 |
| 低能见度阈值 | 阈值越高，越容易触发低能见度事件 |
| 异常停车速度阈值 | 速度低于阈值并持续一定时间后触发异常停车 |
| 同类事件冷却时间 | 防止同类事件短时间内重复刷屏 |

## 📁 输出目录

系统会把每次运行结果保存到独立文件夹中：

```text
outputs/
  runs/
    video_20260512_153000/
      input.mp4
      annotated_raw.mp4
      annotated_h264.mp4
      metrics.csv
      metrics_zh.csv
      events.json
      events.csv
      events_zh.csv
      dispatch_log.csv
      dispatch_log_zh.csv
      run_config.json
      evidence/

    agriculture_20260512_154500/
      crop_input.jpg
      agriculture_metrics.json
      agriculture_metrics_zh.csv
      agriculture_events.json
      agriculture_events_zh.csv
      agriculture_dispatch_log_zh.csv
      agriculture_evidence/
```

## 🧩 Project Structure

```text
IASR-Integrated-air-situation-recognition-system/
  app.py                 # Streamlit 主界面
  perception_core.py     # 视频检测、跟踪、指标计算、事件触发
  event_engine.py        # 统一事件模型、中文事件表、派单日志
  agriculture_core.py    # 农业扩展接口：疑似作物病害检测
  requirements.txt       # Python 依赖
  models/                # 可选：模型权重目录
  outputs/               # 运行输出目录，建议加入 .gitignore
```

## 🧠 System Design

IASR 的核心流程可以概括为：

```text
视频 / 图像输入
      ↓
目标检测与跟踪
      ↓
指标计算：流量、占有率、速度、能见度
      ↓
事件生成：鸟群、入侵、低能见度、异常停车、疑似作物病害
      ↓
预警路由：运行、安防、应急、交管、农业运维
      ↓
结果导出：视频、CSV、JSON、证据截图、派单日志
```

项目更关注“低空运行数字底座”的表达，而不是单一算法性能。它尝试把复杂现场信息变成可解释、可追踪、可联动的事件信号。

## ❓ FAQ

### 1. 为什么提示找不到 `yolov8n.pt`？

请把 `yolov8n.pt` 放到项目根目录或 `models/` 文件夹。也可以在页面侧边栏勾选“允许自动下载模型”。

### 2. 为什么视频处理很慢？

可以尝试：

- 使用 10-30 秒短视频
- 设置“最多处理帧数”为 300-800
- 将“跳帧处理”设为 2 或 3
- 将“最大处理宽度”设为 960
- 使用轻量模型 `yolov8n.pt`

### 3. 为什么网页视频黑屏或无法播放？

系统会生成 `annotated_h264.mp4`，这是浏览器兼容性更好的版本。请优先查看或下载该文件。

### 4. 农业病害识别准确吗？

当前农业模块只是 MVP，用于展示“农业场景插件如何接入统一事件系统”。它不是专业病害诊断模型，结果需要人工复核。

### 5. 事件结果可以直接用于实际决策吗？

不建议。所有预警都是辅助提示，应由人工确认后再用于真实运行或处置。

## 🛣️ Roadmap

- [ ] 支持更多农业病害类别识别
- [ ] 增加烟雾 / 火灾 / 烟羽追踪事件
- [ ] 增加人群聚集、逆行、起降点占用等场景插件
- [ ] 支持多摄像头输入与区域热力图
- [ ] 对接真实平台 API，实现派单、确认、关闭的闭环流程
- [ ] 增加一键运行报告导出功能

## 🌐 English Summary

IASR (Integrated Air Situation Recognition System) is an MVP system for low-altitude situation awareness. It processes videos and agricultural images, extracts traffic and visibility indicators, generates standardized events, and simulates alert routing for different response targets.

Current features include YOLO-based video detection/tracking, traffic metrics, congestion index, visibility estimation, configurable alert routing, Chinese-labeled CSV/JSON exports, and an agricultural extension interface for suspected crop disease detection.

This project is an educational and research-oriented prototype. All alerts are advisory and should be manually verified before real-world use.

## ⚠️ Disclaimer

本项目为课程实践、科研展示与原型验证用途。系统输出的事件和预警不构成专业安全、交通、农业或气象判断，实际使用前必须进行人工复核和专业验证。

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- FFmpeg / imageio-ffmpeg
- Tongji University iS3 Lab
- Tongji University XAI Lab

## Author

**He Jiale**  
Tongji University  
GitHub: [@Gunpowder-git](https://github.com/Gunpowder-git)
