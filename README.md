# Mini 4 Pro Vision Autolanding Android

基于 **DJI Mobile SDK V5**、**OpenCV** 与 **YOLOv8** 的无人机视觉自主起降与巡航 Android 项目。  
本项目面向本科毕业设计场景，目标是在 **GPS 拒止环境** 下，使用 **Android 手机作为边缘计算上位机**，通过 **DJI Mini 4 Pro + RC-N2 遥控器** 实现基于视觉的目标检测、追踪、对准与降落。

## 上游项目说明

本项目不是从零开始创建的 Android 工程，而是基于 **DJI 官方开源项目** 的 Sample 工程进行二次开发。

上游项目为：

- **DJI Mobile-SDK-Android-V5**
- 本项目主要基于其中的 **`SampleCode-V5`** 目录进行修改与扩展

本仓库保留了上游工程所需的多模块结构，并在其基础上新增了视觉检测、仿真测试、实机联调与控制逻辑。

## 项目目标

核心目标：

- 在 GPS 拒止环境下实现无人机对特定视觉标识的自主识别与追踪
- 在 Android 端完成目标检测、ROI 跟踪、坐标还原与控制指令生成
- 通过 DJI Mobile SDK V5 的 Virtual Stick 能力实现飞行控制闭环
- 支持仿真视频验证与真实无人机静态联调，为后续户外实飞做准备

## 硬件环境

- DJI Mini 4 Pro
- RC-N2 遥控器
- Android 手机（USB 连接遥控器）

## 软件技术栈

- Android Studio
- Kotlin / Java
- DJI Mobile SDK V5
- OpenCV Android 4.x
- YOLOv8n（导出为 ONNX）
- Virtual Stick 控制模式

## 核心能力

### 1. 720p 轻量化推理
虽然图传源分辨率更高，但处理链路中会优先缩放至 `1280x720`，以降低移动端推理负担并提升实时性。

### 2. 跳帧检测
采用跳帧推理策略，例如每 3 帧推理 1 次，以在检测精度与运行帧率之间取得平衡。

### 3. 双模鹰眼搜索
- **全图模式**：适合低空或目标较大时的全局搜索
- **鹰眼模式**：当全图未检测到目标时，对中心区域进行裁剪推理，用于高空小目标捕获

### 4. 动态 ROI 追踪
当目标被成功锁定后，后续帧优先在上一帧目标附近进行局部检测，以提升追踪稳定性和检测效率。

### 5. 坐标还原
根据当前推理模式（全图缩放或 ROI 裁剪），将检测框坐标映射回统一的 `1280x720` 坐标系，再归一化送入控制逻辑。

### 6. 状态机控制
项目在实机联调阶段引入飞行状态机，用于管理：

- `SEARCHING`
- `ALIGNING`
- `DESCENDING`

以实现搜索、对准、下降等阶段的逻辑切换。

## 当前工程结构

本仓库保留了 DJI 上游 Sample 工程所需的多模块结构。

```text
mini4pro-vision-autolanding-android/
├── android-sdk-v5-as/          # Android Studio 工程入口
├── android-sdk-v5-sample/      # 主 Sample 模块，包含本项目主要业务代码
├── android-sdk-v5-uxsdk/       # DJI UXSDK 模块
└── opencv/                     # OpenCV 模块
