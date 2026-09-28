# VesselSegmentationDemo

<!--
双语 README：中文（上）与 English（下）。
-->

## 目录（Contents）
- 简介 / Overview
- 特性 / Features
- 依赖与安装 / Requirements & Installation
- 快速开始 / Quick Start
- 数据 / Data
- 模型与推理 / Model & Inference
- 训练（可选） / Training (optional)
- 网络结构说明 / Network Architecture
- 贡献 / Contributing
- 许可证 / License
- 联系 / Contact

---

## 简介（中文）
VesselSegmentationDemo 是一个用于血管（vessel）分割任务的演示仓库，包含示例代码、推理脚本与可视化工具，适合用于教学、快速原型和结果复现。该仓库采用轻量结构，示例数据和模型权重通常不直接托管于仓库中（请参见 data/ 和 models/ 目录说明）。

### 特性（中文）
- 示例推理脚本（infer.py），展示如何加载模型并对单张图像进行分割
- 可选的训练脚本示例（train.py），用于演示训练流程
- notebooks/ 中的 Jupyter 示例，便于交互式探索与可视化
- 结果保存与可视化工具，生成 mask、叠加图等

### 依赖与安装（中文）
推荐使用虚拟环境：

1. 创建并激活虚拟环境：

   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

2. 安装依赖：

   pip install -r requirements.txt

（若无 requirements.txt，请根据项目需要安装：numpy, opencv-python, torch/keras 等）

### 快速开始（中文）
1. 克隆仓库：

   git clone https://github.com/dachen36/VesselSegmentationDemo.git
   cd VesselSegmentationDemo

2. 运行示例推理：

   python infer.py --input examples/example_image.png --output results/mask.png

3. 查看 results/ 下的输出图像（mask 或叠加图）。

---

## Overview (English)
VesselSegmentationDemo is a demonstration repository for vessel segmentation tasks. It contains example code, inference scripts, and visualization utilities suitable for teaching, prototyping, and reproducing basic segmentation pipelines. Example datasets and model weights are usually not included in the repo—see the data/ and models/ folders for instructions.

### Features
- Example inference script (infer.py) that shows how to load a model and perform segmentation on a single image
- Optional training example (train.py) to illustrate training workflow
- Jupyter notebooks in notebooks/ for interactive exploration and visualization
- Utilities for saving and visualizing outputs (masks, overlays, etc.)

### Requirements & Installation
We recommend using a virtual environment:

1. Create and activate a virtual environment:

   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate

2. Install dependencies:

   pip install -r requirements.txt

(If requirements.txt isn't provided, install needed packages such as numpy, opencv-python, torch or tensorflow, etc.)

### Quick Start
1. Clone the repository:

   git clone https://github.com/dachen36/VesselSegmentationDemo.git
   cd VesselSegmentationDemo

2. Run an example inference:

   python infer.py --input examples/example_image.png --output results/mask.png

3. Check outputs under results/ (mask files or overlays).

---

## Data
- data/: 说明如何准备数据集、示例数据格式和必要的预处理步骤。
- 请勿将大数据集直接提交到仓库。推荐提供下载脚本或说明，以便用户从公开数据源获取样本。

## Model & Inference
- models/: 存放模型权重或说明如何下载第三方预训练权重。
- infer.py: 一个示例脚本，通常包含：输入读取 -> 预处理 -> 模型推理 -> 后处理 -> 保存结果。

示例调用：

   python infer.py --input path/to/image.png --output path/to/output_mask.png --model models/best.pth

## Training (optional)
- train.py: 训练流程示例，包含数据加载、损失函数、训练循环和模型保存。
- 如果你计划训练自己的模型，请在 README 中补充数据格式、超参数和训练脚本示例命令。

---

## 网络结构说明（中文）

本项目目前实现的是一个面向血管分割的轻量级 U-Net 基线模型，主要用于论文复现尝试和实验验证。代码中包含两个规模不同的版本：`main.py` 中的 `TinyUNet` 和 `main2.py` 中的 `UNetPro`。

### 整体结构

```text
输入灰度图像（1 × 256 × 256）
        │
        ▼
DoubleConv：提取浅层特征
        │
        ├──────────── 跳跃连接 ────────────┐
        ▼                                  │
MaxPool2d：下采样                         │
        │                                  │
DoubleConv：提取更深层特征                │
        │                                  │
双线性上采样：恢复空间分辨率               │
        │                                  │
        └──── 与浅层特征进行拼接 ──────────┘
                       │
                 DoubleConv
                       │
                 1×1 卷积输出
                       │
                       ▼
                  血管分割掩膜
```

### `TinyUNet`

`TinyUNet` 是轻量版本，结构为：

- 输入通道：1（灰度图像）
- 第一阶段：`1 → 32` 个特征通道
- 下采样后：`32 → 64` 个特征通道
- 双线性上采样恢复分辨率
- 与浅层的 32 通道特征进行跳跃连接
- 拼接后的 `96` 个通道经过卷积变为 `32` 个通道
- 最后通过 `1×1` 卷积输出 1 通道预测图

### `UNetPro`

`UNetPro` 是增强版本，整体结构与 `TinyUNet` 类似，但使用更多特征通道：

- 第一阶段：`1 → 64`
- 下采样后：`64 → 128`
- 上采样后与浅层的 64 通道特征拼接
- 拼接后的 `192` 个通道经过卷积变为 `64` 个通道
- 最后输出 1 通道血管分割预测图

因此，`UNetPro` 具有更强的特征提取能力，但需要更多显存和计算资源。

### DoubleConv 模块

每个 `DoubleConv` 模块包含两次卷积操作：

```text
3×3 卷积 → BatchNorm → ReLU
3×3 卷积 → BatchNorm → ReLU
```

其中：

- 3×3 卷积用于提取局部图像特征；
- BatchNorm 有助于稳定训练；
- ReLU 提供非线性表达能力。

### 跳跃连接（Skip Connection）

U-Net 的跳跃连接会把编码器中的浅层特征直接传给解码器。浅层特征包含较多的边缘和位置信息，这对于恢复细小血管、减少分割边界模糊非常重要。

### 数据预处理流程

```text
读取灰度图像
    → 调整为 256×256
    → 使用 CLAHE 增强局部对比度
    → 归一化到 [0, 1]
    → 输入网络
    → Sigmoid 得到血管概率图
    → 使用 0.5 阈值二值化
    → 输出分割掩膜
```

### 损失函数

项目使用 BCE Loss 和 Dice Loss 的混合损失：

```text
MixedLoss = α × BCE Loss + (1 - α) × Dice Loss
```

- BCE Loss：进行像素级前景/背景分类；
- Dice Loss：衡量预测血管区域与真实标注区域的重叠程度；
- 混合损失更适合前景像素较少的血管分割任务。

### 训练流程

训练过程包括：

1. 从 `images/` 读取原始图像；
2. 从 `masks/` 读取对应的血管标注；
3. 完成尺寸调整、CLAHE 增强和归一化；
4. 使用 U-Net 前向推理；
5. 计算 BCE + Dice 混���损失；
6. 反向传播并更新网络参数；
7. 根据损失变化动态调整学习率；
8. 定期生成原图、真实标注、预测结果和叠加图；
9. 保存训练后的模型权重。

### 复现说明

本项目是对血管分割任务的轻量化复现和实验性实现。当前代码实现的是简化版 U-Net 基线模型，并不一定完全等同于目标论文中的原始网络结构。若要严格复现论文结果，还需要根据论文补充数据集划分、数据增强、训练超参数、评价指标以及论文中的专用模块。

---

## Network Architecture (English)

This project currently implements a lightweight U-Net baseline for vessel segmentation. It is intended for paper-reproduction experiments and practical demonstrations. Two model variants are provided: `TinyUNet` in `main.py` and `UNetPro` in `main2.py`.

### Overall architecture

The network follows an encoder-decoder design:

```text
Grayscale input (1 × 256 × 256)
        │
        ▼
DoubleConv: shallow feature extraction
        │
        ├──────────── skip connection ────────────┐
        ▼                                          │
MaxPool2d: downsampling                            │
        │                                          │
DoubleConv: deeper feature extraction              │
        │                                          │
Bilinear upsampling: restore resolution             │
        │                                          │
        └──── concatenate shallow features ────────┘
                       │
                 DoubleConv
                       │
                 1×1 output convolution
                       │
                       ▼
                Vessel segmentation mask
```

### `TinyUNet`

`TinyUNet` is the lightweight variant:

- Input channels: 1 grayscale channel
- First stage: `1 → 32` feature channels
- Downsampled stage: `32 → 64`
- Bilinear upsampling restores the spatial resolution
- The upsampled features are concatenated with the 32-channel shallow features
- The concatenated 96 channels are processed into 32 channels
- A final `1×1` convolution produces a one-channel prediction map

### `UNetPro`

`UNetPro` follows the same basic design but uses more feature channels:

- First stage: `1 → 64`
- Downsampled stage: `64 → 128`
- The upsampled features are concatenated with 64-channel shallow features
- The concatenated 192 channels are processed into 64 channels
- A final one-channel output represents the vessel segmentation prediction

`UNetPro` has greater feature capacity but requires more memory and computation than `TinyUNet`.

### DoubleConv block

Each `DoubleConv` block consists of:

```text
3×3 convolution → BatchNorm → ReLU
3×3 convolution → BatchNorm → ReLU
```

The convolutions extract local features, BatchNorm helps stabilize training, and ReLU provides nonlinear representation capacity.

### Skip connections

Skip connections transfer shallow encoder features directly to the decoder. These features preserve edge and location information, which helps recover thin vessels and produce sharper segmentation boundaries.

### Preprocessing pipeline

```text
Read grayscale image
    → resize to 256×256
    → enhance local contrast with CLAHE
    → normalize to [0, 1]
    → feed into the network
    → apply Sigmoid for vessel probabilities
    → threshold at 0.5
    → output binary vessel mask
```

### Loss function

The project uses a mixed BCE and Dice loss:

```text
MixedLoss = α × BCE Loss + (1 - α) × Dice Loss
```

BCE handles pixel-level foreground/background classification, while Dice loss measures the overlap between the predicted vessel region and the ground-truth mask. The combination is useful for vessel segmentation, where vessel pixels are usually much fewer than background pixels.

### Training pipeline

The training process includes loading images and masks, resizing and preprocessing them, running U-Net inference, calculating the mixed BCE-Dice loss, updating the model through backpropagation, adjusting the learning rate, periodically generating visualization results, and saving the trained weights.

### Reproduction note

This repository is a lightweight and experimental reproduction for vessel segmentation. The current implementation is a simplified U-Net baseline and may not be identical to the original network in the target paper. Strict reproduction would require matching the paper's dataset split, augmentation strategy, hyperparameters, evaluation metrics, and any paper-specific modules.

---

## Contributing
欢迎贡献：请通过 Fork → 新建分支 → 提交 → 发起 Pull Request 的流程贡献代码。提交 Issue 报告 bug 或提出功能请求。

建议在贡献前先打开 Issue 进行讨论，特别是对数据集或模型格式做出重大更改时。

## License
除非另有说明，否则本仓库代码采用 MIT 许可证或仓库根目录的 LICENSE 文件所示许可。请在使用或分发代码时遵守许可条款。

## Contact
如有问题或建议，请在仓库中创建 Issue，或联系仓库维护者（GitHub: @dachen36）。

---

谢谢使用 VesselSegmentationDemo！
