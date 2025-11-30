# 基于AI的大米质量检测系统

## 项目简介
利用计算机视觉和深度学习技术，实现对大米图像的自动分割、计数和形态特征分析。

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 模型安装：通过 https://huggingface.co/mouseland/cellpose-sam/tree/main 这一链接下载cpsam模型到对应的models文件夹中
3. 运行程序：`python rice_processing.py`

## 项目结构
- `rice_processing.py` - 主处理脚本
- `samples/` - 示例数据
  - `original/` - 原始图像
  - `results/` - 分割结果
  - `csv/` - 特征数据

## 技术栈

Python, Cellpose, SAM, OpenCV, scikit-image

