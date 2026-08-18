# 合成数据分类任务:模型拟合高斯簇分类

## Goal
训练一个 MLP 在合成数据集上达到测试准确率 >= 0.90。

## Codebase
- Training script: train.py(基于 core/train_template.py)
- Checkpoints: ./checkpoints/(best_model.pth kept)

## What to Try
- if test_acc < 0.85: 增大模型容量或加隐藏层
- if 0.85 <= test_acc < 0.90: 调整学习率/增加 epochs
- if test_acc >= 0.90: 停止,记录结果

## Constraints
- PyTorch, CPU, max 10 epochs
- **数据必须合成生成(torch 随机生成 2D 高斯簇,禁止下载任何外部数据)**
  · 训练/测试各 2000 样本,4 个高斯簇,簇中心间距适中
  · 用 torch.Generator 固定 seed 保证可复现

## Current Status
- 尚无实验。从 baseline 开始。
