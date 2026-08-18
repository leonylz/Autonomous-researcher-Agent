# 在 Fashion-MNIST 上训练一个小型 CNN 达到测试准确率 92% 以上,CPU 可复现

## Goal
在 Fashion-MNIST 上训练一个小型 CNN 达到测试准确率 92% 以上,CPU 可复现

## Codebase
- Training script: `train.py` (the agent creates it from `core/train_template.py`)
- Checkpoints: `./checkpoints/` (`best_model.pth` kept)

## What to Try
- if test_acc < 0.90: 增加卷积层数或通道数(如从2层增至3层,通道数64->128)
- if 0.90 <= test_acc < 0.92: 增加训练轮数至15或调整学习率(如从0.001降至0.0005)
- if test_acc >= 0.92: 尝试减少模型参数或使用数据增强以提升泛化
- if 训练不收敛(loss不下降): 降低学习率或使用Adam优化器
- if 过拟合(train_acc高但test_acc低): 添加Dropout或权重衰减

## Constraints
PyTorch, CPU, 可复现(固定随机种子), 小型CNN(参数量<1M), 训练时间<30分钟

## Current Status
- 尚无实验。从 baseline 开始。
