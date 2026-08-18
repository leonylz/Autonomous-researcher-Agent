# T6 — 创新方法验证:论文驱动的 CIFAR-10 突破(离线论文库)

## Goal
给定一个 **baseline**(workspace/train.py:2 卷积 CNN,CIFAR-10 test_acc ≈ 72-75%),
**必须通过论文调研引入至少一个创新方法**,把测试准确率提升到 **≥ 83%**。

## 创新要求(本任务的核心考点)
1. **第一轮必须先做论文调研**:调用 idea_agent 阅读论文库
   `literature/`(**共 15 篇** CIFAR-10 相关论文,全文本地已存:
   mixup 1710.09412 / CutMix 1905.04899 / SGDR 1608.03983 /
   Cutout 1708.04552 / AutoAugment 1805.09501 / Label Smoothing 1512.00567 /
   AdamW 1711.05101 / WideResNet 1605.07146 / Random Erasing 1708.04896 /
   Cyclical LR 1506.01186 / Snapshot Ensembles 1704.00109 /
   RandAugment 1909.13719 / ResNet 1512.03385 / DropBlock 1810.12890 /
   Stochastic Depth 1603.09382)。
   **先 `list_files literature/` 查看全部论文,再逐篇 `read_file` 阅读**,
   提取 1-2 个可落地方法,把发现写入 `IDEA_NOTES.md`。
   **禁止跳过调研直接调参**——普通微调(加层/dropout/lr)到不了 83%。
   (本地库优先;arXiv 联网搜索可达,可补充本地库没有的最新方法;
   数据集下载仍被禁止——只搜论文,不下载数据。)
2. **必须真的用上论文方法**:在 baseline 上实现你选择的创新方法
   (如 mixup/CutMix 数据混合、SGDR 重启调度等),单变量验证。
3. **最终报告必须引用所用论文**([arXiv:xxx] 形式),说明方法来源。

## 修改基线(硬规则,不是复用也不是重写)
- 基线 `workspace/train.py` 是**起点**,任务成果 = 在它之上**增量修改**:
  保持其整体结构与契约(log_metrics/checkpoint/dry-run/数据加载),
  在对应位置**添加**论文方法的实现(如 mixup 加在数据加载/训练循环)。
- **禁止**把基线原样运行当作成果(那只是起点,不是创新);
  **禁止**完全重写为另一个脚本(无法与基线单变量对比)。
- 验证协议:先运行基线确认起点(~72-75%),再跑「基线+创新方法」,
  对比必须能归因于你添加的方法(单变量原则)。

## 数据(本地,禁止下载)
- `workspace/CIFAR-10/`:10 个类别子目录 × 6000 张 PNG(共 60000 张),
  torchvision `ImageFolder` 可直接加载(root="CIFAR-10")。
- **自行划分训练/测试**(建议 5:1 = 5000/1000 每类),固定 seed 保证可复现。

## What to Try
1. 论文调研 → 选方法(mixup 或 CutMix 是 CIFAR-10 上最成熟的正则方法)
2. 实现并验证:先跑 baseline 确认起点,再叠加论文方法单变量对比
3. 若不足:方法组合(mixup + SGDR 调度)或加宽 baseline 架构
4. 保持账本可信:每个实验记录假设 + 指标,方法来源注明论文

## Constraints
- PyTorch, **CPU only**, max **30 epochs per run**(一个 run ≈ 20-45 min)。
- 必须用框架模板契约(dry-run 门 / METRIC_JSON / checkpoints)。
- **创新方法必须真实实现并生效**(loss/指标可验证),不能只写进报告。

## Success Criteria
test_acc ≥ **0.83**,且满足:① idea_agent 的 IDEA_NOTES.md 存在;
② 至少一个实验使用了论文方法;③ 报告/结论含论文引用。
