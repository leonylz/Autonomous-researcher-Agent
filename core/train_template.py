"""
AutoResearcher standard training script template.

AGENT USAGE INSTRUCTIONS (do not delete the sections below):
- The workspace is pre-seeded with this template (train.py and train_template.py
  have identical content); edit train.py directly with read_file + write_file.
  (On Linux/Mac you may also `cp core/train_template.py train.py`; Windows has
  no cp — copying must be done with read_file + write_file.)
- Only modify the three TODO regions: 「model definition」「data loading」
  「batch processing inside the training loop」
- Do NOT delete the checkpoint saving / dry-run logic — otherwise
  launch_experiment will refuse to start
- Checkpoint behavior is driven by the checkpoint section of config.yaml:
    checkpoint:
      save_every_n_epochs: 5   # save a checkpoint every N epochs
      keep_best: true          # always keep the best best_model.pth
      dir: "checkpoints"       # weights directory (relative to workspace)
"""

import argparse
import json
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────
# 0. 契约化指标输出(硬约束,agent 不得删除本函数):
#    每轮训练必须调用 log_metrics() 输出 METRIC_JSON 契约行,
#    monitor 优先解析该行(字段名原样,如 test_acc),正则猜格式只是 fallback。
#    缺本函数 → launch_experiment 的模板硬校验拒绝启动。
# ─────────────────────────────────────────────────────────────────────


def log_metrics(metrics: dict) -> None:
    """契约化指标输出:打印 METRIC_JSON 行,字段名原样保留。

    示例:log_metrics({"epoch": 1, "loss": 0.013, "test_acc": 0.992})
    → METRIC_JSON {"epoch": 1, "loss": 0.013, "test_acc": 0.992}
    """
    print("METRIC_JSON " + json.dumps(metrics, ensure_ascii=False), flush=True)


# ─────────────────────────────────────────────────────────────────────
# 1. 读 config.yaml 的 checkpoint 配置（硬约束来源，agent 不得改读法）
# ─────────────────────────────────────────────────────────────────────

CONFIG_NAME = "config.yaml"


def _load_checkpoint_cfg(project_root=None):
    """读项目 config.yaml 的 checkpoint 段。缺省值：每 5 epoch / 保留最优 / ./checkpoints。"""
    import yaml
    cfg = {}
    # 向上找项目根（脚本通常从 workspace 运行，config 在上一级）
    candidates = []
    if project_root:
        candidates.append(project_root)
    cwd = os.getcwd()
    candidates += [cwd, os.path.dirname(cwd), os.path.dirname(os.path.dirname(cwd))]
    for base in candidates:
        p = os.path.join(base, CONFIG_NAME)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                cfg = (data.get("checkpoint") or {}) or {}
                break
            except Exception:
                continue
    return {
        "save_every_n_epochs": int(cfg.get("save_every_n_epochs", 5)),
        "keep_best": bool(cfg.get("keep_best", True)),
        "dir": str(cfg.get("dir", "checkpoints")),
    }


CHECKPOINT_CFG = _load_checkpoint_cfg()
CHECKPOINT_DIR = CHECKPOINT_CFG["dir"]
SAVE_EVERY_N_EPOCHS = max(1, CHECKPOINT_CFG["save_every_n_epochs"])
KEEP_BEST = CHECKPOINT_CFG["keep_best"]


# ─────────────────────────────────────────────────────────────────────
# 2. 模型定义（TODO: agent 在此填模型结构）
# ─────────────────────────────────────────────────────────────────────

# TODO: agent fills —— 定义你的模型，例如：
#   class SimpleModel(nn.Module):
#       def __init__(self): ...
#       def forward(self, x): ...
# 默认给一个占位（导入 torch 失败时也能 dry-run 检查结构）


def build_model():
    """TODO: agent fills —— 返回模型实例。"""
    import torch.nn as nn
    import torch.nn.functional as F

    class _Placeholder(nn.Module):
        def forward(self, x):
            return x

    return _Placeholder()


# ─────────────────────────────────────────────────────────────────────
# 3. checkpoint 保存 / 恢复（硬约束逻辑，agent 不得删除）
# ─────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, filename: str, ckpt_dir: str = CHECKPOINT_DIR):
    """保存 checkpoint。filename 为相对 ckpt_dir 的文件名。

    ckpt_dir 参数化，避免模块级 CHECKPOINT_DIR 被 main() 局部遮蔽导致
    权重和 training_log.json 写不同目录（BUG 8 修复）。
    """
    import torch
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    print(f"  [Checkpoint] 已保存: {path}")


def load_checkpoint(model, optimizer=None, checkpoint_path=None, device="cpu"):
    """加载 checkpoint 用于 resume 续训。返回 (start_epoch, best_acc)。"""
    import torch
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"  [Resume] 未找到 checkpoint: {checkpoint_path}")
        return 0, 0.0
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    best_acc = float(ckpt.get("best_acc", 0.0))
    print(f"  [Resume] 从 epoch {start_epoch} 续训, best_acc={best_acc:.4f}")
    return start_epoch, best_acc


# ─────────────────────────────────────────────────────────────────────
# 4. 主训练流程
# ─────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size, dry_run=False):
    """TODO: agent fills —— 返回 (train_loader, test_loader)。dry_run 时用小数据。"""
    raise NotImplementedError("agent 必须实现 get_dataloaders()")


def evaluate(model, test_loader, device):
    """TODO: agent fills —— 返回测试准确率（0~1）。"""
    raise NotImplementedError("agent 必须实现 evaluate()")


def train_one_epoch(model, loader, optimizer, device, epoch, max_steps=None):
    """TODO: agent fills —— 训练一个 epoch。返回平均 loss。"""
    raise NotImplementedError("agent 必须实现 train_one_epoch()")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--resume", type=str, default="",
                        help="续训用权重路径，如 checkpoints/checkpoint_epoch_5.pth 或 best_model.pth")
    parser.add_argument("--dry-run", action="store_true",
                        help="dry-run：只跑 2 步验证脚本无误，成功写 dry_run_log.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # 用局部变量 ckpt_dir 承载实际目录，传给 save_checkpoint()（参数化），
    # 避免 global 遮蔽模块级 CHECKPOINT_DIR 导致权重与 log 目录分离。
    ckpt_dir = args.checkpoint_dir or CHECKPOINT_DIR

    device = args.device
    os.makedirs(ckpt_dir, exist_ok=True)

    model = build_model().to(device)

    # dry-run：不训练，只验证数据/前向/后向 2 步
    if args.dry_run:
        print("=== DRY RUN（验证脚本无误，不训练）===")
        train_loader, _ = get_dataloaders(args.batch_size, dry_run=True)
        import torch
        import torch.nn as nn
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        step = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            if out.shape != y.shape and out.dim() > 1 and y.dim() == 1:
                loss = criterion(out, y)
            else:
                loss = torch.tensor(0.0, device=device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step >= 2:
                break
        with open("dry_run_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "dry_run": True,
                "steps": step,
                "time": time.time(),
                # 环境一致性事实源：launch_experiment 会用这两个字段校验
                # 训练命令的解释器/设备与干跑一致，不一致直接拒绝。
                "interpreter": sys.executable,
                "device": device,
            }, f)
        print("DRY RUN PASSED")
        return

    import torch
    import torch.nn as nn
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader, test_loader = get_dataloaders(args.batch_size)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            model, optimizer, args.resume, device=device)

    log_data = {"config": vars(args), "history": [], "best_acc": best_acc}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        test_acc = evaluate(model, test_loader, device)
        log_data["history"].append({"epoch": epoch + 1, "loss": train_loss, "acc": test_acc})
        # 契约化指标输出(必须保留:monitor 靠 METRIC_JSON 行提取指标进账本)
        log_metrics({"epoch": epoch + 1, "loss": round(float(train_loss), 6),
                     "test_acc": round(float(test_acc), 6)})
        print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} | test_acc={test_acc:.4f} | {time.time()-t0:.1f}s")

        # 硬约束：最优模型保存（keep_best）
        if KEEP_BEST and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, "best_model.pth", ckpt_dir)

        # 硬约束：每 save_every_n_epochs 存一个 checkpoint
        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, f"checkpoint_epoch_{epoch+1}.pth", ckpt_dir)

    log_data["final_test_acc"] = best_acc
    with open(os.path.join(ckpt_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"Training finished. best_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
