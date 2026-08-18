# T1 — MNIST CNN from scratch to 99.5%+

## Goal
Train a CNN on MNIST and reach **test accuracy ≥ 99.5%** on the held-out test set.
(99.5% is the bar: a naive 2-conv baseline stalls around 99.0–99.2%; closing the
last 0.3% requires data augmentation, dropout, and/or a stronger architecture.)

## Codebase
- Training script: `train.py` — **workspace 已预置模板副本**(train.py 与
  train_template.py 内容相同),直接 read_file + write_file 编辑其中三个
  TODO 区域即可,不要从零写、也不要用 shell 复制(cp 在 Windows 不存在)。
- Data: **local cache, no download** — torchvision `root="data"` finds MNIST at
  `workspace/data/MNIST` (all raw files pre-extracted). Never try to download.
- Checkpoints: `./checkpoints/` (`best_model.pth` kept)

## What to Try
1. Baseline: simple 2-conv CNN (32/64 channels) + ReLU + maxpool → expect ~99.0–99.2%.
2. If below 99.7%: **data augmentation is the highest-leverage move** (random
   affine/translation, and/or slight rotation). Then add dropout (0.25–0.5) and a
   third conv block or wider channels.
3. LR schedule: Adam + cosine (or StepLR) beats plain Adam for the last 0.3%.
4. Batch norm helps stabilize deeper variants.
5. Keep the change surface small: one variable at a time, so the ledger stays truthful.
6. **Fallback**: if the branches above are exhausted or gains are diminishing
   (<0.3pp per cycle for 2+ cycles): research innovative methods (papers/new
   ideas) via the idea agent FIRST — do not blindly keep tuning.

## Constraints
- PyTorch, **CPU only**, max **15 epochs per run** (a run ≈ 10–15 min on CPU — plan cycles accordingly).
- MUST use the framework training template (`cp core/train_template.py train.py`, only modify the three marked TODOs).
- MUST dry-run before any real training (writes `dry_run_log.json`), otherwise `launch_experiment` refuses.
- Checkpoint behavior is driven by the template contract (`save_every_n_epochs` / `best_model.pth` / `log_metrics`).

## Success Criteria
Test accuracy ≥ **0.995** on the held-out MNIST test set, recorded as
`final_metrics.test_acc` in the ledger — truthful, from a real run.
