# T5 — Crash detection and recovery

## Goal
A training run is **killed mid-way** (simulated: the agent launches training,
the evaluator sends SIGKILL to the process after epoch 5). The agent must
detect the crash, diagnose it from logs, and **recover by resuming from the
last checkpoint** to reach **test accuracy ≥ 70%** on CIFAR-10.

## Codebase
- Training script: `train.py` (agent creates it from `core/train_template.py`)
- The template's checkpoint logic (`save_every_n_epochs` / `best_model.pth`)
  is REQUIRED — resuming depends on it.

## What to Do
1. Launch the baseline; after the crash, the monitor reports the process died
   (truthful outcome: `failed`, not `completed`).
2. Read the log tail: determine how far training got and which checkpoint exists.
3. Resume from `best_model.pth` (or the last `checkpoint_epoch_N.pth`) — never
   restart from scratch unless no checkpoint exists.
4. Verify the resumed run reaches the target.

## Constraints
- PyTorch, GPU 0, max 40 epochs per run (across resume attempts).
- MUST use the framework training template and dry-run gate.
- A crash must be recorded truthfully (status `failed` in the ledger), never as completed.

## Success Criteria
After at least one crash-and-resume cycle, test accuracy ≥ 70%, recorded as
`final_metrics.test_acc`; ledger shows the crashed run with truthful status.
