# T3 — Fix a diverging training configuration

## Goal
The training run **diverges** (loss explodes to NaN). Diagnose why and fix it so
training converges and reaches **test accuracy ≥ 70%** on CIFAR-10 within budget.

## Codebase
- Training script: `train.py` (the agent creates it from `core/train_template.py`)
- The divergence is a configuration bug the agent introduces deliberately in
  cycle 1 (e.g. lr=10 with no schedule), observes the explosion in the log, and
  must then diagnose and fix — the classic "model diverges after N epochs" case.

## What to Investigate
1. Check the loss curve in the log: does it explode immediately or after some epochs?
2. Gradient sanity: add gradient clipping (max_norm=1.0) as a safety net.
3. Try smaller LR (current is deliberately broken) with warmup + cosine.
4. Check data: any NaN/Inf in the batch? (unlikely on CIFAR-10 — don't waste cycles here)
5. Log gradient norms every 100 steps to confirm the fix.

## Constraints
- PyTorch, GPU 0, max 30 epochs per run.
- MUST use the framework training template and dry-run gate.
- The fix must be verified by a real run (loss decreasing and no NaN), not just asserted.

## Success Criteria
Training converges (no NaN, loss decreasing) AND test accuracy ≥ 70%, recorded as
`final_metrics.test_acc` with `final_metrics.status` truthful.
