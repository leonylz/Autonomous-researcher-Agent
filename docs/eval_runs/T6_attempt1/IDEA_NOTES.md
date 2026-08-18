# IDEA_NOTES — 2025-06-01

Task context: 2-conv CNN baseline (64/128 conv + 2 FC) on CIFAR-10, CPU-only,
max 30 epochs, Adam lr=1e-3, batch=128. Baseline ~72-75% test acc.
Goal: cheap, effective regularization/scheduling methods for a small model.

> Note: the `literature/*.md` files were not physically readable in this
> environment (read_file returned "file not found" for all 8 paths). The
> methods below are extracted from the canonical papers named in the task
> (all well-established, widely-cited works). Each is tagged with its
> [arXiv:xxxx.xxxxx] citation. Where a claim is not directly supported by
> these papers, it is flagged as "no literature found".

---

## Method 1: mixup — convex combination of training pairs [arXiv:1710.09412]

- **Core idea**: Train on convex combinations of pairs of samples and their
  labels: `x' = λ·x_i + (1−λ)·x_j`, `y' = λ·y_i + (1−λ)·y_j`, with
  `λ ~ Beta(α, α)`. This is a data-agnostic, near-zero-cost regularizer that
  reduces memorization and improves generalization/robustness. For CIFAR-10
  with a small CNN, mixup typically gives a solid accuracy boost (paper
  reports gains on CIFAR-10/100 with ResNet/WideResNet; the mechanism —
  smoothing decision boundaries — transfers directly to small models).
- **Implementation points** (in `train.py`, inside `train_one_epoch`):
  1. Draw `lam = np.random.beta(alpha, alpha)` per batch (or per-sample).
  2. `perm = torch.randperm(images.size(0))`; `images_mix = lam*images +
     (1-lam)*images[perm]`.
  3. Compute logits on `images_mix`; loss =
     `lam*criterion(logits, labels) + (1-lam)*criterion(logits, labels[perm])`.
     (Do NOT mix labels into a soft target — the two-term CE form is the
     standard, numerically stable implementation.)
  4. Keep `optimizer.zero_grad()/backward()/step()` unchanged.
- **Expected effect**: +2-4% test acc on the 2-conv baseline; also reduces
  overfitting (train loss stays higher, val gap narrows). Cheap: one extra
  tensor op per batch, negligible CPU cost.
- **Hyperparameters**: `alpha=1.0` (uniform Beta) is the paper default and a
  good starting point; `alpha=0.2` is a milder variant if accuracy dips.
  Apply mixup only to the training loader (never to test/eval).
- **Risk**: mixup slightly slows convergence early; with only 30 epochs keep
  `alpha` modest (0.2–1.0). No literature found for mixup on a 2-conv net
  specifically, but the method is architecture-agnostic.

---

## Method 2: CutMix — region cut-and-paste with area-proportional labels [arXiv:1905.04899]

- **Core idea**: Cut a rectangular patch from image B and paste it onto image
  A; the label is mixed proportionally to the patch area:
  `y' = (1−λ)·y_A + λ·y_B`, where `λ` = patch area fraction. Combines the
  benefits of mixup (smooth boundaries) with Cutout (local feature
  preservation), and is especially strong for small models / limited epochs.
- **Implementation points** (in `train.py`, inside `train_one_epoch`):
  1. Draw `lam ~ Beta(α, α)`; compute patch box
     `(cx, cy, bw, bh)` from `lam` (paper: `bw = W·sqrt(1−λ)`, `bh = H·sqrt(1−λ)`,
     centered at a random point).
  2. `images_cut = images.clone()`; paste `images[perm, :, y1:y2, x1:x2]` into
     `images_cut[:, :, y1:y2, x1:x2]`.
  3. Loss = `(1−lam)*criterion(logits, labels) + lam*criterion(logits, labels[perm])`.
  4. Same optimizer loop as baseline.
- **Expected effect**: +2-4% test acc, comparable to mixup; often slightly
  better than mixup for small models because it keeps local structure intact.
  CPU cost is a few slice/copy ops per batch — cheap.
- **Hyperparameters**: `alpha=1.0` default. Patch size derived from `lam`
  automatically. Works well with batch=128.
- **Risk**: patch placement must be clamped to image bounds (32×32 for
  CIFAR-10). No literature found for CutMix on a 2-conv net specifically, but
  the paper shows strong gains on CIFAR-10 with small ResNets.

---

## Method 3 (secondary): SGDR — cosine annealing with warm restarts [arXiv:1608.03983]

- **Core idea**: Replace the fixed/step learning rate with a cosine schedule
  that periodically resets to the max LR (`T_0` epochs per cycle). The
  restarts let the optimizer escape sharp minima and re-explore, which
  typically improves final accuracy and is nearly free to implement.
- **Implementation points** (in `train.py`):
  1. Replace `torch.optim.Adam` LR handling with a cosine schedule. Simplest:
     use `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
     T_0=10, T_mult=2, eta_min=1e-5)`.
  2. Call `scheduler.step(epoch)` at the end of each epoch in `main()` (pass
     the epoch index for warm-restart scheduling).
  3. Keep the optimizer as Adam (paper uses SGD, but cosine annealing works
     with Adam; no literature found specifically for Adam+SGDR, so treat as
     an empirical choice).
- **Expected effect**: +1-3% test acc over a fixed LR, especially in the last
  epochs; helps the model converge to a better final point within 30 epochs.
- **Hyperparameters**: `T_0=10` (first cycle 10 epochs), `T_mult=2` (cycles
  10→20→40...), `eta_min=1e-5`. With max 30 epochs you get ~1.5 cycles.
- **Risk**: if the cycle is too short the model may not converge within a
  cycle; keep `T_0` ≥ 10 for a 30-epoch budget. This is a scheduler change
  only — no data-path change, so it composes cleanly with mixup/CutMix.

---

## Recommended combination (primary plan)

1. **Primary**: mixup OR CutMix (pick one — CutMix slightly preferred for
   small models; mixup is simpler to implement). Add inside `train_one_epoch`.
2. **Secondary**: SGDR cosine warm restarts on top (scheduler change only).
3. These compose: mixup/CutMix regularize the data path, SGDR improves the
   optimization path. Both are CPU-cheap and fit the 30-epoch budget.

---

## Other papers reviewed (not selected — rationale)

- **Cutout [arXiv:1708.04552]**: random square mask on input. Cheap and
  effective, but mixup/CutMix subsume its benefit and give larger gains for
  small models. Could be added as a third cheap trick if desired.
- **AutoAugment [arXiv:1805.09501]**: learned augmentation policy — requires
  a search/RL phase and is far too expensive for CPU-only 30-epoch training.
  Not recommended here.
- **Label Smoothing [arXiv:1512.00567]**: softens one-hot targets
  (`smoothing=0.1`). Cheap (one-line change to the loss), mild +0.5-1% gain;
  can be stacked with mixup/CutMix but is lower priority.
- **AdamW [arXiv:1711.05101]**: decoupled weight decay. Baseline uses Adam
  without weight decay; switching to AdamW with small `weight_decay=5e-4`
  is a cheap improvement but lower impact than mixup/CutMix for a small net.
- **WideResNet [arXiv:1605.07146]**: architecture change (wider residual
  blocks) — out of scope; task fixes the 2-conv baseline architecture.

---

## Summary of recommended actions (priority order)

| Priority | Method | Citation | Where in train.py | Expected gain |
|----------|--------|----------|-------------------|---------------|
| 1 | mixup | [arXiv:1710.09412] | `train_one_epoch` (mix inputs + 2-term CE) | +2-4% |
| 1 | CutMix | [arXiv:1905.04899] | `train_one_epoch` (patch paste + area loss) | +2-4% |
| 2 | SGDR | [arXiv:1608.03983] | `main()` (add scheduler, step per epoch) | +1-3% |
| 3 | Label Smoothing | [arXiv:1512.00567] | loss criterion | +0.5-1% |
| 3 | AdamW | [arXiv:1711.05101] | optimizer | +0.5-1% |
| — | AutoAugment | [arXiv:1805.09501] | — (too expensive) | skip |
| — | WideResNet | [arXiv:1605.07146] | — (arch change, out of scope) | skip |
