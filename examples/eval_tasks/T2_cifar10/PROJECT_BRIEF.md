# T2 — CIFAR-10 ResNet-18 from scratch to 85%+

## Goal
Train a ResNet-18 on CIFAR-10 and reach **test accuracy ≥ 85%** without pretrained weights.

## Codebase
- Training script: `train.py` (the agent creates it from `core/train_template.py`)
- Data: auto-download via torchvision
- Checkpoints: `./checkpoints/` (`best_model.pth` kept)

## What to Try
- Start with a plain ResNet-18 baseline (SGD, lr=0.1, cosine or step schedule).
- If below 80%: check LR schedule and augmentation (RandomCrop+Flip; then Cutout/Mixup).
- If 80–85%: add label smoothing or stronger augmentation; verify with the same budget.
- One variable at a time; compare against the last trusted baseline.

## Constraints
- PyTorch, GPU 0 only, max 60 epochs per run.
- MUST use the framework training template and dry-run gate.
- Budget: max 6 cycles; if the 3rd cycle shows no improvement, re-plan instead of repeating.

## Success Criteria
Test accuracy ≥ 85% on the held-out CIFAR-10 test set, recorded as `final_metrics.test_acc`.
