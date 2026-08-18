"""
用户画像系统。

让 Agent 了解用户的习惯和偏好，自动适配实验建议：
  - "上次你说单卡显存不够，这次默认用 batch_size=64"
  - "你偏好 PyTorch，不提 TensorFlow"
  - "你喜欢先看结论再看分析"

存储：workspace/user_profile.json（可手动编辑）
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger("autoresearcher.profile")


@dataclass
class UserProfile:
    preferred_framework: str = "pytorch"
    typical_batch_size: int = 128
    typical_dataset: str = ""
    max_gpu_hours_per_experiment: int = 24
    notification_preference: str = "push"
    style: str = "concise"               # "concise" | "detailed"
    preferred_lr_range: str = "1e-4,1e-2"
    typical_optimizer: str = "adamw"
    dodges: list[str] = field(default_factory=list)  # 已知坑，Agent 应避开
    constraints: dict = field(default_factory=dict)   # 自定义约束

    def to_prompt(self) -> str:
        """生成注入 prompt 的用户画像片段。"""
        lines = [
            f"- Framework: {self.preferred_framework}",
            f"- Typical batch size: {self.typical_batch_size}",
            f"- Max GPU hours per experiment: {self.max_gpu_hours_per_experiment}",
            f"- Preferred LR range: {self.preferred_lr_range}",
            f"- Typical optimizer: {self.typical_optimizer}",
        ]
        if self.typical_dataset:
            lines.append(f"- Typical dataset: {self.typical_dataset}")
        if self.dodges:
            lines.append(f"- Known pitfalls to avoid: {', '.join(self.dodges)}")
        if self.constraints:
            for k, v in self.constraints.items():
                lines.append(f"- Constraint [{k}]: {v}")
        return "## User Profile\n" + "\n".join(lines)


class UserProfileStore:
    """读写 workspace/user_profile.json。"""

    def __init__(self, workspace: Path):
        self.path = workspace / "user_profile.json"

    def load(self) -> UserProfile:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return UserProfile(**{k: v for k, v in data.items()
                                       if k in UserProfile.__dataclass_fields__})
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(f"Failed to load user profile: {exc}")
        return UserProfile()

    def save(self, profile: UserProfile) -> None:
        self.path.write_text(
            json.dumps(asdict(profile), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_dodge(self, lesson: str) -> None:
        """从实验中学习：记录一个已知陷阱。"""
        profile = self.load()
        if lesson not in profile.dodges:
            profile.dodges.append(lesson)
            self.save(profile)
