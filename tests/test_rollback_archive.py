"""WorkspaceSnapshot 归档内容测试:SQLite 瞬态文件不入快照。

冒烟实测修复:cycle0 快照偶发 Errno 13(Windows 读共享冲突),且 -wal/-shm
打进快照会让 .db 与不同时刻的 WAL 错配。两者都通过跳过瞬态文件解决。
"""
import gc
import tarfile
import tempfile
import unittest
from pathlib import Path

from core.rollback import WorkspaceSnapshot


class SnapshotArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ws = Path(self.tempdir.name) / "workspace"
        self.ws.mkdir()

    def tearDown(self):
        import shutil
        gc.collect()
        shutil.rmtree(self.tempdir.name, ignore_errors=True)

    def _snapshot_and_read_tar(self):
        snap = WorkspaceSnapshot(self.ws, keep=5)
        path = snap.create(label="test", cycle=1)
        with tarfile.open(path, "r:gz") as tar:
            return sorted(tar.getnames())

    def test_transient_sqlite_files_excluded(self):
        (self.ws / "train.py").write_text("print(1)\n", encoding="utf-8")
        # SQLite 运行期瞬态文件(快照时刻可能正被本进程/子进程打开)
        for name in ("checkpoints.db", "checkpoints.db-wal", "checkpoints.db-shm",
                     "memory.db-wal", "hypotheses.db-journal"):
            (self.ws / name).write_text("x", encoding="utf-8")

        names = self._snapshot_and_read_tar()
        self.assertIn("train.py", names)
        self.assertIn("checkpoints.db", names, "主 .db 仍应入档(不含瞬态 WAL)")
        for banned in ("checkpoints.db-wal", "checkpoints.db-shm",
                       "memory.db-wal", "hypotheses.db-journal"):
            self.assertNotIn(banned, names, f"{banned} 不应被打进快照")

    def test_cost_and_audit_still_excluded(self):
        (self.ws / "costs.jsonl").write_text("{}", encoding="utf-8")
        (self.ws / "audit.jsonl").write_text("{}", encoding="utf-8")
        (self.ws / "events.jsonl").write_text("{}", encoding="utf-8")
        names = self._snapshot_and_read_tar()
        self.assertNotIn("costs.jsonl", names)
        self.assertNotIn("audit.jsonl", names)
        self.assertIn("events.jsonl", names, "事件日志是有价值的证据,应保留")

    def test_dataset_dirs_excluded(self):
        """T6 实测修复:60k PNG 的 CIFAR-10 数据目录让快照 tar 10+ 分钟。
        数据集是静态资产,不入快照(快照只保存代码/权重)。"""
        data = self.ws / "CIFAR-10" / "truck"
        data.mkdir(parents=True)
        for i in range(5):
            (data / f"truck-{i:05d}.png").write_bytes(b"x" * 100)
        (self.ws / "train.py").write_text("print(1)\n", encoding="utf-8")

        names = self._snapshot_and_read_tar()
        self.assertIn("train.py", names)
        self.assertNotIn("CIFAR-10", " ".join(names), "数据目录不应被打进快照")
        self.assertFalse(any("truck" in n for n in names), "PNG 不应入档")


if __name__ == "__main__":
    unittest.main()
