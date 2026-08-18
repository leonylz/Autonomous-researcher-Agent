"""
AutoResearcher Experiment Monitor

The key innovation: ZERO LLM calls during experiment training.

While your model trains (hours/days), the monitor only does:
- Process alive check
- Log file tail read
- GPU utilization check

This means running AutoResearcher 24/7 costs the same as running it
only during the THINK and REFLECT phases.
"""

import json
import logging
import os
import shlex
import time
from typing import Optional

from .execution import ExecutionBackend, LocalExecutionBackend

logger = logging.getLogger("autoresearcher.monitor")


class ExperimentMonitor:
    """Zero-LLM experiment monitoring.

    Design principle: During training, the agent is effectively "sleeping"
    at zero cost. It only wakes up (calls LLM) when training completes
    and results need analysis.
    """

    def __init__(
        self,
        poll_interval: int = 900,
        zero_llm: bool = True,
        backend: Optional[ExecutionBackend] = None,
        divergence_detection: bool = True,
        divergence_rise_streak: int = 3,
    ):
        self.poll_interval = poll_interval  # seconds between checks
        self.zero_llm = zero_llm
        self.backend = backend or LocalExecutionBackend(".")
        self._active_experiments: dict[int, dict] = {}
        # G2 发散检测:NaN / loss 连续上升 → 提前终止(省 GPU 时间)
        self.divergence_detection = divergence_detection
        self.divergence_rise_streak = max(2, divergence_rise_streak)
        self._metrics_history: dict[int, list[dict]] = {}

    @staticmethod
    def _divergence_verdict(history: list[dict], rise_streak: int = 3) -> str:
        """指标历史 → 发散判定(纯函数):''(正常)/ 'nan' / 'loss_rising'。

        - NaN/Inf loss(最近 3 条)→ 发散
        - loss 连续 rise_streak 条单调上升,**且同窗口 test_acc 未改善**
          (下降)→ 发散。只凭 loss 抖动不判发散:收敛区间的 loss 噪声
          (如 0.0067→0.0070)常见而 acc 稳定 —— 冒烟实测曾把健康基线
          误判发散并标记 failed。历史里没有 acc 信号时维持旧行为
          (loss-only 保守判定,保证无 acc 的运行仍有安全网)。
        """
        losses = [h.get("loss") for h in history if h.get("loss") is not None]
        if not losses:
            return ""
        for l in losses[-3:]:
            try:
                f = float(l)
                if f != f or f in (float("inf"), float("-inf")):
                    return "nan"
            except (TypeError, ValueError):
                continue
        if len(losses) >= rise_streak:
            try:
                vals = [float(x) for x in losses[-rise_streak:]]
                if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
                    # 幅度门槛(用户审查/ T6 实测):CIFAR 级小网络早期
                    # loss 抖动(0.50→0.51→0.52)也会三连升 —— 单调上升
                    # 不等于发散。要求窗口内相对上升 ≥ 2% 才算。
                    rel_rise = (vals[-1] - vals[0]) / max(abs(vals[0]), 1e-9)
                    if rel_rise < 0.02:
                        return ""
                    accs = []
                    for h in history[-rise_streak:]:
                        acc = next(
                            (h.get(k) for k in ("test_acc", "accuracy", "acc")
                             if h.get(k) is not None), None)
                        if acc is not None:
                            try:
                                accs.append(float(acc))
                            except (TypeError, ValueError):
                                pass
                    # 有 acc 证据且未恶化 → loss 噪声,不判发散
                    if len(accs) >= 2 and accs[-1] >= accs[0] - 1e-6:
                        return ""
                    return "loss_rising"
            except (TypeError, ValueError):
                pass
        return ""

    def _terminate(self, pid: int) -> None:
        """终止训练进程(backend.cancel 优先,失败兜底 os.kill)。"""
        try:
            cancel = getattr(self.backend, "cancel", None)
            if cancel is not None:
                if cancel(pid):
                    return
        except Exception:
            pass
        try:
            import signal as _signal
            import os as _os
            _os.kill(pid, _signal.SIGTERM)
        except Exception:
            pass

    def launch_experiment(self, command: str, log_file: str, gpu: Optional[str] = None) -> dict:
        """Launch an experiment via nohup and track its PID.

        Args:
            command: The training command to run
            log_file: Path to redirect stdout/stderr
            gpu: CUDA_VISIBLE_DEVICES value

        Returns:
            dict with pid, log_file, start_time
        """
        env = {}
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        experiment = self.backend.launch_command(
            argv=shlex.split(command),
            log_file=log_file,
            env=env,
        )
        experiment.update({
            "start_time": time.time(),
            "command": command,
            "status": "running",
        })
        self._active_experiments[experiment["pid"]] = experiment

        logger.info(f"Launched experiment: PID={experiment['pid']}, cmd={command[:80]}...")
        return experiment

    def track_experiment(self, pid: int, log_file: str, command: str = "",
                         gpu: str = "") -> dict:
        """登记一个由工具层启动的实验,供 wait_for_completion 统计。

        新引擎的 launch 走工具层(launch_experiment → backend.launch_command),
        不经过本类的 launch_experiment —— 若不登记,`_active_experiments`
        恒为空,耗时/状态/final_status 统计全部失真(历史 bug)。幂等:同 pid 不重复登记。
        """
        if pid in self._active_experiments:
            return self._active_experiments[pid]
        experiment = {
            "pid": pid,
            "log_file": log_file,
            "start_time": time.time(),
            "command": command,
            "status": "running",
        }
        self._active_experiments[pid] = experiment
        logger.info(f"Tracked experiment: PID={pid}, log={log_file}")
        return experiment

    def wait_for_completion(self, pid: int, log_file: str, notify: bool = True,
                            on_progress=None, should_stop=None,
                            max_wait_hours: float = 0,
                            stale_after_minutes: float = 60) -> dict:
        """Wait for experiment to complete. ZERO LLM calls during wait.

        This is the core cost-saving mechanism. Instead of asking the LLM
        "is training done?", we just check if the process is alive.

        on_progress：每轮轮询回调一次，且**进入循环前立即回调一次**（初始快照），
        退出循环后回调一次（最终状态）——dashboard 不会错过首尾进度。
        should_stop：可选 callable，每轮轮询前调用；返回 True 则提前退出
        （配合 agent 退出信号，避免 kill -0 轮询卡死进程退出路径）。
        max_wait_hours：可选硬上限（0 = 无限）。进程活着但训练卡死时
        避免无限等待。**超时本身不杀进程**——只有"超时 + 日志长时间无更新
        （> stale_after_minutes）"才判定真卡死并 SIGTERM；训练仍在产出日志
        （如用户设的 max_wait_hours 小于训练时长）→ 标记 interrupted 不杀，
        避免提前杀死在跑的正常训练。
        stale_after_minutes：日志无更新的卡死判定阈值（默认 60 分钟）。
        """
        logger.info(f"Monitoring PID={pid}, polling every {self.poll_interval}s")

        interrupted = False
        timed_out = False
        diverged = ""
        started_at = time.time()
        self._last_log_mtime = self._log_mtime(log_file)

        def _progress_once(tail_lines: list):
            if on_progress is None:
                return
            metrics = self._extract_metrics(tail_lines)
            elapsed = time.time() - self._active_experiments.get(pid, {}).get("start_time", time.time())
            on_progress({
                "pid": pid,
                "elapsed_hours": round(elapsed / 3600, 2),
                "detail": tail_lines[-1] if tail_lines else "",
                "epoch": metrics.get("epoch"),
                "loss": metrics.get("loss"),
                "accuracy": metrics.get("accuracy"),
            })

        # 初始快照：进入循环前立即回调一次
        _progress_once(self._safe_tail_file(log_file, lines=5))

        while self._is_process_alive(pid):
            # 最长等待硬上限：进程活着但训练卡死时避免无限等待
            if max_wait_hours > 0 and (time.time() - started_at) > max_wait_hours * 3600:
                logger.warning(
                    f"PID={pid} exceeded max_wait_hours={max_wait_hours}, "
                    f"forcing monitor timeout")
                timed_out = True
                break

            if should_stop is not None:
                try:
                    if should_stop():
                        logger.info(f"PID={pid} monitoring interrupted by stop signal")
                        interrupted = True
                        break
                except Exception:
                    pass  # stop 检查失败不阻塞监控

            time.sleep(self.poll_interval)

            # Log current status (no LLM involved)
            gpu_info = self._safe_gpu_status()
            log_tail = self._safe_tail_file(log_file, lines=5)
            elapsed = time.time() - self._active_experiments.get(pid, {}).get("start_time", time.time())

            logger.info(
                f"PID={pid} alive | elapsed={elapsed/3600:.1f}h | "
                f"GPU={gpu_info.get('utilization', 'N/A')} | "
                f"last_log: {log_tail[-1] if log_tail else 'N/A'}"
            )

            # 记录日志活性（mtime）：用于超时时的"真卡死 vs 仍在跑"判定
            mtime = self._log_mtime(log_file)
            if mtime > 0:
                self._last_log_mtime = mtime

            # 进度回调：把最近日志的 epoch/loss 提取出来（dashboard 实时显示用）
            _progress_once(log_tail)

            # G2 发散检测:NaN / loss 连续上升 → 提前终止(纯规则,零 LLM,
            # 省 GPU 时间 —— 发散训练继续跑只会浪费算力)
            if self.divergence_detection:
                hist = self._metrics_history.setdefault(pid, [])
                m = self._extract_metrics(log_tail)
                if m.get("loss") is not None:
                    hist.append(m)
                verdict = self._divergence_verdict(
                    hist, self.divergence_rise_streak)
                if verdict:
                    # sleep 期间进程可能已自然退出(循环体多跑一轮)——
                    # 此时发散判定已无意义,不杀也不覆盖真实结局
                    # (冒烟实测:健康基线完成 15 epochs,却被迟到一轮的
                    #  loss_rising 判定强制 failed)。
                    if not self._is_process_alive(pid):
                        logger.info(
                            f"PID={pid} {verdict} signal but process already "
                            "exited — keeping natural outcome")
                        break
                    logger.warning(
                        f"PID={pid} divergence detected ({verdict}) — terminating early")
                    self._terminate(pid)
                    diverged = verdict
                    break

        # Experiment finished — ask the backend for the real outcome. Slurm
        # reports the sacct terminal state (so FAILED/TIMEOUT are not mislabelled
        # as success); pid-only backends return unknown and we keep "completed".
        elapsed = time.time() - self._active_experiments.get(pid, {}).get("start_time", time.time())
        log_tail = self._safe_tail_file(log_file, lines=50)

        # 最终回调：退出循环后立即上报一次（中断/正常结束都有）
        _progress_once(log_tail[-5:])

        final = self._safe_final_status(pid)
        success = final.get("success")

        # 超时/中断退出：进程可能还活着——绝不能误报 completed（reflect 会
        # 以为训练成功）。标记 interrupted。
        # 终止策略（关键：不提前杀死用户想保留的训练）：
        #   timed_out（训练卡死超时）→ SIGTERM 终止，防下一轮 GPU 冲突
        #   interrupted（用户 Ctrl+C 退出 agent）→ 不杀，训练继续跑，
        #     重启后由孤儿检测提示用户自行决定
        # 仅 LocalExecutionBackend（SSH/Slurm 的 pid 是远端/slurm job id，
        # os.kill 会误杀本地同 pid 进程，禁止）。
        if interrupted or timed_out:
            status = "interrupted"
            from .execution import LocalExecutionBackend
            if timed_out and isinstance(self.backend, LocalExecutionBackend):
                # 超时杀进程的判据：日志长时间无更新 = 真卡死 → 杀；
                # 日志仍在产出 = 训练还在正常跑（max_wait_hours 设短了）
                # → 不杀，避免提前杀死在跑的程序。
                log_stale = (time.time() - self._last_log_mtime) > stale_after_minutes * 60
                if log_stale:
                    try:
                        import signal as _signal
                        os.kill(pid, _signal.SIGTERM)
                        logger.warning(
                            f"PID={pid} timed out AND log stale >{stale_after_minutes}m: "
                            f"SIGTERM sent (confirmed stuck, prevents GPU conflict)")
                    except (OSError, ProcessLookupError):
                        pass  # 进程已自行退出
                else:
                    logger.info(
                        f"PID={pid} timed out but log still active ({self._last_log_mtime:.0f}): "
                        f"training may still be running — NOT killing")
            elif interrupted:
                logger.info(
                    f"PID={pid} monitoring interrupted (agent exit): "
                    f"training left running; check orphan detection on restart")
            # 中断路径跳过崩溃兜底（status 已确定）
        else:
            status = "failed" if success is False else "completed"

        # 本地/SSH 后端 final_status 返回 success=None，无法识别崩溃。
        # 兜底：若进程非正常结束（成功），检查日志尾部是否有崩溃迹象 →
        # OOM / segfault / Traceback / Killed 标记为 failed（避免静默成功）。
        if status == "completed":
            crash_markers = (
                "Traceback (most recent call last)", "CUDA out of memory",
                "RuntimeError", "Killed", "Segmentation fault", "OutOfMemoryError",
            )
            log_blob = "\n".join(log_tail).lower()
            if any(m.lower() in log_blob for m in crash_markers):
                logger.warning(
                    f"PID={pid} log shows crash indicators; marking failed despite success=None")
                status = "failed"
                success = False

        # 发散终止:提前 kill 的训练标记 failed + terminal_state 说明
        # (truthful:不会被误报为 completed)
        if diverged:
            status = "failed"
            success = False

        if pid in self._active_experiments:
            self._active_experiments[pid]["status"] = status

        result = {
            "pid": pid,
            "status": status,
            "success": success,
            "terminal_state": f"diverged:{diverged}" if diverged
            else final.get("state", "unknown"),
            "elapsed_hours": elapsed / 3600,
            "log_tail": "\n".join(log_tail),
            "metrics": self._extract_metrics(log_tail),
            "interrupted": interrupted,
            "timed_out": timed_out,
            "diverged": diverged,
        }

        logger.info(
            f"Experiment PID={pid} {status} after {result['elapsed_hours']:.1f}h "
            f"(state={result['terminal_state']})"
        )

        if notify:
            self._notify_completion(result)

        return result

    def has_completed_experiments(self) -> bool:
        """Check if any tracked experiment has finished."""
        for pid, exp in list(self._active_experiments.items()):
            if exp["status"] == "running" and not self._is_process_alive(pid):
                exp["status"] = "completed"
                return True
        return False

    def _is_process_alive(self, pid: int) -> bool:
        """Check if process is still running (zero cost)."""
        return self.backend.is_process_alive(pid)

    def _log_mtime(self, log_file: str) -> float:
        """日志文件最后修改时间（epoch）。缺失/不可读 → 0（视为无日志活动）。

        仅对本地后端有效（SSH/Slurm 的 log_file 是远端路径，getmtime 会失败
        返回 0——但非本地后端本来就不执行 kill，无影响）。
        """
        try:
            return os.path.getmtime(log_file)
        except OSError:
            return 0.0

    def _safe_gpu_status(self) -> dict:
        try:
            return self.backend.get_gpu_status()
        except Exception:
            return {"utilization": "N/A"}

    def _safe_final_status(self, pid: int) -> dict:
        try:
            return self.backend.final_status(pid) or {}
        except Exception:
            # Backend without final_status support -> treat as indeterminate.
            return {"state": "unknown", "success": None}

    def _safe_tail_file(self, filepath: str, lines: int = 50) -> list[str]:
        try:
            return self.backend.tail_file(filepath, lines=lines)
        except Exception:
            return []

    def _extract_metrics(self, log_lines: list[str]) -> dict:
        """提取训练指标,两层解析:

        1. **契约行优先**:`METRIC_JSON {...}`(训练模板 log_metrics 输出,
           字段名原样保留如 test_acc)—— 与账本/eval 的 metric_key 直接对上;
        2. 正则 fallback(旧脚本/无契约行):loss/acc/FGD 等常见格式,
           accuracy 归一为 test_acc(模板打印 test_acc,历史 bug:正则
           提取成 accuracy 导致账本取不到)。
        """
        import re
        metrics: dict = {}

        # 1. 契约行(METRIC_JSON)优先,取最新一行
        for line in reversed(log_lines):
            if "METRIC_JSON" not in line:
                continue
            try:
                data = json.loads(line.split("METRIC_JSON", 1)[1].strip())
                if isinstance(data, dict):
                    metrics.update({k: str(v) for k, v in data.items()})
                    break
            except (json.JSONDecodeError, TypeError):
                continue

        # 2. 正则 fallback(契约行缺失时补全;契约行已有时补 epoch 等字段)
        for line in reversed(log_lines):
            for pattern, key in [
                (r"loss[:\s=]+([0-9.]+)", "loss"),
                (r"acc(?:uracy)?[:\s=]+([0-9.]+)", "accuracy"),
                (r"FGD[:\s=]+([0-9.]+)", "FGD"),
                (r"FID[:\s=]+([0-9.]+)", "FID"),
                (r"epoch[:\s=]+(\d+)", "epoch"),
                (r"step[:\s=]+(\d+)", "step"),
            ]:
                if key not in metrics:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        metrics[key] = match.group(1)
        # 归一:accuracy/acc → test_acc(账本/eval 的 metric_key 用 test_acc)
        if "test_acc" not in metrics and "accuracy" in metrics:
            metrics["test_acc"] = metrics["accuracy"]
        return metrics

    def _notify_completion(self, result: dict):
        """Send notification when experiment finishes (success or failure)."""
        outcome = result.get("status", "completed").upper()
        logger.info(
            f"EXPERIMENT {outcome} | PID={result['pid']} | "
            f"Time={result['elapsed_hours']:.1f}h | "
            f"State={result.get('terminal_state', '?')} | "
            f"Metrics={result.get('metrics', {})}"
        )
