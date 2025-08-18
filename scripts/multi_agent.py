# agents/multi_agent.py
"""Main Orchestrator (三代理主算法)

职责：
- 协调 Analyzer → Judge → Coder，生成首个 *seed kernel*。
- 进行一次评估（benchmark），但 **不做修复**。
- 将可运行性标记写入 `metrics["runnable"]` 供后续策略决定（修复/优化）。

约定：
- `analyzer(payload) -> list[str] | dict | str`
- `judge(payload) -> dict`，至少包含字段：`{"need_more": bool}`
- `coder(payload) -> str | list[str]`，返回包含 ```python/```cuda 代码块的文本
- `save_kernel_code(code, out_dir)` 会把代码保存为 .py，返回 Path
- `compare_and_bench(ref_py, test_py, ...)` 返回包含延迟/准确度等指标的字典

使用：
    multi = MultiAgent(coder, analyzer, judge, work_dir, arch_py, gpu_name, ref_py)
    ind = multi.create_seed_kernel_with_agents(max_rounds=3, k=5)

后续可扩展：
- 将“修复/再生成”的策略放在 orchestrator 外层循环中；本文件只生成与评估。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import traceback
import json
from datetime import datetime
from utils.kernel_io import extract_code_block, save_kernel_code, extract_json
from utils.compile_and_run import compare_and_bench
from scripts.individual import KernelIndividual
from prompts.analyzer import build_seed_messages
from prompts.judge import build_judge_seed_messages
from prompts.coder import build_coder_prompts
# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _save_llm_io(agent_name: str, system_prompt: str, user_prompt: str, raw_reply: str, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = save_dir / f"{ts}_{agent_name}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"[Agent]: {agent_name}\n")
        f.write("=== System Prompt ===\n")
        f.write(system_prompt.strip() + "\n\n")
        f.write("=== User Prompt ===\n")
        f.write(user_prompt.strip() + "\n\n")
        f.write("=== Raw Reply ===\n")
        f.write(raw_reply.strip() + "\n")
        
def _last_n_lines(text: str, n: int = 150) -> str:
    lines = str(text).splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else str(text)


def _normalize_strategies(raw: Any) -> List[str]:
    """统一解析 Analyzer 返回的 strategies 列表"""
    if raw is None:
        return []

    # 如果直接是 dict
    if isinstance(raw, dict) and "strategies" in raw:
        return [str(s).strip() for s in raw["strategies"] if str(s).strip()]

    # 如果是字符串，先尝试解析 JSON
    if isinstance(raw, str):
        # 先匹配 ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict) and "strategies" in obj:
                    return [str(s).strip() for s in obj["strategies"] if str(s).strip()]
            except json.JSONDecodeError:
                pass

        # 再匹配裸 JSON { ... } 或 [ ... ]
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict) and "strategies" in obj:
                    return [str(s).strip() for s in obj["strategies"] if str(s).strip()]
            except json.JSONDecodeError:
                pass

        # 如果都不是 JSON，就按行切
        return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    # 如果是 list
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]

    return []



# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

class MultiAgent:
    def __init__(
        self,
        coder,
        analyzer,
        judge,
        work_dir: Path | str,
        arch_py: Path | str,
        gpu_name: str,
        *,
        device_idx: int = 0,
        warmup: int = 5,
        repeat: int = 20,
        tol: float = 1e-4,
    ) -> None:
        self.work_dir = Path(work_dir)
        (self.work_dir / "kernels").mkdir(parents=True, exist_ok=True)

        # 三代理：均为可调用对象（可为函数或类的 __call__）
        self.coder = coder
        self.analyzer = analyzer
        self.judge = judge

        self.arch_py = Path(arch_py)
        self.gpu_name = gpu_name

        self.device_idx = device_idx
        self.warmup = warmup
        self.repeat = repeat
        self.tol = tol

        self.best_kernel: KernelIndividual | None = None
        self.best_score: float = float("-inf")
        self.current_kernel: KernelIndividual | None = None
        self.current_score: float | None = None

    # ------------------------------------------------------------------
    def create_seed_kernel_with_agents(self, *, max_rounds: int = 3, k: int = 5) -> KernelIndividual:
        save_dir = self.work_dir / "llm_logs"
        strategies: List[str] = []

        for round_idx in range(max_rounds):
            # 1) Analyzer
            sys_p, usr_p = build_seed_messages(
                gpu_name=self.gpu_name,
                arch_path=self.arch_py,
                k=k,
                round_idx=round_idx,
                max_rounds=max_rounds,
                existing_strategies=strategies,
            )
            raw_strats = self.analyzer(prompt=usr_p, system_prompt=sys_p)
            _save_llm_io("analyzer", sys_p, usr_p, raw_strats, save_dir)

            strats_json = extract_json(raw_strats)
            new_strats = _normalize_strategies(strats_json)
            for s in new_strats:
                if s and s not in strategies:
                    strategies.append(s)

            # 2) Judge
            sys_p, usr_p = build_judge_seed_messages(
                gpu_name=self.gpu_name,
                arch_path=self.arch_py,
                round_idx=round_idx,
                max_rounds=max_rounds,
                k=k,
                strategies=strategies,
            )
            raw_decision = self.judge(prompt=usr_p, system_prompt=sys_p)
            _save_llm_io("judge", sys_p, usr_p, raw_decision, save_dir)

            decision_json = extract_json(raw_decision)
            if not bool(decision_json.get("need_more", False)):
                break

        # 3) Coder
        sys_p, usr_p = build_coder_prompts(
            gpu_name=self.gpu_name,
            arch_path=self.arch_py,
            strategies=strategies,
        )
        raw_code = self.coder(prompt=usr_p, system_prompt=sys_p)
        _save_llm_io("coder", sys_p, usr_p, raw_code, save_dir)
        code = extract_code_block(raw_code)
        # -------- 4) 落盘 --------
        kernel_dir = self.work_dir / "kernels"
        path = save_kernel_code(code, kernel_dir)
        ind = KernelIndividual(code)
        ind.code_path = path  # type: ignore[attr-defined]

        # -------- 5) 基准评估（失败不修复，仅记录） --------
        try:
            metrics = compare_and_bench(
                ref_py=self.arch_py,
                test_py=path,
                device_idx=self.device_idx,
                warmup=self.warmup,
                repeat=self.repeat,
                tol=self.tol,
            )
            metrics["runnable"] = True
            metrics["phase"] = "seed"
            speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
            metrics["score"] = speedup

            ind.metrics = metrics
            ind.score = speedup

            # 记录当前/最佳
            self.current_kernel = ind
            self.current_score = speedup
            if speedup > (self.best_score if self.best_kernel else float("-inf")):
                self.best_score = speedup
                self.best_kernel = ind
        except Exception:
            ind.metrics = {
                "runnable": False,
                "phase": "seed",
                "error_type": "BenchmarkError",
                "message": _last_n_lines(traceback.format_exc()),
            }
            ind.score = float("-inf")
            self.current_kernel = ind
            self.current_score = None

        return ind

    # def optimize_once(self, prev_ind: KernelIndividual, *, k: int = 5, max_rounds: int = 2) -> KernelIndividual:
    #     """基于上一版本进行**一次优化**（成功/失败都会返回一个新个体）"""
    #     assert prev_ind and prev_ind.code_path, "prev_ind 必须包含 code_path"

    #     # 1) Analyzer：累计优化策略
    #     strategies: list[str] = []
    #     for round_idx in range(max_rounds):
    #         # TODO 设计 Analyzer 用于优化的 prompt 目前假设它返回一个策略列表
    #         payload = {
    #             "mode": "optimize",
    #             "gpu_name": self.gpu_name,
    #             "arch_path": str(self.arch_py),
    #             "k": k,
    #             "current_score": prev_ind.score,
    #             "hardware": self.gpu_name,
    #         }
    #         new_s = _normalize_strategies(self.analyzer(payload))
    #         for s in new_s:
    #             if s and s not in strategies:
    #                 strategies.append(s)
    #         # TODO 设计 Judge 用于优化的 prompt 目前假设它返回一个决策字典
    #         dec = self.judge({
    #             "mode": "optimize",
    #             "strategies": strategies,
    #             "round_idx": round_idx,
    #             "max_rounds": max_rounds,
    #             "gpu_name": self.gpu_name,
    #         }) or {}
    #         if not bool(dec.get("need_more", False)):
    #             break

    #     # 2) Coder：依据优化策略生成代码
    #     # TODO 设计 Coder 用于优化的 prompt 目前假设它返回一个代码块
    #     coder_payload = {
    #         "mode": "optimize",
    #         "arch_path": str(self.arch_py),
    #         "gpu_name": self.gpu_name,
    #         "strategies": strategies,
    #         "old_code": prev_ind.code,
    #     }
    #     raw = self.coder(coder_payload)
    #     code = extract_code_block(_normalize_text(raw))

    #     # 3) 落盘并评估
    #     path = save_kernel_code(code, self.work_dir / "kernels")
    #     ind = KernelIndividual(code); ind.code_path = path  # type: ignore[attr-defined]
    #     try:
    #         metrics = compare_and_bench(
    #             ref_py=self.ref_py, test_py=path,
    #             device_idx=self.device_idx, warmup=self.warmup, repeat=self.repeat, tol=self.tol,
    #         )
    #         metrics["runnable"] = True
    #         metrics["phase"] = "optimize"
    #         speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
    #         metrics["score"] = speedup
    #         ind.metrics, ind.score = metrics, speedup
    #     except Exception:
    #         ind.metrics = {
    #             "runnable": False,
    #             "phase": "optimize",
    #             "error_type": "BenchmarkError",
    #             "message": _last_n_lines(traceback.format_exc()),
    #         }
    #         ind.score = float("-inf")

    #     # 4) 维护 current/best
    #     self.current_kernel, self.current_score = ind, ind.score
    #     if ind.metrics and ind.metrics.get("runnable") and ind.score is not None:
    #         if ind.score > (self.best_score if self.best_kernel else float("-inf")):
    #             self.best_score, self.best_kernel = ind.score, ind
    #     return ind


    # def repair_once(self, prev_ind: KernelIndividual, *, k: int = 5, max_rounds: int = 2) -> KernelIndividual:
    #     """针对“不可运行”的上一版本执行**一次修复**（成功/失败都会返回一个新个体）"""
    #     assert prev_ind and prev_ind.code_path, "prev_ind 必须包含 code_path"

    #     err_type = (prev_ind.metrics or {}).get("error_type")
    #     err_msg = (prev_ind.metrics or {}).get("message", "")

    #     # 1) Analyzer：累计修复策略
    #     strategies: list[str] = []
    #     for round_idx in range(max_rounds):
    #         # TODO 设计 Analyzer 用于修复的 prompt 目前假设它返回一个策略列表
    #         payload = {
    #             "mode": "repair",
    #             "gpu_name": self.gpu_name,
    #             "arch_path": str(self.arch_py),
    #             "k": k,
    #             "error_type": err_type,
    #             "error_message": err_msg,
    #             "old_code": prev_ind.code,
    #         }
    #         new_s = _normalize_strategies(self.analyzer(payload))
    #         for s in new_s:
    #             if s and s not in strategies:
    #                 strategies.append(s)
    #         # TODO 设计 Judge 用于修复的 prompt 目前假设它返回一个决策字典
    #         dec = self.judge({
    #             "mode": "repair",
    #             "strategies": strategies,
    #             "round_idx": round_idx,
    #             "max_rounds": max_rounds,
    #             "gpu_name": self.gpu_name,
    #         }) or {}
    #         if not bool(dec.get("need_more", False)):
    #             break

    #     # 2) Coder：依据修复策略生成补丁后的代码
    #     # TODO 设计 Coder 用于修复的 prompt 目前假设它返回一个代码块
    #     coder_payload = {
    #         "mode": "repair",
    #         "arch_path": str(self.arch_py),
    #         "gpu_name": self.gpu_name,
    #         "strategies": strategies,
    #         "old_code": prev_ind.code,
    #     }
    #     raw = self.coder(coder_payload)
    #     code = extract_code_block(_normalize_text(raw))

    #     # 3) 落盘并评估
    #     path = save_kernel_code(code, self.work_dir / "kernels")
    #     ind = KernelIndividual(code); ind.code_path = path  # type: ignore[attr-defined]
    #     try:
    #         metrics = compare_and_bench(
    #             ref_py=self.ref_py, test_py=path,
    #             device_idx=self.device_idx, warmup=self.warmup, repeat=self.repeat, tol=self.tol,
    #         )
    #         metrics["runnable"] = True
    #         metrics["phase"] = "repair"
    #         speedup = metrics["ref_latency_ms"]["avg"] / max(1e-9, metrics["test_latency_ms"]["avg"])
    #         metrics["score"] = speedup
    #         ind.metrics, ind.score = metrics, speedup
    #     except Exception:
    #         ind.metrics = {
    #             "runnable": False,
    #             "phase": "repair",
    #             "error_type": "BenchmarkError",
    #             "message": _last_n_lines(traceback.format_exc()),
    #         }
    #         ind.score = float("-inf")

    #     # 4) 维护 current/best
    #     self.current_kernel, self.current_score = ind, ind.score
    #     if ind.metrics and ind.metrics.get("runnable") and ind.score is not None:
    #         if ind.score > (self.best_score if self.best_kernel else float("-inf")):
    #             self.best_score, self.best_kernel = ind.score, ind
    #     return ind
    
    
