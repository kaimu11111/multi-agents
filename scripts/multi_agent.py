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
from prompts.analyzer import build_opt_messages
from prompts.judge import build_judge_opt_messages
from prompts.coder import build_coder_prompts
from prompts.coder_opt import build_coder_opt_prompts
from prompts.analyzer_repair import build_repair_messages
from prompts.judge_repair import build_judge_repair_messages
from prompts.coder_repair import build_coder_repair_prompts
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

def _normalize_errors(raw: Any) -> List[str]:
    """统一解析 Analyzer 返回的 errors 列表"""
    if raw is None:
        return []

    # 如果直接是 dict
    if isinstance(raw, dict) and "errors" in raw:
        return [str(e).strip() for e in raw["errors"] if str(e).strip()]

    # 如果是字符串，先尝试解析 JSON
    if isinstance(raw, str):
        # 匹配 ```json ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict) and "errors" in obj:
                    return [str(e).strip() for e in obj["errors"] if str(e).strip()]
            except json.JSONDecodeError:
                pass

        # 匹配裸 JSON { ... } 或 [ ... ]
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict) and "errors" in obj:
                    return [str(e).strip() for e in obj["errors"] if str(e).strip()]
            except json.JSONDecodeError:
                pass

        # 如果不是 JSON，就逐行切
        return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]

    # 如果是 list
    if isinstance(raw, list):
        return [str(e).strip() for e in raw if str(e).strip()]

    return []

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
    def create_seed_kernel_with_agents(self) -> KernelIndividual:
        save_dir = self.work_dir / "llm_logs"
        strategies: List[str] = []
        
        sys_p, usr_p = build_coder_prompts(
            gpu_name=self.gpu_name,
            arch_path=self.arch_py,
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

    def optimize_step(self, k: int = 5, max_rounds: int = 3) -> KernelIndividual:
        """基于上一版本进行**一次优化**（成功/失败都会返回一个新个体）"""

        # 1) Analyzer：累计优化策略
        strategies: list[str] = []
        save_dir = self.work_dir / "llm_logs"
        cur_path = self.current_kernel.code_path
        for round_idx in range(max_rounds):
            # 1) Analyzer
            sys_p, usr_p = build_opt_messages(
                gpu_name=self.gpu_name,
                arch_path=cur_path,
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
            sys_p, usr_p = build_judge_opt_messages(
                gpu_name=self.gpu_name,
                arch_path=cur_path,
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

        # 2) Coder：依据优化策略生成代码
        sys_p, usr_p = build_coder_opt_prompts(
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


    def repair_step(self, max_rounds: int = 3) -> KernelIndividual:
        """基于上一版本进行**一次优化**（成功/失败都会返回一个新个体）"""

        # 1) Analyzer：累计优化策略
        error_list: list[str] = []
        save_dir = self.work_dir / "llm_logs"
        cur_path = self.current_kernel.code_path
        for round_idx in range(max_rounds):
            # 1) Analyzer
            sys_p, usr_p = build_repair_messages(
                gpu_name=self.gpu_name,
                arch_path=cur_path,
                metrics=self.current_kernel.metrics,
                existing_errors=error_list,
                round_idx=round_idx,
                max_rounds=max_rounds,
            )
            raw_strats = self.analyzer(prompt=usr_p, system_prompt=sys_p)
            _save_llm_io("analyzer", sys_p, usr_p, raw_strats, save_dir)

            strats_json = extract_json(raw_strats)
            new_strats = _normalize_errors(strats_json)
            for s in new_strats:
                if s and s not in error_list:
                    error_list.append(s)

            # 2) Judge
            sys_p, usr_p = build_judge_repair_messages(
                gpu_name=self.gpu_name,
                arch_path=cur_path,
                existing_errors=error_list,
                metrics=self.current_kernel.metrics,
                round_idx=round_idx,
                max_rounds=max_rounds,
            )
            raw_decision = self.judge(prompt=usr_p, system_prompt=sys_p)
            _save_llm_io("judge", sys_p, usr_p, raw_decision, save_dir)

            decision_json = extract_json(raw_decision)
            if bool(decision_json.get("is_complete", False)):
                break

        # 2) Coder：依据优化策略生成代码
        sys_p, usr_p = build_coder_repair_prompts(
            gpu_name=self.gpu_name,
            arch_path=cur_path,
            existing_errors=error_list,
            metrics=self.current_kernel.metrics,
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
    
    
