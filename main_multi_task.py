# main.py
from __future__ import annotations
import argparse, json, sys, csv, random, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from agents.coder import Coder
from agents.judge import Judge
from agents.analyzer import Analyzer
from scripts.multi_agent import MultiAgent

# -------------------------- utils --------------------------
def _collect_tasks(maybe_dir: Path) -> List[Path]:
    if maybe_dir.is_file():
        return [maybe_dir.resolve()]
    if maybe_dir.is_dir():
        return sorted([p.resolve() for p in maybe_dir.rglob("*.py") if p.is_file()])
    raise FileNotFoundError(f"{maybe_dir} not found")

def _pick_first_n(tasks: List[Path], n: int) -> List[Path]:
    n = max(1, min(max(n, 0), len(tasks)))
    return tasks[:n]

def _sample_tasks(tasks: List[Path], k: int, seed: int | None) -> List[Path]:
    if not tasks:
        raise RuntimeError("No .py tasks found.")
    k = max(1, min(k, len(tasks)))
    if not seed:
        seed = int(time.time())
    rng = random.Random(seed)
    return rng.sample(tasks, k)

def _score_from_metrics(metrics: Dict[str, Any] | None) -> float:
    if not metrics:
        return 0.0
    try:
        return float(metrics.get("score", 0.0))
    except Exception:
        return 0.0

def _runnable_from_metrics(metrics: Dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return bool(metrics.get("runnable", False))

def _save_summary(batch_dir: Path, rows: List[Dict[str, Any]], avg_speedup: float, accuracy: float) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    out_json = {
        "avg_speedup": avg_speedup,
        "accuracy": accuracy,
        "num_tasks": len(rows),
        "tasks": rows,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    (batch_dir / "summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    # CSV
    csv_path = batch_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "best_score", "best_runnable", "task_dir", "metrics_dir"])
        for r in rows:
            w.writerow([
                r["task"],
                f'{r["best_score"]:.6f}',
                int(bool(r["best_runnable"])),
                r["task_dir"],
                r["metrics_dir"],
            ])
        w.writerow([])
        w.writerow(["avg_speedup", f"{avg_speedup:.6f}"])
        w.writerow(["accuracy", f"{accuracy:.6f}"])
    print(f"[GLOBAL] Saved {batch_dir / 'summary.json'}")
    print(f"[GLOBAL] Saved {csv_path}")

# ---------------------------- main ----------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser("Three-Agent Orchestrator (classes) – multi-task")
    p.add_argument("arch_py", type=Path, help="Single task .py or a directory containing multiple .py tasks")
    p.add_argument("--gpu_name", required=True)
    p.add_argument("--work_dir", type=Path, default=Path("workdir"))
    p.add_argument("--device_idx", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--total_rounds", type=int, default=10, help="Total rounds for each task (default: 10)")
    # Analyzer 配置
    p.add_argument("--analyzer_model", required=True)
    p.add_argument("--analyzer_api", default="vllm")
    p.add_argument("--analyzer_temperature", type=float, default=0.9)
    p.add_argument("--analyzer_top_p", type=float, default=0.9)
    p.add_argument("--analyzer_top_k", type=int, default=50)
    p.add_argument("--analyzer_max_tokens", type=int, default=2048)
    p.add_argument("--analyzer_addr", default="localhost")
    p.add_argument("--analyzer_port", type=int, default=30000)

    # Judge 配置
    p.add_argument("--judge_model", required=True)
    p.add_argument("--judge_api", default="vllm")
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=1.0)
    p.add_argument("--judge_top_k", type=int, default=0)
    p.add_argument("--judge_max_tokens", type=int, default=512)
    p.add_argument("--judge_addr", default="localhost")
    p.add_argument("--judge_port", type=int, default=30000)

    # Coder 配置
    p.add_argument("--coder_model", required=True)
    p.add_argument("--coder_api", default="vllm")
    p.add_argument("--coder_temperature", type=float, default=0.2)
    p.add_argument("--coder_top_p", type=float, default=0.95)
    p.add_argument("--coder_top_k", type=int, default=50)
    p.add_argument("--coder_max_tokens", type=int, default=8192)
    p.add_argument("--coder_addr", default="localhost")
    p.add_argument("--coder_port", type=int, default=30000)
    p.add_argument("--coder_num_candidates", type=int, default=1)


    # 多任务控制
    p.add_argument("--first_n", type=int, default=0, help="When arch_py is a directory, take the first N (sorted)")
    p.add_argument("--num_tasks", type=int, default=1, help="When sampling, how many tasks to pick (if >0 and first_n=0)")
    p.add_argument("--shuffle_seed", type=int, default=0, help="Random seed for sampling (0=time)")

    args = p.parse_args(argv)

    # 任务收集
    all_tasks = _collect_tasks(args.arch_py)
    if args.arch_py.is_file():
        picked = all_tasks
        pick_note = args.arch_py.stem
    else:
        if args.first_n and args.first_n > 0:
            picked = _pick_first_n(all_tasks, args.first_n)
            pick_note = f"first{len(picked)}"
            print(f"[Task Picker] Found {len(all_tasks)} tasks, taking first {len(picked)} (sorted).")
        else:
            picked = _sample_tasks(all_tasks, args.num_tasks, args.shuffle_seed)
            pick_note = f"num{len(picked)}_seed{args.shuffle_seed}"
            print(f"[Task Picker] Found {len(all_tasks)} tasks, sampled {len(picked)} with seed={args.shuffle_seed}.")

    # 批次目录
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    batch_name = f"{run_id}_batch_{pick_note}"
    batch_dir = (args.work_dir / "runs" / batch_name).resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BATCH] Output folder: {batch_dir}")

    rows: List[Dict[str, Any]] = []

    # 遍历任务
    for i, task_py in enumerate(picked, 1):
        print(f"\n===== [{i}/{len(picked)}] Running task: {task_py} =====")

        # 每个任务自己的工作目录（统一放到本批次下）
        task_dir = (batch_dir / task_py.stem).resolve()
        kernels_dir = task_dir / "kernels"
        metrics_dir = task_dir / "metrics"
        kernels_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # 初始化三代理（为当前任务单独实例化，避免状态串扰）
        analyzer = Analyzer(
            llm=args.analyzer_model, api_type=args.analyzer_api,
            cfg=dict(
                temperature=args.analyzer_temperature, top_p=args.analyzer_top_p, top_k=args.analyzer_top_k,
                max_tokens=args.analyzer_max_tokens, server_address=args.analyzer_addr, server_port=args.analyzer_port,
            )
        )
        judge = Judge(
            llm=args.judge_model, api_type=args.judge_api,
            cfg=dict(
                temperature=args.judge_temperature, top_p=args.judge_top_p, top_k=args.judge_top_k,
                max_tokens=args.judge_max_tokens, server_address=args.judge_addr, server_port=args.judge_port,
            )
        )
        coder = Coder(
            llm=args.coder_model, api_type=args.coder_api,
            cfg=dict(
                temperature=args.coder_temperature, top_p=args.coder_top_p, top_k=args.coder_top_k,
                max_tokens=args.coder_max_tokens, server_address=args.coder_addr, server_port=args.coder_port,
            )
        )

        multi = MultiAgent(
            coder=coder, analyzer=analyzer, judge=judge,
            work_dir=task_dir,               # ← 指向每个任务自己的子目录
            arch_py=task_py, gpu_name=args.gpu_name,
            device_idx=args.device_idx, warmup=args.warmup, repeat=args.repeat, tol=args.tol,
        )
        current = None
        for round_idx in range(args.total_rounds):
            print(f"[Round {round_idx}/{args.total_rounds}]")

            if round_idx == 0:
                # 0 轮：只生成 seed
                print("[SEED] Initializing with seed kernel…")
                current = multi.create_seed_kernel_with_agents()
                current.save_metrics(metrics_dir)

            else:
                # 后续轮次：根据 runnable 决定 repair / optimize
                runnable = bool(
                    getattr(current, "metrics", {}) and current.metrics.get("runnable", False)
                )

                if runnable:
                    print(f"[Round {round_idx}] Current kernel is runnable → optimizing…")
                    current = multi.optimize_step()
                else:
                    print(f"[Round {round_idx}] Current kernel is NOT runnable → repairing…")
                    current = multi.repair_step()

                current.save_metrics(metrics_dir)

                            
        # 记录 best（稳妥判空）
        best_kernel = getattr(multi, "best_kernel", None)
        best_score = 0.0
        best_runnable = False
        if best_kernel is not None:
            try:
                best_score = float(getattr(best_kernel, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                best_score = 0.0
            m = getattr(best_kernel, "metrics", None)
            if isinstance(m, dict):
                best_runnable = bool(m.get("runnable", False))

        rows.append({
            "task": str(task_py),
            "best_score": best_score,
            "best_runnable": best_runnable,
            "task_dir": str(task_dir),
            "metrics_dir": str(metrics_dir),
        })

    # 全局统计（按各任务的 best 汇总）
    if rows:
        avg_speedup = sum(float(r.get("best_score", 0.0)) for r in rows) / len(rows)
        accuracy = sum(1 for r in rows if bool(r.get("best_runnable", False))) / len(rows)
        print("\n===== SUMMARY =====")
        for r in rows:
            print(f"{r['task']}: best_score={float(r.get('best_score', 0.0)):.4f}  runnable={bool(r.get('best_runnable', False))}")
        print(f"\n[GLOBAL] Avg speedup={avg_speedup:.4f}, Accuracy={accuracy:.4f}")
        _save_summary(batch_dir, rows, avg_speedup, accuracy)
    else:
        print("No tasks were run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
