# main.py
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.coder import Coder
from agents.judge import Judge
from agents.analyzer import Analyzer
from scripts.multi_agent import MultiAgent

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser("Three-Agent Orchestrator (classes)")
    p.add_argument("arch_py", type=Path)
    p.add_argument("--gpu_name", required=True)
    p.add_argument("--work_dir", type=Path, default=Path("workdir"))
    p.add_argument("--device_idx", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--tol", type=float, default=1e-4)

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

    # 流程控制
    p.add_argument("--seed_rounds", type=int, default=3)
    p.add_argument("--seed_k", type=int, default=5)
    p.add_argument("--repairs", type=int, default=3)
    p.add_argument("--opt_iters", type=int, default=5)
    p.add_argument("--opt_rounds", type=int, default=2)
    p.add_argument("--opt_k", type=int, default=5)

    args = p.parse_args(argv)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.work_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "kernels").mkdir(parents=True, exist_ok=True)

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
        work_dir=args.work_dir, arch_py=args.arch_py, gpu_name=args.gpu_name,
        device_idx=args.device_idx, warmup=args.warmup, repeat=args.repeat, tol=args.tol,
    )

    print("[Seed] generating…")
    seed = multi.create_seed_kernel_with_agents(max_rounds=args.seed_rounds, k=args.seed_k)
    seed.save_metrics(Path("workdir") / "metrics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
