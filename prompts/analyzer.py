# -*- coding: utf-8 -*-
"""
Seed-only Analyzer prompt module that inlines the entire architecture file into the user prompt.

Exports
-------
- PROMPT_SEED_SYSTEM: str  -> System prompt (fixed rules for seed stage)
- PROMPT_SEED_USER:   str  -> User prompt template (with placeholders)
- render_seed_prompt: func -> Renders the final user prompt string by reading arch_path
- build_seed_messages:func -> Returns (system_prompt, user_prompt) tuple for convenience

Notes
-----
1) The LLM is expected to return ONLY JSON of the form: {"strategies": ["...", "..."]}.
2) The architecture file is embedded verbatim inside a fenced code block for clarity.
3) Curly braces used for demonstration JSON are doubled {{ }} to be safe with str.format().
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import json
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"
__all__ = [
    "PROMPT_SEED_SYSTEM",
    "PROMPT_SEED_USER",
    "render_seed_prompt",
    "build_seed_messages",
]


PROMPT_OPT_SYSTEM = """\
You are a senior GPU performance analyst and CUDA kernel optimization advisor.
Task: Based on the provided GPU/architecture information and context, produce a set of strategies for generating the first seed kernel.
Goal: The resulting kernel should compile, run correctly, and provide a reasonable baseline performance (not maximum performance).

Rules:
1. Output format: ONLY JSON of the form {"strategies": ["...", "..."]}. No explanations, no Markdown fences, no extra text.
2. Number of strategies: ≤ k. Each strategy must be minimal, atomic, and non-overlapping; avoid vague phrases.
3. Seed focus: Prefer high-value, low-risk, easy-to-implement "foundation" strategies.
4. Hardware awareness: Use gpu_name/arch_path context to suggest concrete parameters (tile/block sizes, vector widths, double-buffering, etc.).
5. Deduplication: Do not include strategies that conflict with or duplicate any in existing_strategies.

IMPORTANT:
Do not think too much. Think a little then give the Strategies list.
"""

# Placeholders: {gpu_name} {arch_path} {k} {round_idx} {max_rounds} {existing_strategies_json} {arch_file_content}
PROMPT_OPT_USER = """\
# MODE: seed
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}
k: {k}
round_idx: {round_idx}
max_rounds: {max_rounds}

# CONTEXT
existing_strategies:
{existing_strategies_json}

# ARCHITECTURE FILE (verbatim)
```python
{arch_file_content}
```

# NOTE
This is the seed stage: provide high-value, low-risk, directly implementable "foundation" strategies only.
Example topics (seed scope):
- [tiling] Reasonable BLOCK_M/BLOCK_N/BLOCK_K and thread-block layout (explicit candidates or preferred choice)
- [smem]   Shared-memory tiling and bank-conflict avoidance
- [vectorize] Vectorized load/store width (e.g., float4) with alignment fallback
- [coalesce] Coalesced global memory pattern (row/col-major, stride)
- [double-buffer] Whether to enable K-dim double buffering (default stage=2 is optional)
- [broadcast] Block-level reuse of common weights/fragments
- [guard] Boundary/masking via predication
- [grid]  Launch config derivation (gridDim/blockDim)
- [occupancy] Conservative unroll vs register pressure to maintain occupancy
- [align] Alignment requirements (16/32/128B) and safe fallback

# OUTPUT (STRICT)
# Return this JSON:
```json
{{
  "strategies": [
    "Strategy 1 (tags + key parameters + brief rationale/expected impact)",
    "Strategy 2",
    "..."
  ]
}}
```

"""

def _load_gpu_info(gpu_name: str) -> tuple[str, str]:
    """Return (arch, formatted_items) from prompts/hardware/gpu_specs.py"""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    gpu_info = getattr(module, "GPU_SPEC_INFO")
    if gpu_name not in gpu_info:
        raise KeyError(f"GPU '{gpu_name}' not found in GPU_SPEC_INFO")
    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")
    return gpu_arch, items

def render_seed_prompt(
    *,
    gpu_name: str,
    arch_path: str,
    k: int,
    round_idx: int = 0,
    max_rounds: int = 3,
    existing_strategies: List[str] | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Read the full file at `arch_path` and embed it verbatim into the user prompt.

    Parameters
    ----------
    gpu_name : str
        GPU name (e.g., "Quadro RTX 6000").
    arch_path : str
        Path to the architecture Python file to inline.
    k : int
        Maximum number of strategies to request from the model.
    round_idx : int, default 0
        Current analyzer round (0-based).
    max_rounds : int, default 3
        Maximum analyzer rounds.
    existing_strategies : List[str] | None
        Already-collected strategies (used for deduplication hints).
    encoding : str, default "utf-8"
        File encoding for reading `arch_path`.

    Returns
    -------
    str
        A formatted user prompt string ready to send to the LLM.
    """
    arch_src = Path(arch_path).read_text().strip()
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    return PROMPT_OPT_USER.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        k=k,
        round_idx=round_idx,
        max_rounds=max_rounds,
        existing_strategies_json= existing_strategies,
        arch_file_content=arch_src,
    )


def build_opt_messages(
    *,
    gpu_name: str,
    arch_path: str,
    k: int,
    round_idx: int = 0,
    max_rounds: int = 3,
    existing_strategies: List[str] | None = None,
    encoding: str = "utf-8",
) -> tuple[str, str]:
    """Convenience helper returning (system_prompt, user_prompt)."""
    user = render_seed_prompt(
        gpu_name=gpu_name,
        arch_path=arch_path,
        k=k,
        round_idx=round_idx,
        max_rounds=max_rounds,
        existing_strategies=existing_strategies,
        encoding=encoding,
    )
    return PROMPT_OPT_SYSTEM, user
