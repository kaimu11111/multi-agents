from __future__ import annotations
import json
from typing import List, Tuple
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"
PROMPT_JUDGE_OPT_SYSTEM = """\
You are a decision maker for CUDA kernel seed generation.
Given the current list of candidate strategies, decide whether we should gather MORE strategies 
(another analyzer round) or STOP and proceed to coding the seed kernel.

Output format: return ONLY JSON like:
{
  "need_more": true,
  "reason": "Why more strategies are needed or not needed, 1-2 sentences",
  "confidence": 0.85
}

Decision principles:
1) If there are gaps, vagueness, or lack of coverage, set need_more=true.
2) Confidence score: 0.0 ~ 1.0, where 1.0 means absolutely certain.
3) If round_idx >= max_rounds-1, lean toward need_more=false unless coverage is clearly insufficient.
Be concise. No Markdown, no extra text.
"""

PROMPT_JUDGE_OPT_USER = """\
# MODE: seed (Judge)
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}
round_idx: {round_idx}
max_rounds: {max_rounds}
k: {k}

# CURRENT STRATEGIES (deduped, atomic)
{strategies_json}

# ARCHITECTURE FILE (verbatim)
```python
{arch_file_content}
```

# OUTPUT (STRICT)
Return ONLY JSON:
```json
{{
  "need_more": true|false,
  "reason": "short reason (1-2 sentences)",
  "confidence": float between 0.0 and 1.0
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

def render_judge_opt_prompt(
    *,
    gpu_name: str,
    arch_path: str,
    round_idx: int,
    max_rounds: int,
    k: int,
    strategies: List[str],
    encoding: str = "utf-8",
) -> str:
    """Render the user prompt by reading the full file at `arch_path` and embedding it verbatim."""
    arch_text = Path(arch_path).read_text(encoding=encoding)
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    return PROMPT_JUDGE_OPT_USER.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        round_idx=round_idx,
        max_rounds=max_rounds,
        k=k,
        strategies_json=json.dumps(strategies, ensure_ascii=False, indent=2),
        arch_file_content=arch_text,
    )


def build_judge_opt_messages(
    *,
    gpu_name: str,
    arch_path: str,
    round_idx: int,
    max_rounds: int,
    k: int,
    strategies: List[str],
    encoding: str = "utf-8",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Judge in seed mode."""
    user = render_judge_opt_prompt(
        gpu_name=gpu_name,
        arch_path=arch_path,
        round_idx=round_idx,
        max_rounds=max_rounds,
        k=k,
        strategies=strategies,
        encoding=encoding,
    )
    return PROMPT_JUDGE_OPT_SYSTEM, user
