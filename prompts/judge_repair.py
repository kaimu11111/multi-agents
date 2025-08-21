from __future__ import annotations
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]


HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"

PROMPT_JUDGE_REPAIR_SYSTEM = """\
You are a decision maker for CUDA kernel repair.
Given the current ERROR LIST, metrics, and the current kernel source,
decide whether the error list is COMPLETE (i.e., covers all real issues).

Output format: return ONLY JSON like:
{
  "is_complete": true|false,
  "missing_errors": ["..."],        // errors that are evident but absent from the list
  "spurious_errors": ["..."],       // listed items that aren't supported by source/metrics
  "reason": "Why the list is or isn't complete, 1-2 sentences",
  "confidence": 0.85
}

Decision principles:
1) Judge unique root causes (not duplicate symptoms).
2) Mark 'missing' if an issue is evident in source/metrics but not in the list.
3) Mark 'spurious' if an item is unsupported by source/metrics.
4) Confidence: 0.0 ~ 1.0 (1.0 = absolutely certain).
Be concise. No Markdown, no extra text.
"""

PROMPT_JUDGE_REPAIR_USER = """\
# MODE: repair (Judge)
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}
round_idx: {round_idx}
max_rounds: {max_rounds}

# EXISTING ERRORS (deduped, atomic)
{existing_errors_json}

# METRICS (pretty JSON)
{metrics_json}

# CURRENT KERNEL (verbatim)
```python
{current_kernel_src}
```

# OUTPUT (STRICT)
Return ONLY JSON:
```json
{{
  "is_complete": true|false,
  "missing_errors": ["..."],
  "spurious_errors": ["..."],
  "reason": "short reason (1-2 sentences)",
  "confidence": 0.0-1.0
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

def render_judge_repair_prompt(
    *,
    gpu_name: str,
    current_kernel_path: str,
    existing_errors: Optional[List[str]],
    metrics: Optional[Dict[str, object]] = None,
    round_idx: int = 0,
    max_rounds: int = 3,
    encoding: str = "utf-8",
) -> str:
    """Render the user prompt by reading the current kernel and embedding context."""
    kernel_text = Path(current_kernel_path).read_text(encoding=encoding)
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    return PROMPT_JUDGE_REPAIR_USER.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,   
        round_idx=round_idx,
        max_rounds=max_rounds,
        existing_errors_json=json.dumps(existing_errors or [], ensure_ascii=False, indent=2),
        metrics_json=json.dumps(metrics or {}, ensure_ascii=False, indent=2),
        current_kernel_src=kernel_text,
    )

def build_judge_repair_messages(
    *,
    gpu_name: str,
    current_kernel_path: str,
    existing_errors: Optional[List[str]],
    metrics: Optional[Dict[str, object]] = None,
    round_idx: int = 0,
    max_rounds: int = 3,
    encoding: str = "utf-8",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the Judge in repair mode."""
    user = render_judge_repair_prompt(
        gpu_name=gpu_name,
        current_kernel_path=current_kernel_path,
        existing_errors=existing_errors,
        metrics=metrics,
        round_idx=round_idx,
        max_rounds=max_rounds,
        encoding=encoding,
    )
    return PROMPT_JUDGE_REPAIR_SYSTEM, user
