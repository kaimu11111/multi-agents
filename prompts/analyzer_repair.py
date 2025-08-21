from __future__ import annotations


from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import importlib.util
import sys
__all__ = [
"PROMPT_REPAIR_SYSTEM",
"PROMPT_REPAIR_USER",
"render_repair_prompt",
"build_repair_messages",
]

ROOT = Path(__file__).resolve().parents[1]


HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"

PROMPT_REPAIR_SYSTEM = """\
You are a senior CUDA debugging and correctness engineer.


Task
----
Given GPU/architecture context, the current CUDA kernel, and run metrics,
identify the concrete ERROR POINTS and provide the corresponding FIX SUGGESTIONS.
Focus strictly on correctness and compilation.


Rules
-----
1) Output format: ONLY one JSON object.
2) Be concise: each error entry should be a short description, each fix entry a minimal safe change.


Required JSON schema
--------------------
```json
{
"errors": [
"Error point 1 (with brief reason)",
"Error point 2"
],
"fixes": [
"Fix suggestion 1 (minimal safe change for error 1)",
"Fix suggestion 2"
]
}
```
"""

PROMPT_REPAIR_USER = """\
# MODE: repair
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}
round_idx: {round_idx}
max_rounds: {max_rounds}

# CONTEXT
existing_errors:
{existing_errors_json}

# CONTEXT
metrics:
{metrics_json}


# CURRENT KERNEL (verbatim)
```python
{current_kernel_src}
```


# NOTE
This is the repair stage: provide high-value, low-risk, directly implementable fixes only.


# OUTPUT (STRICT)
# Return this JSON:
```json
{{
"errors": [
"Error point 1 (with tag and brief reason)",
"Error point 2"
],
"fixes": [
"Fix suggestion 1 (minimal safe change for error 1)",
"Fix suggestion 2"
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

def _read_text(path: Union[str, Path], *, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding).strip()


def _to_pretty_json(obj: Any, *, max_chars: int = 40_000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = json.dumps(str(obj), ensure_ascii=False, indent=2)
    if len(s) > max_chars:
        head = s[: max_chars // 2]
        tail = s[-max_chars // 2 :]
        s = head + "\n...<TRUNCATED>...\n" + tail
    return s


def render_repair_prompt(
    *,
    gpu_name: str,
    current_kernel_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    existing_errors: Optional[List[str]] = None,
    round_idx: int = 0,
    max_rounds: int = 3,
    encoding: str = "utf-8",
) -> str:
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    cur_src = _read_text(current_kernel_path, encoding=encoding)
    existing_errors_json = _to_pretty_json(existing_errors if existing_errors is not None else [])
    metrics_json = _to_pretty_json(metrics if metrics is not None else {})
    return PROMPT_REPAIR_USER.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        round_idx=round_idx,
        max_rounds=max_rounds,
        existing_errors_json=existing_errors_json,
        metrics_json=metrics_json,
        current_kernel_src=cur_src,
    )


def build_repair_messages(
    *,
    gpu_name: str,
    current_kernel_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    existing_errors: Optional[List[str]] = None,
    round_idx: int = 0,
    max_rounds: int = 3,
    encoding: str = "utf-8",
) -> Tuple[str, str]:
    user = render_repair_prompt(
        gpu_name=gpu_name,
        current_kernel_path=current_kernel_path,
        metrics=metrics,
        existing_errors=existing_errors,
        round_idx=round_idx,
        max_rounds=max_rounds,
        encoding=encoding,
    )
    return PROMPT_REPAIR_SYSTEM, user
