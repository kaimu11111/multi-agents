# prompts/coder_prompt.py
from __future__ import annotations
import importlib.util
import importlib.machinery
from pathlib import Path
from textwrap import dedent
import sys
import json
from typing import List, Optional, Tuple, Dict, Union
ROOT = Path(__file__).resolve().parents[1]


HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"

CODER_REPAIR_SYSTEM = """\
You are a senior CUDA-kernel correctness engineer. Your job is to generate a high-quality,
**compilable and runnable** Python script that builds and launches hand-written CUDA code
by **repairing the CURRENT kernel**. Apply only the provided fixes/error context. Do not
introduce performance optimizations or architectural changes. Prioritize correctness and
compilation. Preserve public APIs/signatures and any unrelated logic.

IMPORTANT:
Do not think too much. Think a little then give the one—fenced Python block.
The output must be exactly one fenced code block starting with ```python and ending with ```.
```python
# <complete ModelNew code>
```
"""


# New user template with strict ModelNew requirements and optional few-shot example
CODER_USER_TMPL ="""
You are a CUDA‑kernel optimisation specialist.
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}

Task
----
Repair the **current CUDA kernel** using the given context. Make the **minimal** changes
needed to compile and run correctly. Do not change public APIs or behavior; avoid any
performance tuning.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Reply with **one—and only one—fenced Python block**.  No prose.
2. The block must be directly runnable:
       python model_new.py
3. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
4. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

[EXISTING_ERRORS JSON]
{existing_errors_json}

[METRICS JSON]
{metrics_json}

Current kernel to repair (verbatim):
```python
{current_kernel_src}
```
Now output **only** the complete, runnable `ModelNew` script that satisfies
**ALL OUTPUT RULES (STRICT)** above.
# ==========================================================
# OUTPUT FORMAT – copy verbatim
Return **exactly one** fenced block labelled `python`.  No text before or after.
Use this skeleton (the model must fill it):

```python
# <complete ModelNew code>
```
# ==========================================================
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


def _read_text(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def build_coder_repair_prompts(
    *,
    gpu_name: str,
    current_kernel_path: str | Path,
    existing_errors: Union[List[str], Dict[str, object], None] = None,
    metrics: Optional[Dict[str, object]] = None,
    kernel_encoding: str = "utf-8",
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the Repair Coder with strict output rules.

    Args:
        gpu_name: GPU name key in GPU_SPEC_INFO.
        current_kernel_path: path to the current kernel file (embedded verbatim).
        existing_errors: list of known errors OR dict (JSON-serialized).
        metrics: metrics/result dict providing context.
        kernel_encoding: file encoding for the kernel file.

    Returns:
        (system_prompt, user_prompt)
    """
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    kernel_src = _read_text(current_kernel_path, encoding=kernel_encoding)

    if isinstance(existing_errors, dict) or existing_errors is None:
        existing_errors_json = json.dumps(existing_errors or [], ensure_ascii=False, indent=2)
    else:
        existing_errors_json = json.dumps([str(e) for e in existing_errors], ensure_ascii=False, indent=2)

    metrics_json = json.dumps(metrics or {}, ensure_ascii=False, indent=2)

    user = CODER_USER_TMPL.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        current_kernel_src=kernel_src,
        existing_errors_json=existing_errors_json,
        metrics_json=metrics_json,
    )
    return CODER_REPAIR_SYSTEM, dedent(user)
