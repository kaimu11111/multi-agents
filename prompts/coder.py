# prompts/coder_prompt.py
from __future__ import annotations
import importlib.util
import importlib.machinery
from pathlib import Path
from textwrap import dedent
import sys
import json
from typing import List, Optional, Tuple
ROOT = Path(__file__).resolve().parents[1]
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"   # original Model
FEWSHOT_NEW = ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised ModelNew
# Paths

HW_FILE = ROOT / "prompts" / "hardware" / "gpu_specs.py"

CODER_SYSTEM = """\
You are a senior CUDA-kernel optimisation specialist. Your job is to generate a high-quality,
compilable, and runnable Python script that builds and launches **hand-written CUDA kernels**
according to the provided STRATEGIES and the target ARCHITECTURE. Prioritise correctness first,
then baseline performance. Respect the architecture's I/O signatures. Prefer coalesced access,
reasonable tiling, and shared memory as indicated by strategies. If a strategy cannot be safely
applied, choose the closest safe alternative and keep the code runnable.

IMPORTANT:
Do not think too much. Think a little then give the one—fenced Python block.
"""

# New user template with strict ModelNew requirements and optional few-shot example
CODER_USER_TMPL = """\
You are a CUDA‑kernel optimisation specialist.
Target GPU: **NVIDIA {gpu_name} ({gpu_arch})**
{gpu_items}

Task
----
Generate **hand‑written CUDA kernels** that replace *all* PyTorch operator(s)
inside the original `class Model` (shown later). You may fuse multiple
operators into a single kernel if that yields better performance. Leave any
non‑replaced parts of the model unchanged.

[STRATEGIES JSON]
{strategies_json}

Target architecture (to optimise):
```python
{arch_file_content}
```

Few‑shot example (reference only – do **not** echo):
**Original**
```python
{few_base}
```
**Optimised**
```python
{few_new}
```

OUTPUT RULES (STRICT)
──────────────────────────────────────────────
1. Reply with **one—and only one—fenced Python block**. No prose.
2. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, and `load_inline` (or cpp_extension if you choose).
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper/launcher).
   3. `cpp_src` – C++ prototypes for *all* kernels you expose.
   4. **Exactly one** load/compile call (e.g., `load_inline`).
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls your CUDA kernels.
3. Handle boundary conditions and alignment safely (fallback to scalar path if misaligned).
4. **Do NOT include** testing code, logging/prints, `if __name__ == "__main__"`, or extra prose.


# ==========================================================
# OUTPUT FORMAT – copy verbatim
Return **exactly one** fenced block labelled `python`. No text before or after.
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

def build_coder_prompts(
    *,
    gpu_name: str,
    arch_path: str | Path,
    strategies: List[str] | dict,
    arch_encoding: str = "utf-8",
    few_base: Optional[str] = None,
    few_new: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the Coder with strict ModelNew output rules.

    Args:
        gpu_name: GPU name key in GPU_SPEC_INFO.
        arch_path: path to the architecture .py file (embedded verbatim).
        strategies: list of strategy strings OR a dict like {"strategies":[...], ...}
        arch_encoding: file encoding for the architecture file.
        few_base: optional few-shot "Original" example (Python code string).
        few_new: optional few-shot "Optimised" example (Python code string).
        history_block: optional context of existing kernels (any text/code to show differences).

    Returns:
        (system_prompt, user_prompt)
    """
    gpu_arch, gpu_items = _load_gpu_info(gpu_name)
    arch_src = _read_text(arch_path, encoding=arch_encoding)

    if isinstance(strategies, dict):
        strategies_json = json.dumps(strategies, ensure_ascii=False, indent=2)
    else:
        strategies_json = json.dumps({"strategies": [str(s) for s in strategies]}, ensure_ascii=False, indent=2)

    few_base = FEWSHOT_BASE.read_text().strip()
    few_new = FEWSHOT_NEW.read_text().strip()


    user = CODER_USER_TMPL.format(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        arch_file_content=arch_src,
        strategies_json=strategies,
        few_base=few_base,
        few_new=few_new,
    )
    return CODER_SYSTEM, dedent(user)

# Quick demo
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_name", required=True)
    ap.add_argument("--arch_path", required=True)
    ap.add_argument("--strategies_json", help="JSON string; defaults to a small example")
    ap.add_argument("--few_base_path")
    ap.add_argument("--few_new_path")
    ap.add_argument("--history_path")
    args = ap.parse_args()

    if args.strategies_json:
        try:
            strategies = json.loads(args.strategies_json)
        except Exception:
            print("Invalid strategies_json; falling back to example list.")
            strategies = {"strategies": [
                "[tiling] BLOCK_M=128, BLOCK_N=64, BLOCK_K=32",
                "[smem] cache A/B tiles in shared memory; avoid bank conflicts (align to 128B)",
                "[vectorize] float4 loads with 16B alignment; fallback to scalar when misaligned",
            ]}
    else:
        strategies = {"strategies": [
            "[tiling] BLOCK_M=128, BLOCK_N=64, BLOCK_K=32",
            "[smem] cache A/B tiles in shared memory; avoid bank conflicts (align to 128B)",
            "[vectorize] float4 loads with 16B alignment; fallback to scalar when misaligned",
        ]}

    few_base = Path(args.few_base_path).read_text(encoding="utf-8") if args.few_base_path else None
    few_new = Path(args.few_new_path).read_text(encoding="utf-8") if args.few_new_path else None
    history = Path(args.history_path).read_text(encoding="utf-8") if args.history_path else None

    sys_p, usr_p = build_coder_prompts(
        gpu_name=args.gpu_name,
        arch_path=args.arch_path,
        strategies=strategies,
        few_base=few_base,
        few_new=few_new,
    )

    print("=== SYSTEM PROMPT ===")
    print(sys_p)
    print("\n=== USER PROMPT ===")
    print(usr_p)
