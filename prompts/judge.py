from __future__ import annotations
import json
from typing import List, Tuple
from pathlib import Path
PROMPT_JUDGE_SEED_SYSTEM = """\
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
1) If core baseline topics ([tiling], [smem], [vectorize], [coalesce], [guard], [grid]) are mostly covered and concrete, set need_more=false.
2) If there are gaps, vagueness, or lack of coverage, set need_more=true.
3) Confidence score: 0.0 ~ 1.0, where 1.0 means absolutely certain.
4) If round_idx >= max_rounds-1, lean toward need_more=false unless coverage is clearly insufficient.
Be concise. No Markdown, no extra text.
"""

PROMPT_JUDGE_SEED_USER = """\
# MODE: seed (Judge)
GPU: {gpu_name}
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

def render_judge_seed_prompt(
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
    return PROMPT_JUDGE_SEED_USER.format(
        gpu_name=gpu_name,
        round_idx=round_idx,
        max_rounds=max_rounds,
        k=k,
        strategies_json=json.dumps(strategies, ensure_ascii=False, indent=2),
        arch_file_content=arch_text,
    )


def build_judge_seed_messages(
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
    user = render_judge_seed_prompt(
        gpu_name=gpu_name,
        arch_path=arch_path,
        round_idx=round_idx,
        max_rounds=max_rounds,
        k=k,
        strategies=strategies,
        encoding=encoding,
    )
    return PROMPT_JUDGE_SEED_SYSTEM, user
