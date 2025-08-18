# agents/coder.py
from __future__ import annotations
from typing import Any, Dict, List
from agents.query_server import query_server

class Analyzer:
    def __init__(self, llm: str, api_type: str, cfg: Dict[str, Any] | None = None):
        """
        llm:       模型名称，如 "deepseek-reasoner" / "gpt-4o-mini" / "Qwen/Qwen3-8B"
        api_type:  "openai" | "deepseek" | "together" | "vllm" | "sglang" ...
        cfg:       运行时配置（可覆盖默认值），例如：
                   {
                     "temperature": 0.2, "top_p": 0.9, "top_k": 50,
                     "max_tokens": 2048,
                     "server_address": "localhost", "server_port": 8000,
                     "num_completions": 1,
                     "is_reasoning_model": False, "reasoning_effort": None
                   }
        """
        self.model_name = llm           # 避免和“可调用 llm”混淆
        self.server_type = api_type
        self.cfg = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 2048,
            "server_address": "localhost",
            "server_port": 30000,
            "num_completions": 1,
            "is_reasoning_model": False,
            "reasoning_effort": None,
        }
        if cfg:
            self.cfg.update(cfg)

    def __call__(self, prompt: str | List[dict], system_prompt: str =
                 "You are a CUDA kernel optimisation assistant.") -> str:
        resp = query_server(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            top_k=self.cfg["top_k"],
            max_tokens=self.cfg["max_tokens"],
            num_completions=self.cfg["num_completions"],
            server_port=self.cfg["server_port"],
            server_address=self.cfg["server_address"],
            server_type=self.server_type,
            model_name=self.model_name,             # ← 修正参数名
            is_reasoning_model=self.cfg["is_reasoning_model"],
            reasoning_effort=self.cfg["reasoning_effort"],
        )
        return resp[0] if isinstance(resp, list) else resp
