# baseline
启动vllm example：
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server   --model Qwen/QwQ-32B   --tensor-parallel-size 4   --dtype half   --max-model-len 32768   --reasoning-parser qwen3   --port 30000   --trust-remote-code
```

运行程序 example for multi-task：
```bash
python3 main_multi_task.py KernelBench/test     --first_n 10     --gpu_name "Quadro RTX 6000"     --work_dir workdir     --device_idx 7         --analyzer_model "Qwen/QwQ-32B"         --analyzer_api vllm         --analyzer_addr localhost         --analyzer_port 8000         --analyzer_max_tokens 8192         --judge_model "Qwen/QwQ-32B"         --judge_api vllm         --judge_addr localhost         --judge_port 8000         --judge_max_tokens 2048         --coder_model "Qwen/QwQ-32B"         --coder_api vllm         --coder_addr localhost         --coder_port 8000         --coder_max_tokens 16384
```

运行程序 example for single task：
```bash
python3 main_multi_task.py KernelBench/level1/1_Square_matrix_multiplication_.py     --gpu_name "Quadro RTX 6000"     --work_dir workdir     --device_idx 7       --analyzer_model "Qwen/QwQ-32B"         --analyzer_api vllm         --analyzer_addr localhost         --analyzer_port 8000         --analyzer_max_tokens 8192         --judge_model "Qwen/QwQ-32B"         --judge_api vllm         --judge_addr localhost         --judge_port 8000         --judge_max_tokens 2048         --coder_model "Qwen/QwQ-32B"         --coder_api vllm         --coder_addr localhost         --coder_port 8000         --coder_max_tokens 16384
```
