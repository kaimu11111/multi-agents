# multi-agents

'''bash
python3 main_multi_task.py KernelBench/test     --first_n 10     --gpu_name "Quadro RTX 6000"     --work_dir workdir     --device_idx 7     --seed_rounds 3     --seed_k 5         --analyzer_model "Qwen/QwQ-32B"         --analyzer_api vllm         --analyzer_addr localhost         --analyzer_port 8000         --analyzer_max_tokens 8192         --judge_model "Qwen/QwQ-32B"         --judge_api vllm         --judge_addr localhost         --judge_port 8000         --judge_max_tokens 2048         --coder_model "Qwen/QwQ-32B"         --coder_api vllm         --coder_addr localhost         --coder_port 8000         --coder_max_tokens 16384
'''
