#!/bin/sh

python3 -m vllm.entrypoints.openai.api_server \
    --model /home/do/ssd/modelhub/llama-2-7b-hf \
    --gpu-memory-utilization 0.9
