#!/bin/sh
bin/server -ngl 81 -t 12 \
    -m /home/do/ssd/modelhub/llama-2-7b-hf/llama2-7b.f16.bin \
    -c 2048 -np 2 --host 0.0.0.0 --port 8200 \
    -b 2048 -ub 512
