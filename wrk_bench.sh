#!/bin/sh
wrk -t 8 -c 80 -d 60s -s wrk/vllm.lua -v http://127.0.0.1:8000/v1/completions
