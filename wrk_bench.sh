#!/bin/sh
wrk2 -t 8 -c 80 -d 60s http://127.0.0.1:8000/v1/completions -L -U -s wrk/llamacpp.lua -R 100

