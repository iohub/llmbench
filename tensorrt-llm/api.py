from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


app = FastAPI()


class OpenAIReq(BaseModel):
    prompt: str
    max_tokens: int




def tokenize(tokenizer,
                input_text=None,
                prompt_template=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    print(input_text)
    for curr_text in input_text:
        if prompt_template is not None:
            curr_text = prompt_template.format(input_text=curr_text)
        input_ids = tokenizer.encode(curr_text,
                                        add_special_tokens=add_special_tokens,
                                        truncation=True,
                                        max_length=max_input_length)
        batch_input_ids.append(input_ids)

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def decode_text(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths):
    
    batch_size, num_beams, _ = output_ids.size()
    texts = []
    for batch_idx in range(batch_size):
        inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
        )
        input_text = tokenizer.decode(inputs)
        print(f'Input [Text {batch_idx}]: \"{input_text}\"')
        for beam in range(num_beams):
            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            texts.append(output_text)
    return texts



DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'qwen':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    'llama':
    '<s>[INST] 你是一个AI编码助手，使用中文回答一下问题： {input_text} [/INST]'
}


def load_runner(engine_dir, lora_dir, max_input_len, max_output_len, num_beams):
    runtime_rank = tensorrt_llm.mpi_rank()
    use_py_session = False
    if not PYTHON_BINDINGS and not use_py_session:
        logger.warning("Python bindings of C++ session is unavailable, fallback to Python session.")
        use_py_session = True
    
    runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=engine_dir,
                         lora_dir=lora_dir,
                         rank=runtime_rank,
                         debug_mode=False,
                         lora_ckpt_source=None)

    if not use_py_session:
        runner_kwargs.update(
            max_batch_size=1,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_beam_width=num_beams,
        )
    return runner_cls.from_dir(**runner_kwargs)    

engine_dir = '/home/do/ssd/modelhub/trt-engines/llama-2-7b'
tokenizer_dir = '/home/do/ssd/modelhub/llama-2-7b-hf'

model_name, model_version = read_model_name(engine_dir)
tokenizer, pad_id, end_id = load_tokenizer(
    tokenizer_dir=tokenizer_dir,
    model_name=model_name,
    model_version=model_version
)

CTX = {
    'runner': load_runner(engine_dir=engine_dir, 
                          lora_dir=None,
                          max_input_len=1000,
                          max_output_len=200,
                          num_beams=1),
    'tokenizer': tokenizer,
    'padid': pad_id,
    'endid': end_id,
}


def generate(input_text, max_tokens):

    runtime_rank = tensorrt_llm.mpi_rank()
    tokenizer = CTX['tokenizer']
    stop_words_list = None
    bad_words_list = None

    prompt_template = DEFAULT_PROMPT_TEMPLATES['llama']
    batch_input_ids = tokenize(tokenizer=tokenizer,
                                  input_text=input_text,
                                  prompt_template=prompt_template,
                                  add_special_tokens=True,
                                  max_input_length=1024,
                                  pad_id=CTX['padid'],
                                  num_prepend_vtokens=[],
                                  model_name=model_name,
                                  model_version=model_version)
    input_lengths = [x.size(0) for x in batch_input_ids]
 
    runner = CTX['runner']

    temperature, top_k, top_p, num_beams = 1.0, 10, 0.9, 1

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids,
            max_new_tokens=max_tokens,
            end_id=end_id,
            pad_id=pad_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            streaming=False,
            output_sequence_lengths=True,
            return_dict=True)
        torch.cuda.synchronize()


    if runtime_rank == 0:
        output_ids = outputs['output_ids']
        sequence_lengths = outputs['sequence_lengths']
        return decode_text(tokenizer,
                        output_ids,
                        input_lengths,
                        sequence_lengths)




@app.post("/v1/completions")
async def api_v1_completions(req: OpenAIReq):
    texts = generate([req.prompt], req.max_tokens)
    return {"texts": texts}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)