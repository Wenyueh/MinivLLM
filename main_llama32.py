import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,

    'dtype': 'bfloat16',
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,

    'block_size': 256,
    'world_size': 1,

    'enforce_eager': True,

    # model params
    'model_name_or_path': 'meta-llama/Llama-3.2-1B-Instruct',
    'vocab_size': 128256, 
    'hidden_size': 2048,
    'head_dim': 64, 
    'num_qo_heads': 32,
    'num_kv_heads': 8,
    'has_attn_bias': False,
    'rms_norm_epsilon': 1e-5,
    'rope_base': 500000, 
    'max_position_embeddings': 32768, # should be >= max_model_length, max position index allowed in rotary embedding
    'intermediate_size': 8192,
    'ffn_bias': False,
    'num_layers': 16,
    'tie_word_embeddings': True,
    'eos': 128009,  # should match tokenizer.eos_token_id. But strangely, HF says eos is [128001, 128008, 128009].
}

def main():
    model_name = config.get('model_name_or_path')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(config=config)
    
    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
