import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,
    'vocab_size': 151936,  # Fixed: was 151643, HF model uses 151936
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,  # Fixed: was 64, should be 128 (hidden_size / num_heads for GQA output)
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,  # Fixed: was 10000, HF uses rope_theta=1000000
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768, # should be >= max_model_length, max position index allowed in rotary embedding
    'ffn_bias': False,  # Fixed: HF Qwen3 doesn't use MLP bias
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # Fixed: should match tokenizer.eos_token_id
}

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
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