import argparse
import time
import torch

from transformers import AutoTokenizer,AutoModelForCausalLM

# ===== minivllm =====
from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

# ===== vllm =====
from vllm import LLM as VLLM
from vllm import SamplingParams as VLLMSamplingParams



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
    'max_model_length': 512,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # Fixed: should match tokenizer.eos_token_id
}

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "introduce yourself" ,
    "list all prime numbers within 100" ,
    "give me your opinion on the impact of artificial intelligence on society" ,
]

WARMUP_STEPS = 2
OUTPUT_TOKENS = 256  # ouput token num
device = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_minivllm(tokenizer, world_size):
    mini_config = dict(config)
    mini_config["world_size"] = world_size
    llm = MiniLLM(config=mini_config)
    sampling = MiniSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
        max_model_length=mini_config["max_model_length"],
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(x) for x in outputs["token_ids"])
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_vllm(tokenizer, world_size):
    # vLLM
    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False, 
        gpu_memory_utilization=0.75,  
        max_model_len=config["max_model_length"],
        tensor_parallel_size=world_size,
        speculative_config=None, 
    )

    sampling = VLLMSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_transformers_test(tokenizer):
    # transformers
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # Prepare attention_mask explicitly
    attention_mask = inputs["attention_mask"]

    # warmup
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=OUTPUT_TOKENS)

    cuda_sync()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_new_tokens=OUTPUT_TOKENS)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = outputs.shape[0] * (outputs.shape[1] - inputs["input_ids"].shape[1])
    latency = end - start

    tps = total_tokens / latency

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": tps,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end generation throughput benchmark")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--backend",
        choices=["minivllm", "vllm", "transformers"],
        default="minivllm",
    )
    args = parser.parse_args()

    if args.world_size < 1:
        parser.error("--world-size must be at least 1")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')

    print(f"Running {args.backend} benchmark (requested world_size={args.world_size})...")
    if args.backend == "minivllm":
        result = run_minivllm(tokenizer, args.world_size)
        effective_gpus = args.world_size
    elif args.backend == "vllm":
        result = run_vllm(tokenizer, args.world_size)
        effective_gpus = args.world_size
    else:
        # This plain Transformers baseline is intentionally single-GPU; it does
        # not implement tensor parallelism in this script.
        result = run_transformers_test(tokenizer)
        effective_gpus = 1

    results = {args.backend: result}

    print("\n=== Benchmark Results ===")
    print(f"requested_world_size: {args.world_size}")
    print(f"effective_gpus: {effective_gpus}")
    for k, v in results.items():
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv:.4f}")



if __name__ == "__main__":
    main()
