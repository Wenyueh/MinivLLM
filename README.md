<p align="center">
  <img src="./assets/minivllm.png" alt="MinivLLM" width="50%" height="50%">
</p>

<p align="center">
| <a href="./README.md"><b>English</b></a> 
| <a href="./README_zh.md"><b>简体中文</b></a> |
</p>

# miniVLLM

A custom implementation of vLLM inference engine with attention mechanism benchmarks, based on Nano-vLLM but with self-contained paged attention and flash attention implementation. 

Benchmarking on flash attention in prefilling time and paged attention in decoding time are provided.

**New to vLLM?** Check out [HowToApproachvLLM.md](HowToApproachvLLM.md) for a step-by-step implementation guide covering layers, models, paged attention, CUDA graphs, and scheduling.

## Quickstart

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync

# Run the main inference engine
uv run python main.py

# Run prefilling benchmark
uv run python benchmark_prefilling.py

# Run decoding benchmark
uv run python benchmark_decoding.py

# Run the Llama-3.2 demo
uv run python main_llama32.py

# Compare throughput with vLLM and transformers (needs a GPU)
uv run python benchmark_tps.py
```

To run multi-GPU setting, simply change world_size to n > 1 in config in main.py

## What Each Script Does

```bash
uv run python main.py
```

This is the main inference engine demo

Demonstrates the complete LLM inference pipeline using a custom engine implementation:
- Create a small version of Qwen3 with random initialization
- Creates 60 chat prompts (2 base prompts repeated 30 times each)
- Processes them through the custom LLM engine with batch processing
- Uses paged attention and KV cache management for efficient inference
- Generates up to 256 tokens per prompt with temperature sampling

This showcases how the custom vLLM implementation handles batched text generation with memory-efficient attention.

```bash
uv run python benchmark_prefilling.py
```

This is the prefilling phase comparison

Compares three attention implementations during the **prefilling phase** (processing input prompts):

1. **PyTorch Standard (O(N²) memory)**: Traditional attention that materializes full attention matrix
2. **Naive Triton (O(N²) memory)**: GPU kernel that also uses O(N²) memory, limited by shared memory constraints (≤128 tokens)
3. **Flash Attention (O(N) memory)**: Memory-efficient online softmax algorithm that processes attention in blocks

```bash
uv run python benchmark_decoding.py
```

This is the decoding phase comparison

Compares three implementations during the **decoding phase** (generating output tokens one at a time):

1. **Naive PyTorch**: Simple loop-based implementation using paged KV cache
2. **Optimized PyTorch**: Vectorized implementation with batch gathering and masking
3. **Triton Kernel**: Custom GPU kernel optimized for paged attention decode

```bash
uv run python main_llama32.py
```

The same engine with a different model: it runs the demo above with a
`meta-llama/Llama-3.2-1B-Instruct` configuration (16 layers, GQA with 8 KV heads,
tied embeddings) to show that nothing in the implementation is Qwen-specific.

```bash
uv run python benchmark_tps.py
```

Throughput comparison (tokens/s) between this engine, vLLM and the plain
`transformers` implementation on the same prompts. It needs a CUDA GPU and the
`vllm` package that `uv sync` installs.


## Project Structure

```
myvllm/
├── src/
│   └── myvllm/           # Core vLLM implementation
│       ├── models/       # Model implementations
│       ├── engine/       # LLM engine logic, including sequence definition for input prompts, block management for KV cache management for GPU, scheduler for iteration-based scheduling of sequences, runner for actual implementation of running prefilling and decoding, and engine for generation API interface
│       ├── layers/       # Model layer components (activation, attention, embeddings, etc.)
│       ├── utils/        # Utility helpers and inference context management
│       └── sampling_parameters.py
├── tests/                   # CPU-only unit tests (no GPU needed)
├── main.py                  # Full inference demo (Qwen3-0.6B)
├── main_llama32.py          # Same demo with a Llama-3.2-1B-Instruct config
├── benchmark_prefilling.py  # Prefilling attention comparison
├── benchmark_decoding.py    # Decoding attention comparison
├── benchmark_tps.py         # Throughput comparison vs vLLM / transformers
└── HowToApproachvLLM.md     # Step-by-step implementation guide
```

## Tests

The unit tests are CPU-only: they never import torch or triton, so no GPU is
needed.

```bash
pip install pytest numpy xxhash
python -m pytest tests -q
```

## Requirements

- Python ≥3.11, < 3.12
- CUDA-capable GPU to run any of the inference or benchmark scripts (the unit tests run on CPU)
- Dependencies: `transformers`, `torch`, `xxhash`, `vllm` (managed by uv)


## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=Wenyueh/MinivLLM&type=date&legend=top-left)](https://star-history.dera.page/?utm_source=chatgpt.com#Wenyueh/MinivLLM&type=date&legend=top-left)