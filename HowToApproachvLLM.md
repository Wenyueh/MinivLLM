<h1 align="center">vLLM Implementation Roadmap</h1>
<p align="center">
| <a href="./HowToApproachvLLM.md"><b>English</b></a> 
| <a href="./HowToApproachvLLM_zh.md"><b>简体中文</b></a> |
</p>

This document provides a step-by-step guide to understanding and replicating vLLM. Follow this order for the best learning experience.

This package is developed using one A6000 GPU.

[Video link](https://www.bilibili.com/video/BV1Vjz1B2EQu)

---

## Step 1: Layers

Build the fundamental neural network layers first. These are the building blocks for the model.

### 1.1 Activation Function ✅

Path: [activation.py](src/myvllm/layers/activation.py)

Start with activation functions (e.g., SiLU, GELU).

**Key Learning: torch.compile optimization**
- Benchmark with this pattern:
	```python
	for _ in range(10): # Warm-up iterations
		_ = layer(input_tensor)
	
	times = []
	for _ in range(100): # Timing iterations
		torch.cuda.synchronize()
		start_time = time.time()
		output_tensor = layer(input_tensor)
		torch.cuda.synchronize()
		end_time = time.time()
		times.append(end_time - start_time)
	```

**Benchmark Results:**
| tensor shape         | torch.compile | time (ms) |
| ---------------      | ------------- | --------- |
| (400, 800)           | on            |  0.2044   |
| (400, 800)           | off           |  0.0823   |
| (4000, 8000)         | on            |  0.4494   |
| (4000, 8000)         | off           |  0.5290   |
| (8, 4000, 8000)      | on            |  2.3865   |
| (8, 4000, 8000)      | off           |  3.7650   |

**Takeaway:** torch.compile helps for larger computation, but adds overhead for small ops.

---

### 1.2 RMS LayerNorm ✅

Path: [layernorm.py](src/myvllm/layers/layernorm.py)

Implement RMS normalization for stable training.

**Key Concepts:**
- Normalizes activations without mean centering (only uses RMS)
- More efficient than LayerNorm for large models
- Critical for training stability
- Benchmark with this pattern:
	```python
    for _ in range(10): # Warm-up iterations
        _ = layer(x)
    
    # Without residuals
    times = [] 
    for _ in range(100): # Timing iterations
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
	```

**Benchmark Results:**
| tensor shape    | torch.compile | residuals | time (ms) |
| --------------- | ------------- | --------- | --------: |
| (400, 800)      | off           | off       |  0.1630   |
| (400, 800)      | off           | on        |  0.1703   |
| (400, 800)      | on            | off       |  0.2024   |
| (400, 800)      | on            | on        |  0.3470   |
| (4000, 8000)    | off           | off       |  1.3725   |
| (4000, 8000)    | off           | on        |  1.9269   |
| (4000, 8000)    | on            | off       |  0.6029   |
| (4000, 8000)    | on            | on        |  1.1786   |
| (8, 4000, 8000) | off           | off       | 10.4689   |
| (8, 4000, 8000) | off           | on        | 15.3257   |
| (8, 4000, 8000) | on            | off       |  3.6483   |
| (8, 4000, 8000) | on            | on        |  8.1566   |

**Takeaway:** Similar to activation function benchmarking, torch.compile helps for larger computation, but adds overhead for small ops.

---

### 1.3 Linear Layers (with Tensor Parallelism) ✅

Path: [linear.py](src/myvllm/layers/linear.py)

This is the most complex layer due to distributed training support.

**Core Concept: Weight Loading in Distributed Models**
```python
# When loading checkpoints into sharded models:
for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # Check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # Call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. Extract the shard corresponding to the current GPU
            # 2. Copy it to param.data
        else:
            # Default: copy directly
            param.data.copy_(loaded_weight)
```

**Types of Parallel Linear Layers:**

1. **ColumnParallelLinear** ✅
   - Splits output dimension across GPUs
   - Each GPU computes part of the output features
   - No communication needed during forward pass

2. **RowParallelLinear** ✅
   - Splits input dimension across GPUs
   - Requires `dist.all_reduce` to sum partial results
   - Used after ColumnParallel layers

3. **MergedColumnParallelLinear** ✅
   - Merges multiple column-parallel layers (e.g., gate + up projections)
   - Must shard both `param_data` and `loaded_weight` to match correct matrices
   - Efficient for MLP layers

4. **QKVColumnParallel** ✅
   - Special case for attention Q, K, V projections
   - Each GPU stores complete heads (don't shard head_size)
   - Enables independent attention computation per GPU

**MLP Layer Pattern:**
- One ColumnParallel → One RowParallel → `dist.all_reduce`
- Output sharding of first layer = Input sharding of second layer

---

### 1.4 Vocab Embedding & LM Head ✅

Path: [embedding_head.py](src/myvllm/layers/embedding_head.py)


**Vocab Embedding:**
- Partition vocabulary across GPUs
- Each GPU only stores part of the vocabulary

**LM Head:**
- Can share weights with vocab embedding (tied embeddings)
- `F.linear` automatically transposes weights
- Use `dist.gather` or `dist.all_gather` for final logits

**Key Differences:**
- `dist.gather(tensor, gather_list, dst)`: Only dst GPU receives all data
- `dist.all_gather(tensor_list, tensor)`: All GPUs receive all data (no dst parameter)

**Memory Layout - contiguous():**
```python
# Continuous memory
x = [1, 2, 3, 4, 5, 6]  # physically: [1][2][3][4][5][6]

# Non-continuous memory
y = x.reshape(2, 3).T  # logically: [[1,4],[2,5],[3,6]]
                        # physically: [1][2][3][4][5][6] ← old order!
                        # Uses stride() to access elements
```
- `contiguous()` keeps memory blocks adjacent → faster access, no stride needed

---

### 1.5 Attention Layer

Path: [attention.py](src/myvllm/layers/attention.py)

Implement attention mechanism (preferably FlashAttention).

**Key Tensor Concepts:**
- **stride()**: When a tensor is stored in memory, it's a contiguous 1D array. Stride tells you how many elements to skip to move to the next element along a dimension.
	```
	Memory layout: [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23]
	                  ↑                    ↑                   ↑
	             row 0                  row 1               row 2
	```
- **numel()**: Total number of parameters

**GPU Architecture (A100):**
- Each 3D grid has 4 WARPs
- Each WARP has 32 threads
- Each grid processes 128 threads simultaneously

**Triton Kernel Note:**
- When passing PyTorch tensor to Triton kernel, **Triton automatically extracts the pointer** (memory address) from the tensor

---

### 1.6 Rotary Embedding (RoPE) ✅

Path: [rotary_embedding.py](src/myvllm/layers/rotary_embedding.py)

Implement rotary position embeddings for position-aware attention.

**Understanding Base Parameter:**

1. **Large base → Low frequencies:**
   - Unique encoding for distant positions
   - Less local smoothness
   - Cannot distinguish nearby positions well

2. **Small base → High frequencies:**
   - Periodic collision at distant positions
   - Better for short sequences

3. **Different dimensions have different collision positions:**
	```
	Dim 0 (freq=1.0):   Good for positions 0-10 (then repeats) 
	Dim 2 (freq=0.1):   Good for positions 0-100 (then repeats) 
	Dim 4 (freq=0.01):  Good for positions 0-1000 (then repeats) 
	Dim 6 (freq=0.001): Good for positions 0-10000 (then repeats)
	```

**Long Context Strategies (when inference context exceeds training length):**

1. Directly use RoPE (may degrade)
2. Change base: Higher base = lower frequency + better smoothness
3. Scale position: 0, 1, 2, 3 → 0, 0.1, 0.2, 0.3
4. **YARN** ✅
   - High frequency: Model trained on many periods → can extrapolate
   - Low frequency: Model never seen full cycle → compress position to stay in-distribution
5. **NTK** ✅
   - Dynamically increase base for longer contexts

---

## Step 2: Build the Model ✅

Path: [qwen3.py](src/myvllm/models/qwen3.py)

Combine all layers to build the complete Qwen model.

**Key Architecture Decisions:**

**Why `self.num_heads` is per-GPU in Attention?**
- No communication needed during attention computation
- Each GPU works on different heads independently
- Full flow:
  1. Input replicated to all GPUs
  2. QKV projection (ColumnParallel) splits by output dim
  3. `.view()` reshapes local Q, K, V
  4. Run attention on local parameters
  5. Apply RMS and rotary embedding locally
  6. Output projection (RowParallel) uses `dist.all_reduce` to sum

**Why RMS only applied to Q and K?**
- Q and K participate in attention weight computation
- Removes large values that cause instability in softmax
- V doesn't need normalization (doesn't affect score computation)

**Why MergedColumnParallelLinear for gate_up?**
- Compatibility with model checkpoints!
- Checkpoint structure:
	```python
	checkpoint = {
	    'mlp.gate_proj.weight': torch.randn(intermediate_size, hidden_size),
	    'mlp.up_proj.weight': torch.randn(intermediate_size, hidden_size),
	    'mlp.down_proj.weight': torch.randn(hidden_size, intermediate_size),
	}
	```
- Can't just use `intermediate_size * 2` with regular ColumnParallel

**Residual Connections:**
- Always add residual after attention output's layernorm
- Always add residual after final layer's normalization

**Verify it runs!** ✅

---

## Step 3: Sequence Management

Now that the model works, implement the scheduling and memory management system.

### 3.1 Sequence Class

Path: [sequence.py](src/myvllm/engine/sequence.py)

**Purpose:** Store all information about a sequence (prompt + generated tokens).

**Key Implementation Details:**

```python
# In __init__:
self.token_ids = copy(token_ids)  # MUST copy! Creates new list
```

**Why copy?** Without `copy()`, `self.token_ids` would reference the external list, getting affected by outside changes. Copying ensures independence.

**Sequence Status Tracking:**
- Waiting
- Running  
- Finished

**Important Attributes:**
- `token_ids`: All tokens (prompt + generated)
- `num_tokens`: Current length
- `block_table`: Which memory blocks store this sequence's KV cache
- `status`: Current state in the system

---

### 3.2 Block Class

Path: [block_manager.py](src/myvllm/engine/block_manager.py)


**Purpose:** Represent a fixed-size memory block for storing KV cache.

**Key Concepts:**

**Reference Counting (`ref_count`):**
- Tracks how many sequences are using this block
- Critical for **prefix caching** - reusing KV cache when sequences share prefixes
- When deallocating a sequence, check `ref_count` to decide if block should be cleared

**Why Hashing?**
- Purpose: Enable **prefix caching** by looking up blocks by content
- Without hashing: Cannot find if tokens `[1,2,3,...,256]` have been cached
- With hashing: `hash_value = compute_hash([1,2,3,...,256])` → `block_id = hash_to_block_id.get(hash_value)`
- Hash only computed when block is full (all 256 tokens present)

**Why include prefix in hash function argument?**
- Ensures uniqueness across different contexts even if current block tokens are the same
- Example: `[prefix_hash_1][1,2,3]` vs `[prefix_hash_2][1,2,3]` are different

**Why `ref_count = 1` in reset()?**
- When a block is allocated (`_allocate_block` calls `reset()`), it's immediately used by one sequence
- Starting at 1 (not 0) reflects this immediate usage

**Cache Miss Detection:**
```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```
**Why both conditions?**
- `block_id == -1`: Hash not found
- `token_ids != ...`: Avoid hash collision! Different tokens might hash to same value

---

### 3.3 BlockManager Class

Path: [block_manager.py](src/myvllm/engine/block_manager.py)

**Purpose:** Manage KV cache memory allocation/deallocation across all sequences.

**Key Methods:**

**`can_append(seq)`:**
- Check if there's available block/room on GPU to append another token to the sequence
- Returns True/False

**`append()`:**
- Actually allocates new blocks when needed
- Called after `can_append()` returns True
- Manages block table updates

**`allocate_with_cache(seq)`:**
- Try to reuse cached blocks (prefix caching)
- Allocate new blocks only for uncached tokens

**`deallocate(seq)`:**
- Decrease `ref_count` for all blocks in sequence
- Free blocks when `ref_count` reaches 0

---

## Step 4: Model Runner ✅

Path: [model_runner.py](src/myvllm/engine/model_runner.py)

**Purpose:** Bridge between sequences and model execution. Handles data preparation, CUDA graph optimization, and sampling.

### 4.1 Load Weights

Weights can be loaded on the CPU or GPU, but loading weights on different devices may cause weight issues. You can refer to [Issues #36](https://github.com/Wenyueh/MinivLLM/issues/36) for details.

```python
# Load weights in GPU (model moved to GPU before loading weights)
self.model = self.model.cuda(rank)

# Load pretrained weights if model_name_or_path is provided
if config.get('model_name_or_path'):
    from myvllm.utils.loader import load_weights_from_checkpoint
    load_weights_from_checkpoint(self.model, config['model_name_or_path'])

# Load weights in CPU (move the model to GPU after loading weights)
# self.model = self.model.cuda(rank)
```

### 4.2 Core Methods Overview

```python
class ModelRunner:
    def __init__(self): pass
    
    def read_shm(self): pass         # Read from shared memory (worker process)
    def write_shm(self): pass        # Write to shared memory (master process)
    
    def warmup_model(self): pass     # Measure peak memory usage
    def allocate_kv_cache(self): pass # Allocate KV cache memory
    
    def prepare_prefill(self): pass  # Prepare data for prefill forward pass
    def prepare_decode(self): pass   # Prepare data for decode forward pass  
    def prepare_sample(self): pass   # Prepare temperature for sampling
    
    def run_model(self): pass        # Execute model (with CUDA graph for decode)
    def run(self): pass              # Main entry: prepare → run → sample
    
    def capture_cudagraph(self): pass # Capture CUDA graphs for optimization
```

---

### 4.3 Shared Memory Communication

**`read_shm()`:** (Worker process reads from master)

```python
n = int.from_bytes(self.shm.buf[0:4], "little")
```
**Why 4 bytes for length?** Writer always uses 4 bytes: `n.to_bytes(4, "little")` regardless of `n` value.

**Synchronization:**
- `self.event.wait()`: Block until master signals "message ready" by calling `event.set()`
- `self.event.clear()`: Reset signal for next message (back to "not ready" state)

**`write_shm()`:** (Master process writes to workers)

```python
for event in self.events:  # Note: plural, list of events
    event.set()
```
**Why loop?** One event per worker - master signals each worker separately.

**Note on `self.event` vs `self.events`:**
- Master: `self.events = [Event(), Event(), Event()]` (list)
- Worker: `self.event = Event()` (single)

---

### 4.4 Memory Management

**`warmup_model()`:**

**Why run warmup before processing requests?**
- Memory measurement: Run max batch to figure out peak memory usage
- Measures model memory (weights + activations) **without** KV cache
- Uses `torch.cuda.memory_stats()['allocated_bytes.all.peak']`
- Result used in `allocate_kv_cache()` to determine available memory

**`allocate_kv_cache()`:**

**Purpose:** Based on block_size, determine how many KV cache blocks can be allocated.

**Key Design:**
- Reserve memory for peak usage (even when not all in use)
- Reserve for **model**, not per-sequence
- Use `slot_mapping` to track which sequence's which token goes where
- This is the key to **PagedAttention**

---

### 4.5 Data Preparation

**`prepare_prefill(seqs)`:**

**Purpose:** Prepare data for prefill forward pass with prefix caching support.

**Outputs:**
- `input_ids`: All tokens from all sequences in one list
- `positions`: Position indices for each token
- `cu_seqlens_q/k`: Cumulative sequence lengths (boundaries)
- `slot_mapping`: Where to write new KV values
- `block_tables`: Where to read KV values

**Why flatten input_ids into one list?**
- FlashAttention requirement: single kernel launch
- `cu_seqlens_q` marks boundaries: `[0, 3, 5, 9]`
  ```
  │ │ │ │
  │ │ │ └─ end of seq3 (position 9)
  │ │ └──── end of seq2 (position 5)
  │ └─────── end of seq1 (position 3)
  └────────── start (position 0)
  ```

**Why no `cu_seqlens_v`?**
- Same as K (key and value have same sequence structure)

**Why prepare block_tables with matching lengths?**
- Attention kernel needs to read KV cache:
  ```python
  k = kv_cache[..., block_id * block_size : (block_id+1) * block_size, ...]
  ```

**Why `pin_memory=True`?**
- **Pinned memory** = page-locked in physical RAM (cannot swap to disk)
- Enables direct CPU→GPU transfer via DMA (Direct Memory Access)
- Much faster:
  ```
  Normal:    pageable → pinned buffer → GPU (2 copies)
  Pinned:    pinned → GPU (1 copy, DMA)
  ```

**Why `non_blocking=True`?**
- Controls whether CPU waits for transfer completion
- `non_blocking=False`: CPU blocks until GPU has data
- `non_blocking=True`: CPU continues immediately (async transfer)
- Enables parallel transfers!

**Why slot_mapping only includes uncached blocks?**
- Only write KV for **new tokens**, not re-write cached ones
- Cached KV already exists in memory

---

**`prepare_decode(seqs)`:**

**Purpose:** Prepare data for decoding (one token per sequence).

**New slot mapping:**
```python
new_slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
```

**Why no worry about overlapping slots?**
- `append()` in BlockManager guarantees no overlap:
  ```python
  # Seq has 256 tokens (block full)
  seq.num_tokens = 256
  256 % 256 = 0  → Block full, finalize it
  
  # Next token appended → num_tokens = 257
  257 % 256 = 1  → Need new block!
  block = self._allocate_block(self.free_block_ids[0])
  seq.block_table.append(block.block_id)
  ```

---

**`prepare_sample(seqs)`:**

**Purpose:** Prepare temperature values (with padding to match batch size).

---

### 4.6 Model Execution

**`run_model()`:**

**For Prefill:** Directly compute forward pass.

**For Decode:** Use CUDA graph for speed!

```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```
**Why find smallest graph that fits?**
- Captured graphs might not exist for every batch size
- Reuse larger graph with padding

**Why fill `slot_mapping` and `context_lens` with sentinels?**
- Using larger graph than needed → fill unused slots with dummy values

---

**`run()`:**

**Main entry point:**
1. Combine `prepare_prefill` + `run_model` + `sample`
2. Call `reset_context()` to clear cached data

**Why only rank 0 samples?**
- In tensor parallelism, **all ranks compute same logits** (or reduced/gathered to rank 0)
- Only need to **sample once** to get token IDs
- Avoids duplicate/inconsistent sampling

---

### 4.7 CUDA Graph Optimization

**`capture_cudagraph()`:**

**Purpose:** Record CUDA kernel sequence for fast replay (eliminates kernel launch overhead).

**Why only for decoding?**
- Decode has fixed input pattern (1 token per sequence)
- Prefill has variable input lengths

**Capture Strategy:**
- Pre-allocate buffers at maximum sizes
- Capture for common batch sizes: `[1, 2, 4, 8] + list(range(16, max_bs + 1, 16))`
- Capture largest batch first (memory pool sized for largest case)

**Why warmup before capture?**
- CUDA graphs require all memory allocations **before** capture
- Warmup triggers lazy allocations → stable memory during capture

**Why `torch.cuda.synchronize()` before `reset_context()`?**
- Ensure capture completes before resetting for next capture

**`@torch.inference_mode()`:**
- Decorator that disables gradient tracking
- Optimizes inference performance

---

### 4.8 Auxiliary Methods

**`loop()`:**
- Worker process main loop
- Waits for events and calls requested methods

**`call()`:**
- Called by both master and workers
- Master writes to shared memory
- Workers read from shared memory

---

### 4.9 Relationship: torch.compile vs CUDA Graph

**torch.compile:**
- Fuses multiple operations into one kernel
- Saves kernel execution time
- Example:
  ```python
  @torch.compile
  def attention(q, k, v):
      scores = q @ k.T         # ┐
      probs = softmax(scores)  # ├─ Fused into ONE kernel
      output = probs @ v       # ┘
      return output
  ```

**CUDA Graph:**
- Records kernel sequence for replay
- Saves kernel launch overhead (no CPU involvement)
- Captures the execution graph

**Combined:** `torch.compile` reduces kernels, CUDA graph eliminates launch overhead.

---

## Step 5: Scheduler ✅

Path: [scheduler.py](src/myvllm/engine/scheduler.py)

**Purpose:** Decide which sequences to run in each iteration, manage waiting/running queues.

### 5.1 Core Design

**Two Queues:**
1. **Waiting queue**: New sequences not yet started
2. **Running queue**: Currently running sequences

---

### 5.2 Scheduling Logic

**Priority: Prefill > Decode**

The scheduler **always tries prefill first**, even if running queue is not empty!

**Schedule Flow:**
1. **Try to add prefill sequences:**
   - Check if new sequences from waiting queue can fit
   - Stop when no more space for prefill

2. **If no new prefill added, schedule decode:**
   - Continue existing running sequences
   - If no space for more, **preempt** lowest-priority sequence

---

### 5.3 Postprocessing

**After generation:**
- Check if sequences are finished (EOS token or max length)
- If finished: deallocate blocks via BlockManager
- Move completed sequences out of running queue

---

## Step 6: LLM Engine ✅

Path: [llm_engine.py](src/myvllm/engine/llm_engine.py)

**Purpose:** Top-level API orchestrating scheduler, model runner, and request handling.

### 6.1 Core Methods

**`add_request(prompt_str)`:**
- Transform prompt string → Sequence object
- Add to scheduler's waiting queue

**`step()`:**
- Call `scheduler.schedule()` to get sequences to run
- Call `model_runner.run()` to execute them
- Update sequence states

**`generate(prompts)`:**
- Main entry point for inference
- For each prompt:
  1. Add to scheduler
  2. Call `step()` repeatedly until done
  3. Print generation speed stats

---

### 6.2 Initialization Order

**Why does the Scheduler init after the ModelRunner?**

When `world_size > 1`, `ModelRunner.__init__` calls `dist.init_process_group('nccl', ...)`, which is a **collective barrier** — rank-0 blocks until every worker process has also called it. Only once all ranks have joined the process group does `ModelRunner.__init__` return. The Scheduler is created after that, ensuring the distributed environment is fully established before the engine is considered ready.

When `world_size == 1` no worker processes are spawned and there is no barrier, so the ordering has no practical effect in that case.

---

### 6.3 Cleanup

**Why `exit()` and `atexit.register(self.exit)`?**
```python
def exit(self):
    # Cleanup code
    self.workers.join()  # Wait for workers to finish

atexit.register(self.exit)
```

**Purpose:** When Python program stops, automatically:
1. Call `engine.exit()` to clean up resources
2. Wait for worker processes to finish gracefully
3. Prevent zombie processes or corrupted state

---

## Summary: Learning Order

1. **Layers** (activation → layernorm → linear → vocab/lmhead → attention → rotary)
2. **Model** (assemble layers, verify it runs)
3. **Sequence Management** (Sequence → Block → BlockManager)
4. **Model Runner** (data prep, CUDA graphs, sampling)
5. **Scheduler** (queue management, prefill/decode scheduling)
6. **LLM Engine** (top-level orchestration)

Each step builds on the previous, gradually constructing a complete inference system with advanced optimizations like PagedAttention, CUDA graphs, and prefix caching.

## Course Exercise

Interested readers can try adding `meta-llama/Llama-3.2-1B-Instruct` to MinivLLM locally as an exercise.

`meta-llama/Llama-3.2-1B-Instruct` (hereinafter referred to as Llama3.2) has a similar structure to `Qwen/Qwen3-0.6B`. The only slight difference in model components lies in the implementation of Rotary Embedding. Provided that the field names remain the same, the existing weight loading code in `loader.py` can be used directly for Llama3.2 without any modifications.

Reference materials:
- For the implementation of Llama3.2, you can refer to [Llama3.2 in mini-sglang](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/models/llama.py)
- The differences in Rotary Embedding implementation can be found in [Rotary Embedding in mini-sglang](https://github.com/sgl-project/mini-sglang/blob/dae78f6bb97d5c5aaadbc0772fc964d48a8ee726/python/minisgl/layers/rotary.py#L72-L86).
- Various model parameters can be found in the `config.json` file of [Llama3.2 on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main).

To complete the exercise, you can first clone the repository locally, then delete the Llama3.2 implementation in the repository: `rm src/myvllm/models/llama.py`. After that, create your own `src/myvllm/models/llama.py` file. By referring to the Llama3.2 implementation in the provided link, implement Llama3.2 yourself based on MinivLLM.

Adding Llama3.2 only involves modifications to the following files:
- `src/myvllm/models/llama.py`: Model implementation. This requires you to implement it yourself.
- `src/myvllm/layers/rotary_embedding.py`: You need to add the different implementation for Llama3.2.
- `src/myvllm/engine/model_runner.py`: The ModelRunner needs to be able to call the implemented Llama3.2.
- `main_llama32.py`: Responsible for testing the implementation effect of Llama3.2.

Run `main_llama32.py`, and the effect should look like this:

![llama32-effect](assets/llama32-effect.png)

Since the last three files (`rotary_embedding.py`, `model_runner.py`, `main_llama32.py`) require only minor modifications, MinivLLM has already implemented them. All you need to do is delete the `src/myvllm/models/llama.py` file, then repeatedly compare [Llama3.2 in mini-sglang](https://github.com/sgl-project/mini-sglang/blob/main/python/minisgl/models/llama.py) with `src/myvllm/models/qwen3.py`, and implement your own Llama3.2 in `src/myvllm/models/llama.py`. Once implemented, run `uv run main_llama32.py` to test. If your implementation is correct, you should see an effect similar to the one above. If you are truly stuck, refer to the original `src/myvllm/models/llama.py` in the repository in a timely manner.