# Sparse Attention & Block Eviction Design

## Overview

Enable mid-sequence token eviction to reduce GPU memory during long generations.

## Key Constraints

- Blocks are atomic (256 tokens) - can only free entire blocks
- RoPE encodes relative distance - no update needed after eviction
- Prefix caching uses `ref_count` for shared blocks

## Components

### 1. Sequence: Track Dead Ranges

```python
dead_ranges: list[tuple[int, int]] = []
def mark_dead(self, start: int, end: int):
    self.dead_ranges.append((start, end))
```

### 2. Attention Kernel: Skip Dead Tokens

The Triton decode kernel now takes `dead_mask` and `max_context_len`:

```python
# For each token position:
is_dead = tl.load(dead_mask_ptr + batch_idx * max_context_len + token_idx)
if is_dead == 0:  # Only process alive tokens
    # load K/V, compute attention score
```

Dead tokens get score `-1e10` (masked out in softmax).

### 3. Block Manager: Evict Fully-Dead Blocks

```python
def evict_dead_blocks(self, seq) -> int:
    for block_idx, physical_id in enumerate(seq.block_table):
        if _is_range_fully_dead(block_idx * block_size, (block_idx+1) * block_size):
            block.ref_count -= 1
            if block.ref_count == 0:
                deallocate(physical_id)
            seq.block_table[block_idx] = -1
```

**Why mark as `-1` instead of removing from block_table?**

Keeping token positions stable. If we compact `block_table` or `token_ids`, we'd need to update all position-based tracking (`dead_ranges`, `context_lens`, RoPE positions, etc.). Marking as `-1` keeps everything in place - token position N is always at index N.

### 4. Model Runner: Build Dead Mask

```python
dead_mask = torch.zeros(batch_size, max_context_len)
for start, end in seq.dead_ranges:
    dead_mask[batch_idx, start:end] = 1
```

CUDA graph captures `dead_mask` tensor; values copied each step.

## Files Changed

| File | Change |
|------|--------|
| `sequence.py` | `dead_ranges`, `mark_dead()`, `block_size` instance param |
| `block_manager.py` | `evict_dead_blocks()`, `_is_range_fully_dead()`, ref_count fixes |
| `context.py` | `dead_mask` field in Context dataclass |
| `model_runner.py` | Build dead_mask, update graph vars |
| `attention.py` | Kernel takes `dead_mask_ptr`, `max_context_len`; skips dead tokens |
| `scheduler.py` | Special token handling (`summary_end`, `content_start`, `content_end`) |
| `llm_engine.py` | Pass `block_size` to sequences, config for special tokens |

## Usage Guide

### 1. Manual Dead Range Marking

You can manually mark token ranges as dead during generation:

```python
# After generation or during a custom generation loop
seq.mark_dead(start=100, end=200)  # Mark tokens [100, 200) as dead
block_manager.evict_dead_blocks(seq)  # Free fully-dead blocks
```

**Use cases:**
- Evicting system prompts after they've been processed
- Removing intermediate reasoning tokens in multi-step inference
- Freeing attention computation on irrelevant context

### 2. Automatic Dead Range Marking with Special Tokens

Configure special tokens in your config to automatically trigger dead range marking:

```python
config = {
    'block_size': 256,
    # ... other config ...

    # Sparse attention special tokens
    'summary_end': 264,      # Trigger token - when generated, marks content as dead
    'content_start': 374,    # Marks the start of content to be marked dead
    'content_end': 892,      # Marks the end of content to be marked dead
}
```

**How it works:**
1. When `summary_end` token is generated during inference
2. System searches **backwards** through the sequence to find:
   - Closest `content_end` token before `summary_end`
   - Closest `content_start` token before `summary_end`
3. If both found AND `content_start` comes before `content_end` in sequence order:
   - Marks range `[content_start, content_end]` (inclusive) as dead
   - Automatically calls `evict_dead_blocks()` to free GPU memory

**Example sequence:**
```
... [content_start] ... long content ... [content_end] ... summary ... [summary_end] ...
     ↑_______________mark as dead________________↑
```

**Benefits:**
- Automatically evicts long intermediate content after summarization
- Reduces KV cache memory during long conversations
- Works seamlessly during generation without manual intervention

### 3. Important: Block Size Configuration

**CRITICAL:** The `block_size` must match between `Sequence` and `BlockManager`:

```python
# Correct - block_size is consistent
config = {'block_size': 256}
llm = LLMEngine(config=config)  # Passes block_size to sequences

# Wrong - will cause allocation bugs!
# Don't hardcode different block sizes in different components
```

The `Sequence` class now takes `block_size` as an instance parameter (not a hardcoded class variable). This ensures sequences and block manager always use the same block size.

## Testing: sparse_main.py

Run the comprehensive test suite to verify sparse attention functionality:

```bash
python sparse_main.py
```

### Test Suite Overview

**Test 1: Basic Sparse Attention with Dead Token Marking**
- Demonstrates the API for sparse attention
- Generates text with and without sparse attention enabled
- Note: Current implementation doesn't mark dead ranges automatically in this test

**Test 2: Sequence Dead Range Marking & Block Eviction**
- Creates sequences with 800 tokens (4 blocks @ 256 tokens/block)
- Marks ranges as dead: `[0, 256)`, `[256, 512)`, `[512, 600)`, `[600, 768)`
- Verifies block eviction works correctly:
  - Full blocks are freed and block_table entries marked as `-1`
  - Partial blocks are NOT freed until fully dead
  - Free block count increases after eviction
- **Tests the interval merging algorithm**: marks multiple adjacent ranges that together fully cover a block

**Test 3: Dead Mask Construction**
- Creates sequences with dead ranges
- Simulates dead_mask tensor construction (as done in `model_runner.prepare_decode()`)
- Verifies dead_mask correctly marks dead token positions as 1, alive as 0
- Shows memory savings percentages

**Test 4: Block Eviction with Prefix Caching**
- Creates two sequences sharing a common prefix (first 256 tokens)
- Verifies prefix block has `ref_count = 2` (shared by both sequences)
- Marks prefix as dead in one sequence - block NOT freed (still referenced)
- Marks prefix as dead in both - block IS freed (ref_count reaches 0)
- **Tests reference counting correctness** after recent bugfix

**Test 5: Special Tokens for Automatic Sparse Attention**
- Configures scheduler with special tokens: `summary_end=264`, `content_start=374`, `content_end=892`
- Creates test sequence with multiple occurrences of special tokens
- Simulates generating `summary_end` token
- Verifies:
  - Correct range identified (closest `content_start` to closest `content_end`)
  - Range only marked if `content_start` comes before `content_end`
  - Blocks evicted if fully covered by dead range
- **Uses block_size=16** to test smaller block granularity

### Expected Output

Each test prints detailed state at every step:
- Block allocation counts
- Dead range positions
- Block table contents (physical block IDs, `-1` for freed blocks)
- Free block counts before/after eviction
- Verification results (✓ or ✗)