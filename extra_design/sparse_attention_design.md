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
| `sequence.py` | `dead_ranges`, `mark_dead()` |
| `block_manager.py` | `evict_dead_blocks()`, `_is_range_fully_dead()` |
| `context.py` | `dead_mask` field in Context dataclass |
| `model_runner.py` | Build dead_mask, update graph vars |
| `attention.py` | Kernel takes `dead_mask_ptr`, `max_context_len`; skips dead tokens |
