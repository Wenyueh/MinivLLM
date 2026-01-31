"""
Test script for sparse attention implementation.

This script demonstrates and tests:
1. Marking token ranges as dead using seq.mark_dead()
2. Evicting fully-dead blocks using block_manager.evict_dead_blocks()
3. Running inference with sparse attention (dead_mask in kernel)
"""

import sys, os
from pathlib import Path
import torch
import torch.distributed as dist

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoTokenizer
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams
from myvllm.engine.sequence import Sequence

config = {
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,
    'vocab_size': 151936,
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768,
    'ffn_bias': False,
    'max_num_batch_tokens': 4096,
    'max_model_length': 512,  # Increased for sparse attention testing
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
    # Sparse attention special tokens
    'summary_end': 264,      # When this token is generated, trigger dead range marking
    'content_start': 374,    # Marks the start of content to be marked dead
    'content_end': 892,      # Marks the end of content to be marked dead
}


def test_sparse_attention_basic():
    """Test 1: Basic sparse attention with dead token marking"""
    print("\n" + "="*80)
    print("TEST 1: Basic Sparse Attention with Dead Token Marking")
    print("="*80)

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)

    llm = LLM(config=config)

    # Create prompts
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128, max_model_length=512)

    # Generate normally first
    print("\n[1a] Generating without sparse attention...")
    outputs_normal = llm.generate(prompts, sampling_params)

    for i, (prompt, output) in enumerate(zip(prompts, outputs_normal['text'])):
        print(f"\n--- Normal Output {i+1} ---")
        print(f"Prompt: {prompt[:50]}...")
        print(f"Completion: {output[:100]}...")

    # Now test with sparse attention - manually mark dead ranges
    print("\n[1b] Generating with sparse attention (manual dead range marking)...")
    print("NOTE: This demonstrates the API, but engine doesn't expose sequences directly.")
    print("See test_sequence_dead_marking() for direct sequence testing.")

    outputs_sparse = llm.generate(prompts, sampling_params)

    for i, (prompt, output) in enumerate(zip(prompts, outputs_sparse['text'])):
        print(f"\n--- Sparse Attention Output {i+1} ---")
        print(f"Prompt: {prompt[:50]}...")
        print(f"Completion: {output[:100]}...")

    print("\n[Test 1 Complete]")


def test_sequence_dead_marking():
    """Test 2: Direct sequence dead range marking and block eviction"""
    print("\n" + "="*80)
    print("TEST 2: Sequence Dead Range Marking & Block Eviction")
    print("="*80)

    from myvllm.engine.block_manager import BlockManager

    # Create a block manager
    block_size = 256
    num_blocks = 100
    block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)

    # Create a sequence with many tokens
    token_ids = list(range(800))  # 800 tokens = ~4 blocks (256 tokens each)
    seq = Sequence(token_ids=token_ids, block_size=block_size)

    print(f"\n[2a] Created sequence with {len(seq)} tokens")
    print(f"     Number of blocks needed: {seq.num_blocks}")
    print(f"     Available blocks: {len(block_manager.free_block_ids)}")

    # Allocate blocks for the sequence
    if block_manager.can_allocate(seq):
        block_manager.allocate(seq)
        print(f"\n[2b] Allocated {len(seq.block_table)} blocks for sequence")
        print(f"     Block table: {seq.block_table}")
        print(f"     Remaining free blocks: {len(block_manager.free_block_ids)}")
    else:
        print("\n[ERROR] Cannot allocate blocks!")
        return

    # Mark first block (tokens 0-255) as dead
    print(f"\n[2c] Marking tokens [0, 256) as dead (first block)...")
    seq.mark_dead(0, 256)
    print(f"     Dead ranges: {seq.dead_ranges}")

    # Evict dead blocks
    blocks_freed = block_manager.evict_dead_blocks(seq)
    print(f"\n[2d] Evicted {blocks_freed} blocks")
    print(f"     Updated block table: {seq.block_table}")
    print(f"     Free blocks now: {len(block_manager.free_block_ids)}")

    # Mark second block (tokens 256-511) as dead
    print(f"\n[2e] Marking tokens [256, 512) as dead (second block)...")
    seq.mark_dead(256, 512)
    print(f"     Dead ranges: {seq.dead_ranges}")

    # Evict more dead blocks
    blocks_freed = block_manager.evict_dead_blocks(seq)
    print(f"\n[2f] Evicted {blocks_freed} more blocks")
    print(f"     Updated block table: {seq.block_table}")
    print(f"     Free blocks now: {len(block_manager.free_block_ids)}")

    # Mark partial range (not a full block)
    print(f"\n[2g] Marking tokens [512, 600) as dead (partial block)...")
    seq.mark_dead(512, 600)
    print(f"     Dead ranges: {seq.dead_ranges}")

    # Try to evict - should evict 0 blocks since it's not fully dead
    blocks_freed = block_manager.evict_dead_blocks(seq)
    print(f"\n[2h] Evicted {blocks_freed} blocks (expected 0 - partial block)")
    print(f"     Block table unchanged: {seq.block_table}")

    # Mark the rest of the third block to make it fully dead
    print(f"\n[2i] Marking tokens [600, 768) as dead (completing third block)...")
    seq.mark_dead(600, 768)
    print(f"     Dead ranges: {seq.dead_ranges}")

    # Now evict should work
    blocks_freed = block_manager.evict_dead_blocks(seq)
    print(f"\n[2j] Evicted {blocks_freed} blocks (expected 1 - now fully dead)")
    print(f"     Updated block table: {seq.block_table}")
    print(f"     Free blocks now: {len(block_manager.free_block_ids)}")

    print("\n[Test 2 Complete]")


def test_dead_mask_construction():
    """Test 3: Dead mask construction for decode phase"""
    print("\n" + "="*80)
    print("TEST 3: Dead Mask Construction")
    print("="*80)

    from myvllm.engine.model_runner import ModelRunner
    from multiprocessing import Event

    # Create sequences with dead ranges
    seqs = []

    # Sequence 1: tokens [0, 100), dead range [20, 40)
    seq1 = Sequence(token_ids=list(range(100)), block_size=256)
    seq1.mark_dead(20, 40)
    seqs.append(seq1)

    # Sequence 2: tokens [0, 150), dead ranges [10, 30) and [80, 100)
    seq2 = Sequence(token_ids=list(range(150)), block_size=256)
    seq2.mark_dead(10, 30)
    seq2.mark_dead(80, 100)
    seqs.append(seq2)

    print(f"\n[3a] Created {len(seqs)} sequences:")
    for i, seq in enumerate(seqs):
        print(f"     Seq {i+1}: {len(seq)} tokens, dead_ranges={seq.dead_ranges}")

    # Simulate dead mask construction (from prepare_decode)
    context_lens = [len(seq) for seq in seqs]
    max_context_len = max(context_lens)
    dead_mask = torch.zeros(len(seqs), max_context_len, dtype=torch.int32)

    for batch_idx, seq in enumerate(seqs):
        for dead_start, dead_end in seq.dead_ranges:
            start = max(0, dead_start)
            end = min(max_context_len, dead_end)
            if start < end:
                dead_mask[batch_idx, start:end] = 1

    print(f"\n[3b] Dead mask shape: {dead_mask.shape}")
    print(f"     Max context length: {max_context_len}")

    # Show which tokens are dead for each sequence
    for i in range(len(seqs)):
        dead_positions = torch.where(dead_mask[i] == 1)[0].tolist()
        print(f"\n[3c] Seq {i+1} dead token positions: {dead_positions[:20]}..." if len(dead_positions) > 20 else f"\n[3c] Seq {i+1} dead token positions: {dead_positions}")
        print(f"     Total dead tokens: {len(dead_positions)} / {context_lens[i]}")
        print(f"     Memory savings: {len(dead_positions) / context_lens[i] * 100:.1f}% marked as dead")

    print("\n[Test 3 Complete]")


def test_block_eviction_with_prefix_cache():
    """Test 4: Block eviction with prefix caching (ref_count handling)"""
    print("\n" + "="*80)
    print("TEST 4: Block Eviction with Prefix Caching")
    print("="*80)

    from myvllm.engine.block_manager import BlockManager

    block_size = 256
    num_blocks = 100
    block_manager = BlockManager(num_blocks=num_blocks, block_size=block_size)

    # Create two sequences with shared prefix (first 256 tokens)
    prefix = list(range(256))
    seq1_tokens = prefix + list(range(256, 400))
    seq2_tokens = prefix + list(range(256, 450))

    seq1 = Sequence(token_ids=seq1_tokens, block_size=block_size)
    seq2 = Sequence(token_ids=seq2_tokens, block_size=block_size)

    print(f"\n[4a] Created 2 sequences with shared prefix:")
    print(f"     Seq1: {len(seq1)} tokens ({seq1.num_blocks} blocks)")
    print(f"     Seq2: {len(seq2)} tokens ({seq2.num_blocks} blocks)")

    # Allocate blocks
    block_manager.allocate(seq1)
    block_manager.allocate(seq2)

    print(f"\n[4b] Allocated blocks:")
    print(f"     Seq1 block_table: {seq1.block_table}")
    print(f"     Seq2 block_table: {seq2.block_table}")
    print(f"     Seq1 cached tokens: {seq1.num_cached_tokens}")
    print(f"     Seq2 cached tokens: {seq2.num_cached_tokens}")

    # Check if prefix block is shared (should have ref_count = 2)
    if seq1.block_table[0] == seq2.block_table[0]:
        shared_block_id = seq1.block_table[0]
        ref_count = block_manager.blocks[shared_block_id].ref_count
        print(f"\n[4c] Prefix block {shared_block_id} is shared!")
        print(f"     Reference count: {ref_count}")
    else:
        print(f"\n[4c] WARNING: Prefix blocks are not shared")

    # Mark the shared prefix as dead in seq1
    print(f"\n[4d] Marking prefix [0, 256) as dead in seq1...")
    seq1.mark_dead(0, 256)

    # Try to evict from seq1
    blocks_freed = block_manager.evict_dead_blocks(seq1)
    print(f"\n[4e] Evicted {blocks_freed} blocks from seq1")
    print(f"     Seq1 block_table: {seq1.block_table}")

    # Check ref_count of shared block
    if seq1.num_blocks > 0 and seq1.block_table[0] != -1:
        # Block wasn't freed because it's still referenced by seq2
        print(f"     Shared block still in use (ref_count should be 1)")
        shared_block_id = seq2.block_table[0]
        ref_count = block_manager.blocks[shared_block_id].ref_count
        print(f"     Reference count: {ref_count}")
    else:
        print(f"     Seq1's first block marked as -1 (evicted)")
        if len(seq2.block_table) > 0:
            shared_block_id = seq2.block_table[0]
            ref_count = block_manager.blocks[shared_block_id].ref_count
            print(f"     Shared block {shared_block_id} ref_count: {ref_count}")

    # Mark the prefix as dead in seq2 as well
    print(f"\n[4f] Marking prefix [0, 256) as dead in seq2...")
    seq2.mark_dead(0, 256)

    # Evict from seq2 - now block should be freed
    blocks_freed = block_manager.evict_dead_blocks(seq2)
    print(f"\n[4g] Evicted {blocks_freed} blocks from seq2")
    print(f"     Seq2 block_table: {seq2.block_table}")
    print(f"     Free blocks now: {len(block_manager.free_block_ids)}")

    print("\n[Test 4 Complete]")


def test_special_tokens_sparse_attention():
    """Test 5: Special tokens triggering automatic dead range marking"""
    print("\n" + "="*80)
    print("TEST 5: Special Tokens (summary_end, content_start, content_end)")
    print("="*80)

    from myvllm.engine.scheduler import Scheduler
    from myvllm.sampling_parameters import SamplingParams

    # Create scheduler with special token config
    scheduler = Scheduler(
        max_num_sequences=16,
        max_num_batched_tokens=1024,
        max_cached_blocks=1024,
        block_size=16,
        eos=151645,
        summary_end=264,
        content_start=374,
        content_end=892
    )

    print(f"\n[5a] Scheduler configured with special tokens:")
    print(f"     summary_end: {scheduler.summary_end}")
    print(f"     content_start: {scheduler.content_start}")
    print(f"     content_end: {scheduler.content_end}")

    # Create a test sequence matching the example:
    # ... 374, 3415, 5634, ..., 892, ..., 264
    test_tokens = [1, 423, 54, 542, 6, 234, 5, 134, 52, 45, 6, 24, 5, 634, 31, 543, 65, 234, 234, 5,
                   7456, 245, 6524, 374, 134, 4531, 1324, 435, 1234, 52, 6, 656, 457, 374, 123, 1324,
                   431, 31, 3, 12, 892, 3214, 1324, 425, 23, 314, 374, 3415, 5634, 432, 432, 6543,
                   3124, 4123, 5423, 5234, 1, 1, 1,1 ,1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1 ,1, 1, 1, 1, 1,1, 1, 1, 1, 654, 45, 892, 1234, 4135, 4235, 564, 6543, 764, 3142]

    seq = Sequence(token_ids=test_tokens, sampling_params=SamplingParams(), block_size=16)
    scheduler.block_manager.allocate(seq)
    seq.status = 1  # RUNNING

    print(f"\n[5b] Created test sequence with {len(seq)} tokens")
    print(f"     Tokens include content_start (374), content_end (892), and will add summary_end (264)")
    print(f"     Allocated {len(seq.block_table)} blocks")

    # Find positions of special tokens for verification
    positions_374 = [i for i, t in enumerate(seq.token_ids) if t == 374]
    positions_892 = [i for i, t in enumerate(seq.token_ids) if t == 892]
    print(f"\n[5c] Special token positions in sequence:")
    print(f"     content_start (374) at positions: {positions_374}")
    print(f"     content_end (892) at positions: {positions_892}")

    # Simulate postprocess with summary_end token (264)
    print(f"\n[5d] Simulating generation of summary_end token (264)...")
    scheduler.running.append(seq)
    scheduler.postprocess([seq], [264])

    print(f"\n[5e] After postprocess:")
    print(f"     Sequence dead_ranges: {seq.dead_ranges}")
    print(f"     Block table: {seq.block_table}")

    # Verify the expected behavior
    if seq.dead_ranges:
        start, end = seq.dead_ranges[0]
        print(f"\n[5f] Verification:")
        print(f"     Marked range: [{start}, {end})")
        print(f"     Marked tokens: {seq.token_ids[start:end]}")
        print(f"     Expected: content_start at {positions_374[-1]} to content_end at {positions_892[-1]}")
        if start == positions_374[-1] and end == positions_892[-1] + 1:
            print(f"     ✓ Correct range marked!")
        else:
            print(f"     ✗ Unexpected range marked")
    else:
        print(f"\n[5f] WARNING: No dead ranges were marked")

    print("\n[Test 5 Complete]")


def main():
    """Run all sparse attention tests"""
    print("\n" + "#"*80)
    print("# SPARSE ATTENTION TEST SUITE")
    print("#"*80)

    try:
        # Test 1: Basic sparse attention with generation
        test_sparse_attention_basic()
    except Exception as e:
        print(f"\n[ERROR in Test 1] {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 2: Sequence dead marking and block eviction
        test_sequence_dead_marking()
    except Exception as e:
        print(f"\n[ERROR in Test 2] {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 3: Dead mask construction
        test_dead_mask_construction()
    except Exception as e:
        print(f"\n[ERROR in Test 3] {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 4: Block eviction with prefix caching
        test_block_eviction_with_prefix_cache()
    except Exception as e:
        print(f"\n[ERROR in Test 4] {e}")
        import traceback
        traceback.print_exc()

    try:
        # Test 5: Special tokens for automatic sparse attention
        test_special_tokens_sparse_attention()
    except Exception as e:
        print(f"\n[ERROR in Test 5] {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#"*80)
    print("# ALL TESTS COMPLETE")
    print("#"*80 + "\n")


if __name__ == "__main__":
    main()
