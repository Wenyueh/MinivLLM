import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.scheduler import Scheduler


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=8,
    max_cached_blocks=64,
    block_size=4,
):
    return Scheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
    )


def ref_counts(block_manager):
    return {b.block_id: b.ref_count for b in block_manager.blocks if b.ref_count > 0}


class TestLivePrefixSharing:
    """
    Two sequences with an identical, full-block-aligned prefix scheduled in the
    same prefill batch. BlockManager.allocate() finds the first sequence's live
    blocks via hash_to_block_id, shares them (ref_count += 1) and advances the
    second sequence's num_cached_tokens -- but the prefill attention kernel only
    attends over the K/V computed in that pass and derives RoPE positions from
    cu_seqlens_q, so a consumed cache hit would silently corrupt the second
    sequence's output. Until the paged prefill kernel + position offset exist,
    sharing live blocks must not happen.
    """

    def _two_identical_seqs(self):
        tokens = list(range(100, 112))  # 12 tokens / block_size 4 = 3 full blocks
        return Sequence(tokens, block_size=4), Sequence(tokens, block_size=4)

    def test_no_live_sharing_between_duplicate_prompts(self):
        scheduler = make_scheduler()
        seq_a, seq_b = self._two_identical_seqs()
        scheduler.add_sequence(seq_a)
        scheduler.add_sequence(seq_b)

        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert seq_a in scheduled and seq_b in scheduled
        assert seq_a.num_cached_tokens == 0, (
            "seq_a is allocated first and must never consume the prefix cache"
        )
        assert seq_b.num_cached_tokens == 0, (
            "seq_b shared live blocks with seq_a and advanced num_cached_tokens; "
            "the prefill kernel cannot consume cached K/V, so this silently "
            "corrupts seq_b's attention output"
        )
        assert not set(seq_a.block_table) & set(seq_b.block_table), (
            "duplicate prompts must not share physical blocks until the prefill "
            "path can read K/V from the cache"
        )
        assert all(count == 1 for count in ref_counts(scheduler.block_manager).values())

    def test_finished_sequence_blocks_are_not_reused(self):
        # Freed blocks keep their slot but not their content: a later identical
        # prompt must allocate fresh blocks, not revive freed ones.
        scheduler = make_scheduler()
        tokens = list(range(100, 112))
        seq_a = Sequence(tokens, block_size=4)
        scheduler.add_sequence(seq_a)
        scheduler.schedule()
        scheduler.block_manager.deallocate(seq_a)
        seq_a.status = SequenceStatus.FINISHED

        seq_b = Sequence(tokens, block_size=4)
        scheduler.add_sequence(seq_b)
        scheduled, is_prefill = scheduler.schedule()

        assert is_prefill
        assert seq_b.num_cached_tokens == 0
        assert not set(seq_a.block_table) & set(seq_b.block_table)

    def test_oversized_prompt_rejected_up_front(self):
        scheduler = make_scheduler(max_cached_blocks=2)
        seq = Sequence(list(range(100, 116)), block_size=4)  # 16 tokens / 4 = 4 blocks > 2
        with pytest.raises(ValueError):
            scheduler.add_sequence(seq)


class TestBlockAccounting:
    def test_ref_count_returns_to_zero_after_generation(self):
        scheduler = make_scheduler()
        tokens = list(range(100, 112))
        seq = Sequence(tokens, block_size=4)
        scheduler.add_sequence(seq)
        scheduler.schedule()

        # simulate the decode phase appending enough tokens to cross block boundaries
        for _ in range(8):
            assert scheduler.block_manager.can_append(seq)
            scheduler.block_manager.append(seq)
            seq.append_token(1)

        scheduler.block_manager.deallocate(seq)
        assert all(b.ref_count == 0 for b in scheduler.block_manager.blocks)
        assert len(scheduler.block_manager.free_block_ids) == len(scheduler.block_manager.blocks)
