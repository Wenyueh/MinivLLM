"""CPU-only tests for the KV-cache block accounting.

BlockManager and Sequence.last_block_num_tokens decide which physical slot a
token is written to. When they drift the engine does not crash, it silently
writes wrong output -- and the July fixes in this area (#77, #84, #85) landed
without a single test: the regression tests they came with lived in pull
requests that were closed.

Deliberately not asserted here: anything about prefix-cache reuse (see #98,
which reports that it currently produces wrong output) and any behaviour that
only exists because a bare `assert` or an unguarded list index happens to
raise today.

These tests do not import torch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def make_manager(num_blocks: int = 8, block_size: int = 4) -> BlockManager:
    return BlockManager(num_blocks, block_size)


class TestLastBlockNumTokens:
    def test_exact_multiple_of_block_size(self):
        seq = Sequence(list(range(8)), block_size=4)
        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 4
        assert seq.block(1) == [4, 5, 6, 7]

    def test_partial_last_block(self):
        seq = Sequence(list(range(6)), block_size=4)
        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 2
        assert seq.block(1) == [4, 5]

    def test_single_block(self):
        seq = Sequence([1, 2, 3], block_size=4)
        assert seq.num_blocks == 1
        assert seq.last_block_num_tokens == 3
        assert seq.block(0) == [1, 2, 3]

    def test_empty_sequence_owns_no_block(self):
        seq = Sequence([], block_size=4)
        assert seq.num_blocks == 0
        # token_ids[-0:] returns the whole list, so an empty sequence must
        # report 0 here rather than "-0".
        assert seq.last_block_num_tokens == 0


class TestCanAppend:
    """can_append() must ask for a block exactly when append() allocates one."""

    def test_token_inside_the_current_block_needs_no_free_block(self):
        bm = make_manager()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        seq.append_token(4)  # fills the block the sequence already owns
        bm.free_block_ids.clear()
        assert bm.can_append(seq) is True

    def test_boundary_token_requires_a_free_block(self):
        bm = make_manager()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        seq.append_token(4)
        bm.append(seq)  # finalise the block that just became full
        seq.append_token(5)  # first token of the next block
        assert bm.can_append(seq) is True
        bm.free_block_ids.clear()
        assert bm.can_append(seq) is False

    def test_append_consumes_exactly_one_block(self):
        bm = make_manager()
        seq = Sequence([1, 2, 3], block_size=4)
        bm.allocate(seq)
        seq.append_token(4)
        bm.append(seq)
        free_before = len(bm.free_block_ids)
        seq.append_token(5)
        bm.append(seq)
        assert len(bm.free_block_ids) == free_before - 1
        assert len(seq.block_table) == 2


class TestRefCountLifecycle:
    def test_allocate_deallocate_round_trip(self):
        bm = make_manager()
        free_before = len(bm.free_block_ids)
        seq = Sequence([1, 2, 3, 4], block_size=4)
        bm.allocate(seq)
        assert len(bm.free_block_ids) == free_before - 1
        assert bm.blocks[seq.block_table[0]].ref_count == 1
        bm.deallocate(seq)
        assert len(bm.free_block_ids) == free_before
        assert seq.block_table == []

    def test_ref_count_returns_to_zero_not_minus_one(self):
        bm = make_manager()
        seq = Sequence([1, 2, 3, 4], block_size=4)
        bm.allocate(seq)
        bm.deallocate(seq)
        assert bm.blocks[0].ref_count == 0
        assert 0 in bm.free_block_ids
        assert 0 not in bm.used_block_ids

    def test_deallocate_frees_every_block_of_a_long_sequence(self):
        bm = make_manager(num_blocks=8)
        free_before = len(bm.free_block_ids)
        seq = Sequence(list(range(10)), block_size=4)  # 3 blocks
        bm.allocate(seq)
        assert len(seq.block_table) == 3
        assert len(bm.free_block_ids) == free_before - 3
        bm.deallocate(seq)
        assert len(bm.free_block_ids) == free_before
        assert bm.used_block_ids == set()

    def test_pool_exhaustion_is_reported_by_can_allocate(self):
        """The admission check the scheduler uses, without pinning what an
        unguarded allocate() does today."""
        bm = make_manager(num_blocks=1)
        bm.allocate(Sequence([1, 2, 3, 4], block_size=4))
        other = Sequence([5, 6, 7, 8], block_size=4)
        assert bm.can_allocate(other) is False
        assert len(bm.free_block_ids) == 0


class TestHashing:
    def test_same_tokens_and_prefix_hash_equally(self):
        bm = make_manager()
        assert bm.compute_hash([1, 2, 3, 4], -1) == bm.compute_hash([1, 2, 3, 4], -1)

    def test_different_prefix_hashes_differently(self):
        bm = make_manager()
        assert bm.compute_hash([1, 2, 3, 4], 7) != bm.compute_hash([1, 2, 3, 4], 9)

    def test_partial_block_is_not_cached(self):
        bm = make_manager()
        seq = Sequence([1, 2, 3], block_size=4)  # last block is partial
        bm.allocate(seq)
        assert seq.num_cached_tokens == 0
        assert seq.block_table[0] in bm.used_block_ids
