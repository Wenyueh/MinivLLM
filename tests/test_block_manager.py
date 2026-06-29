from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def test_allocate_sets_ref_count_and_deallocate_releases_block():
    manager = BlockManager(num_blocks=2, block_size=4)
    seq = Sequence([1, 2, 3], block_size=4)

    manager.allocate(seq)

    assert seq.block_table == [0]
    assert manager.blocks[0].ref_count == 1
    assert manager.used_block_ids == {0}
    assert list(manager.free_block_ids) == [1]

    manager.deallocate(seq)

    assert seq.block_table == []
    assert manager.blocks[0].ref_count == 0
    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [1, 0]


def test_prefix_cache_hit_increments_and_decrements_ref_count():
    manager = BlockManager(num_blocks=2, block_size=4)
    seq1 = Sequence([1, 2, 3, 4], block_size=4)
    seq2 = Sequence([1, 2, 3, 4], block_size=4)

    manager.allocate(seq1)
    manager.allocate(seq2)

    assert seq1.block_table == [0]
    assert seq2.block_table == [0]
    assert manager.blocks[0].ref_count == 2
    assert manager.used_block_ids == {0}
    assert list(manager.free_block_ids) == [1]

    manager.deallocate(seq1)

    assert manager.blocks[0].ref_count == 1
    assert manager.used_block_ids == {0}
    assert list(manager.free_block_ids) == [1]

    manager.deallocate(seq2)

    assert manager.blocks[0].ref_count == 0
    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [1, 0]
