from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence


def test_can_append_allows_filling_existing_block_without_free_block():
    manager = BlockManager(num_blocks=1, block_size=4)
    seq = Sequence([1, 2, 3], block_size=4)
    manager.allocate(seq)
    seq.append_token(4)

    assert list(manager.free_block_ids) == []
    assert manager.can_append(seq)

    manager.append(seq)

    assert seq.block_table == [0]
    assert list(manager.free_block_ids) == []


def test_can_append_rejects_new_block_when_no_free_block_exists():
    manager = BlockManager(num_blocks=1, block_size=4)
    seq = Sequence([1, 2, 3, 4], block_size=4)
    manager.allocate(seq)
    seq.append_token(5)

    assert list(manager.free_block_ids) == []
    assert not manager.can_append(seq)
