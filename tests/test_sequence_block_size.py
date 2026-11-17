import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


def make_scheduler():
    return Scheduler(
        max_num_sequences=2,
        max_num_batched_tokens=16,
        max_cached_blocks=4,
        block_size=4,
        eos=-1,
    )


@pytest.mark.parametrize("block_size", [2, 8])
def test_scheduler_rejects_sequence_with_different_block_size(block_size):
    scheduler = make_scheduler()
    valid = Sequence([4, 5, 6], block_size=4)
    scheduler.add_sequence(valid)
    seq = Sequence([1, 2, 3, 4, 5], block_size=block_size)

    with pytest.raises(ValueError, match="block_size"):
        scheduler.add_sequence(seq)

    assert list(scheduler.waiting) == [valid]
    assert not scheduler.running
    assert len(scheduler.block_manager.free_block_ids) == 4


def test_scheduler_accepts_sequence_with_matching_block_size():
    scheduler = make_scheduler()
    seq = Sequence([1, 2, 3], block_size=4)

    scheduler.add_sequence(seq)
    batch, is_prefill = scheduler.schedule()

    assert is_prefill
    assert batch == [seq]


def test_matching_sequence_can_decode_and_release_its_cache():
    scheduler = make_scheduler()
    seq = Sequence([1, 2, 3], block_size=4, sampling_params=SamplingParams(max_tokens=2))
    scheduler.add_sequence(seq)

    batch, is_prefill = scheduler.schedule()
    assert is_prefill
    scheduler.postprocess(batch, [9])
    batch, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert batch == [seq]
    scheduler.postprocess(batch, [10])

    assert seq.completion_token_ids == [9, 10]
    assert seq.is_finished
    assert scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids
    assert len(scheduler.block_manager.free_block_ids) == 4
