import pickle

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams


def test_prefill_pickle_round_trip_preserves_sequence_state():
    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=32,
        ignore_eos=True,
        max_model_length=128,
    )
    seq = Sequence([1, 2, 3], block_size=4, sampling_params=sampling_params)
    seq.status = SequenceStatus.RUNNING
    seq.num_cached_tokens = 2
    seq.block_table = [7]

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.seq_id == seq.seq_id
    assert restored.block_size == 4
    assert restored.status is SequenceStatus.RUNNING
    assert restored.token_ids == [1, 2, 3]
    assert restored.last_token == 3
    assert restored.num_tokens == 3
    assert restored.num_prompt_tokens == 3
    assert restored.num_cached_tokens == 2
    assert restored.block_table == [7]
    assert restored.temperature == 0.5
    assert restored.max_tokens == 32
    assert restored.ignore_eos is True
    assert restored.max_model_length == 128


def test_decode_pickle_round_trip_keeps_compact_tokens_and_required_state():
    seq = Sequence(
        [1, 2],
        block_size=4,
        sampling_params=SamplingParams(temperature=0.75),
    )
    seq.status = SequenceStatus.RUNNING
    seq.block_table = [5]
    seq.append_token(9)

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.token_ids == [9]
    assert restored.last_token == 9
    assert restored.num_tokens == 3
    assert restored.num_prompt_tokens == 2
    assert restored.block_size == 4
    assert restored.block_table == [5]
    assert restored.status is SequenceStatus.RUNNING
    assert restored.temperature == 0.75
    assert restored.num_blocks == 1
    assert restored.last_block_num_tokens == 3
