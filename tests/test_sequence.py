"""CPU-only regression tests for Sequence bookkeeping.

`Sequence` objects cross the process boundary to the model runners through
pickle-backed shared memory. Anything the worker reads has to survive that
round trip; `model_runner.prepare_prefill` reads `block_size`,
`block_table` and the sampling parameters, so a sequence that loses them
fails with an AttributeError on the worker (issue #89).
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.sampling_parameters import SamplingParams


class TestPickleRoundTrip:
    def _roundtrip(self, seq: Sequence) -> Sequence:
        return pickle.loads(pickle.dumps(seq))

    def test_block_size_survives(self):
        seq = Sequence([1, 2, 3], block_size=4)
        restored = self._roundtrip(seq)
        assert restored.block_size == 4
        # num_blocks and last_block_num_tokens are derived from block_size.
        assert restored.num_blocks == seq.num_blocks == 1
        assert restored.last_block_num_tokens == 3

    def test_sampling_parameters_survive(self):
        params = SamplingParams(temperature=0.7, max_tokens=32, ignore_eos=True)
        seq = Sequence([1, 2, 3], block_size=4, sampling_params=params)
        restored = self._roundtrip(seq)
        assert restored.temperature == 0.7
        assert restored.max_tokens == 32
        assert restored.ignore_eos is True

    def test_status_seq_id_and_block_table_survive(self):
        seq = Sequence([1, 2, 3], block_size=4)
        seq.status = SequenceStatus.RUNNING
        seq.block_table = [5, 6]
        restored = self._roundtrip(seq)
        assert restored.status == SequenceStatus.RUNNING
        assert restored.seq_id == seq.seq_id
        assert restored.block_table == [5, 6]

    def test_decode_state_keeps_only_the_last_token(self):
        seq = Sequence([1, 2, 3], block_size=4)
        seq.append_token(4)
        restored = self._roundtrip(seq)
        assert restored.num_tokens == 4
        # Decode only ships the newest token, as before.
        assert restored.token_ids == [4]
        assert restored.last_token == 4

    def test_attribute_added_later_survives_without_a_protocol_change(self):
        """The old hand-written tuple silently dropped fields added after it.

        Routing the whole __dict__ means a new attribute cannot fall out of the
        worker protocol again (this is what made #89 possible).
        """
        seq = Sequence([1, 2, 3], block_size=4)
        seq.some_field_added_later = "value"
        restored = self._roundtrip(seq)
        assert restored.some_field_added_later == "value"
