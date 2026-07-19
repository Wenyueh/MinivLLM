import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from myvllm.engine.llm_engine import LLMEngine
from myvllm.engine.block_manager import BlockManager
from myvllm.engine.sequence import Sequence
from myvllm.models.qwen3 import get_qwen_positions
from myvllm.sampling_parameters import SamplingParams


def test_sequence_pickle_restores_block_metadata():
    sequence = Sequence([1, 2, 3], block_size=2)
    restored = pickle.loads(pickle.dumps(sequence))

    assert restored.block_size == 2
    assert restored.num_blocks == 2
    assert restored.last_block_num_tokens == 1


def test_duplicate_prefix_hash_keeps_all_valid_block_ids():
    manager = BlockManager(num_blocks=2, block_size=2)
    tokens = [1, 2]
    hash_value = manager.compute_hash(tokens, -1)
    first = manager._allocate_new_block(0, hash_value, tokens)
    manager._register_hash(hash_value, first.block_id)
    second = manager._allocate_new_block(1, hash_value, tokens)
    manager._register_hash(hash_value, second.block_id)

    manager._unregister_hash(hash_value, second.block_id)

    assert manager.hash_to_block_ids[hash_value] == {first.block_id}


def test_prefix_positions_start_after_cached_context():
    context = SimpleNamespace(
        is_prefill=True,
        cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 7, 13], dtype=torch.int32),
        context_lens=None,
    )

    positions = get_qwen_positions(context, torch.device("cpu"), 5)

    assert positions.tolist() == [4, 5, 6, 4, 5]


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_sampling_params_reject_non_positive_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=max_tokens)


def test_step_empty_batch_has_consistent_return_shape():
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(schedule=lambda: ([], False))

    assert engine.step() == ([], 0, False)


def test_empty_prompt_is_rejected():
    engine = object.__new__(LLMEngine)
    engine.tokenizer = SimpleNamespace(encode=lambda prompt: [])
    engine.config = {"block_size": 4}
    engine.scheduler = MagicMock()

    with pytest.raises(ValueError, match="at least one token"):
        engine.add_prompt("", SamplingParams())


def test_exit_is_idempotent():
    engine = object.__new__(LLMEngine)
    runner = MagicMock()
    engine.model_runner = runner
    engine.processes = []

    engine.exit()
    engine.exit()

    runner.call.assert_called_once_with("exit")


def test_generate_skips_special_tokens_when_decoding():
    engine = object.__new__(LLMEngine)
    engine.config = {"block_size": 4}
    engine.tokenizer = MagicMock()
    engine.tokenizer.encode.return_value = [1]
    engine.tokenizer.decode.return_value = "answer"
    engine.scheduler = MagicMock()
    engine.scheduler.is_finished.side_effect = [False, True]
    engine.step = MagicMock(return_value=([(0, [10, 11])], 1, False))

    output = engine.generate(["prompt"], SamplingParams())

    assert output["text"] == ["answer"]
    engine.tokenizer.decode.assert_called_once_with(
        [10, 11], skip_special_tokens=True
    )
