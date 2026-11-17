import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


@pytest.fixture
def engine(monkeypatch):
    # Admission tests use the real engine and scheduler without loading CUDA kernels.
    backend = ModuleType("myvllm.engine.model_runner")
    backend.ModelRunner = object
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    monkeypatch.delitem(sys.modules, "myvllm.engine.llm_engine", raising=False)
    from myvllm.engine.llm_engine import LLMEngine

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = {"block_size": 4}
    engine.scheduler = Scheduler(4, 16, 2, 4, eos=-1)
    engine.tokenizer = SimpleNamespace(
        encode=lambda text: list(range(1, len(text) + 1)),
        decode=lambda tokens: " ".join(map(str, tokens)),
    )
    engine.model_runner = SimpleNamespace(
        call=lambda method, seqs, prefill: torch.full((len(seqs),), 42),
    )
    yield engine
    sys.modules.pop("myvllm.engine.llm_engine", None)


def test_failed_batch_does_not_leak_requests_into_next_generation(engine):
    params = SamplingParams(max_tokens=1)
    with pytest.raises(ValueError, match="KV cache"):
        engine.generate(["ok", "x" * 9], params)

    result = engine.generate(["new"], params)

    assert result["token_ids"] == [[42]]
    assert result["text"] == ["42"]
    assert engine.scheduler.is_finished()


def test_failed_admission_preserves_existing_waiting_request(engine):
    existing = Sequence([100], block_size=4)
    engine.scheduler.add_sequence(existing)

    with pytest.raises(ValueError, match="KV cache"):
        engine.generate(["ok", "x" * 9], SamplingParams())

    assert list(engine.scheduler.waiting) == [existing]
    assert not engine.scheduler.running
    assert len(engine.scheduler.block_manager.free_block_ids) == 2


def test_tokenizer_failure_rolls_back_only_the_current_batch(engine):
    original_encode = engine.tokenizer.encode
    error = ValueError("tokenization failed")

    def encode(text):
        if text == "bad":
            raise error
        return original_encode(text)

    engine.tokenizer.encode = encode
    with pytest.raises(ValueError) as caught:
        engine.generate(["ok", "bad"], SamplingParams())

    assert caught.value is error
    assert engine.scheduler.is_finished()


def test_successful_batch_returns_one_result_per_prompt(engine):
    result = engine.generate(["ok", "new"], SamplingParams(max_tokens=1))

    assert result["token_ids"] == [[42], [42]]
    assert engine.scheduler.is_finished()
