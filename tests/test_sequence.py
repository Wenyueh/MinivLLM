import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myvllm.engine.sequence import Sequence


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, 0),
        (1, 1),
        (3, 3),
        (4, 4),
        (5, 1),
        (8, 4),
    ],
)
def test_last_block_num_tokens(num_tokens, expected):
    seq = Sequence(list(range(num_tokens)), block_size=4)

    assert seq.last_block_num_tokens == expected


@pytest.mark.parametrize(
    ("num_tokens", "expected_blocks"),
    [
        (3, [[0, 1, 2]]),
        (4, [[0, 1, 2, 3]]),
        (5, [[0, 1, 2, 3], [4]]),
        (8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ],
)
def test_block_returns_only_tokens_in_requested_block(num_tokens, expected_blocks):
    seq = Sequence(list(range(num_tokens)), block_size=4)

    assert [seq.block(i) for i in range(seq.num_blocks)] == expected_blocks
