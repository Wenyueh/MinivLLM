import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from collections import deque
from unittest.mock import MagicMock
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    max_num_batched_tokens=100,
    max_num_sequences=10,
    max_cached_blocks=100,
    block_size=4,
):
    return Scheduler(
        max_num_sequences=max_num_sequences,
        max_num_batched_tokens=max_num_batched_tokens,
        max_cached_blocks=max_cached_blocks,
        block_size=block_size,
        eos=0,
    )


def inject_running(scheduler: Scheduler, *seqs: Sequence):
    """Put sequences directly into the running queue, bypassing prefill."""
    for seq in seqs:
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)


def all_tracked(scheduler: Scheduler, scheduled: list[Sequence]) -> set:
    """Return the set of all sequences the scheduler currently knows about."""
    return set(scheduler.running) | set(scheduler.waiting) | set(scheduled)


class TestBug2TokenLimitBreak:
    """
    Setup: 3 sequences in running, all can_append=True.
           max_num_batched_tokens=2 → only 2 fit per step.
    Expected after schedule(): seq_c should still be in running.
    Buggy behaviour: seq_c is popleft-ed, limit is hit, break fires,
                     seq_c is never restored → permanently lost.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        seq_c = Sequence([7, 8, 9])
        inject_running(scheduler, seq_a, seq_b, seq_c)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()

        return seq_a, seq_b, seq_c, scheduled, is_prefill

    def test_seq_count_is_correct(self):
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        # Only 2 tokens fit in the batch
        assert len(scheduled) == 2

        # ---- THE BUG: seq_c disappears ----
        # After extendleft, running should be [seq_a, seq_b, seq_c]
        assert seq_c in scheduler.running, (
            "Bug 2: seq_c was popleft-ed and the break fired before it could be "
            "added to scheduled_sequences or put back into self.running → LOST"
        )

    def test_seq_count_limit_variant(self):
        """Same bug but triggered by max_num_sequences instead of token budget."""
        scheduler = make_scheduler(max_num_sequences=2, max_num_batched_tokens=100)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        assert not is_prefill
        assert len(scheduled) == 2

        assert seq_c in scheduler.running, (
            "Bug 2 (seq-count variant): seq_c lost when len(scheduled_sequences) "
            ">= max_num_sequences caused the break"
        )

    def test_no_sequence_is_lost(self):
        """Total universe of sequences must be conserved."""
        scheduler = make_scheduler(max_num_batched_tokens=2)
        seq_a, seq_b, seq_c, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        for seq in (seq_a, seq_b, seq_c):
            assert seq in tracked, f"seq {seq.seq_id} disappeared from the scheduler"


class TestBug1CanAppendFailure:
    """
    Setup: 2 sequences in running, can_append returns False for the first.
    Expected: seq_a is either put back into running (to retry later) or
              preempted into waiting; it must NOT disappear entirely.
    Buggy behaviour: seq_a is popleft-ed, can_append fails, the code does
                     self.preempt(self.running.pop()) which preempts seq_b,
                     but seq_a is never handled → lost.
    """

    def _run(self, scheduler: Scheduler):
        seq_a = Sequence([1, 2, 3])
        seq_b = Sequence([4, 5, 6])
        inject_running(scheduler, seq_a, seq_b)

        mock_bm = MagicMock()
        # First call (for seq_a): cannot append; subsequent calls: True
        mock_bm.can_append.side_effect = [False, True, True, True]
        mock_bm.append.return_value = None
        mock_bm.deallocate.return_value = None
        scheduler.block_manager = mock_bm

        scheduled, is_prefill = scheduler.schedule()
        return seq_a, seq_b, scheduled, is_prefill

    def test_seq_a_not_lost(self):
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, (
            "Bug 1: seq_a was popleft-ed, can_append returned False, "
            "self.preempt(self.running.pop()) preempted seq_b instead, "
            "and seq_a was never restored → LOST"
        )

    def test_total_conservation(self):
        """Neither seq must disappear."""
        scheduler = make_scheduler()
        seq_a, seq_b, scheduled, is_prefill = self._run(scheduler)

        tracked = all_tracked(scheduler, scheduled)
        assert seq_a in tracked, f"seq_a disappeared"
        assert seq_b in tracked, f"seq_b disappeared"


class TestSchedulerHappyPath:
    def test_prefill_scheduled_first(self):
        scheduler = make_scheduler(max_num_batched_tokens=100, max_cached_blocks=50)
        seq = Sequence([1, 2, 3, 4])
        scheduler.add_sequence(seq)

        scheduled, is_prefill = scheduler.schedule()
        assert is_prefill
        assert seq in scheduled
        assert seq in scheduler.running

    def test_all_running_seqs_scheduled_when_budget_allows(self):
        scheduler = make_scheduler(max_num_batched_tokens=10)
        seq_a = Sequence([1])
        seq_b = Sequence([2])
        inject_running(scheduler, seq_a, seq_b)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = True
        scheduler.block_manager.append.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 2
        # Both should be back in running after schedule()
        assert seq_a in scheduler.running
        assert seq_b in scheduler.running

    def test_preempt_only_seq_when_cant_append_and_running_empty(self):
        """Original else-branch: running=[seq], can_append=False → preempt seq → waiting."""
        scheduler = make_scheduler()
        seq = Sequence([1, 2])
        inject_running(scheduler, seq)

        scheduler.block_manager = MagicMock()
        scheduler.block_manager.can_append.return_value = False
        scheduler.block_manager.deallocate.return_value = None

        scheduled, is_prefill = scheduler.schedule()
        assert not is_prefill
        assert len(scheduled) == 0
        assert seq in scheduler.waiting
        assert seq.status == SequenceStatus.WAITING
