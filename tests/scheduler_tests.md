# Scheduler Tests

## Setup

```bash
pip install pytest xxhash
```

## Run

All tests:
```bash
python3 -m pytest tests/test_scheduler.py -v
```

A specific class:
```bash
python3 -m pytest tests/test_scheduler.py::TestBug2TokenLimitBreak -v
python3 -m pytest tests/test_scheduler.py::TestBug1CanAppendFailure -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerHappyPath -v
```

## Test Classes

### TestBug2TokenLimitBreak

Guards against sequences being silently dropped when the token budget (or sequence-count limit) is exhausted mid-loop.

Tests:
- `test_seq_count_is_correct` — only 2 sequences fit in a 2-token budget; `seq_c` must remain in `running`
- `test_seq_count_limit_variant` — same bug triggered by `max_num_sequences` instead of token budget
- `test_no_sequence_is_lost` — total sequence conservation: every sequence must be in `running`, `waiting`, or `scheduled`

### TestBug1CanAppendFailure

Guards against sequences being lost when `block_manager.can_append` returns `False`.

Tests:
- `test_seq_a_not_lost` — `seq_a` must appear in `running`, `waiting`, or `scheduled` after the call
- `test_total_conservation` — neither `seq_a` nor `seq_b` may disappear

### TestSchedulerHappyPath

Basic correctness of the scheduler under normal conditions.

Tests:
- `test_prefill_scheduled_first` — a newly added sequence is scheduled as prefill and moved to `running`
- `test_all_running_seqs_scheduled_when_budget_allows` — when the token budget is large enough, all running sequences are scheduled and remain in `running`
- `test_preempt_only_seq_when_cant_append_and_running_empty` — when the only running sequence cannot append, it is preempted to `waiting` with status `WAITING`
