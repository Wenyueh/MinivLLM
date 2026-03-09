from collections import deque
from dataclasses import dataclass

from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager

import torch


@dataclass
class SchedulerOutput:
    scheduled_seqs: list[Sequence]
    num_scheduled_tokens: dict[int, int]

    def __getstate__(self):
        return {
            'scheduled_seqs': self.scheduled_seqs,
            'num_scheduled_tokens': self.num_scheduled_tokens
        }
    
    def __setstate__(self, state):
        self.scheduled_seqs = state['scheduled_seqs']
        self.num_scheduled_tokens = state['num_scheduled_tokens']


class Scheduler:
    def __init__(
            self, 
            max_num_sequences: int, 
            max_num_batched_tokens: int, 
            max_cached_blocks: int,
            long_prefill_token_threshold: int, 
            block_size: int, 
            eos: int
        ):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.long_prefill_token_threshold = long_prefill_token_threshold
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs: list[Sequence] = []
        preempted_seqs: list[Sequence] = []

        num_scheduled_tokens: dict[int, int] = {}
        token_budget = self.max_num_batched_tokens

        seq_idx = 0
        while seq_idx < len(self.running) and token_budget > 0:
            sequence = self.running[seq_idx]

            # num_new_tokens = (prompt + output) - computed
            num_new_tokens = sequence.num_tokens - sequence.num_computed_tokens
            
            # chunk_prefill
            if 0 < self.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            while True:
                # allocate block
                is_allocated = self.block_manager.allocate(sequence,num_new_tokens)
                if is_allocated:
                    break
                
                # preempt
                preempted_seq = self.running.pop()
                self.preempt(preempted_seq)
                preempted_seqs.append(preempted_seq)
                if preempted_seq == sequence:
                    break
            
            if is_allocated == False:
                break

            scheduled_seqs.append(sequence)
            num_scheduled_tokens[sequence.seq_id] = num_new_tokens
            token_budget -= num_new_tokens
            seq_idx += 1

        
        if not preempted_seqs: 
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_sequences:
                    break
                sequence = self.waiting[0]
                
                # prefix caching
                cache_hit_blocks, num_new_computed_tokens = self.block_manager.get_cache_hit_tokens(sequence)
                num_computed_tokens = num_new_computed_tokens
                num_new_tokens = sequence.num_tokens - num_computed_tokens

                # chunked_prefill
                if 0 < self.long_prefill_token_threshold < num_new_tokens:
                    num_new_tokens = self.long_prefill_token_threshold
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0  

                # allocate block
                is_allocated = self.block_manager.allocate(
                    sequence,
                    num_new_tokens,
                    num_new_computed_tokens,
                    cache_hit_blocks
                )
                if is_allocated == False:
                    break
                
                # allocate block success
                sequence = self.waiting.popleft()
                self.running.append(sequence)
                scheduled_seqs.append(sequence)
                num_scheduled_tokens[sequence.seq_id] = num_new_tokens
                token_budget -= num_new_tokens
                sequence.status = SequenceStatus.RUNNING

        return SchedulerOutput(scheduled_seqs, num_scheduled_tokens)      


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.PREEMPTED
        seq.num_computed_tokens = 0
        seq.num_preeptions += 1
        self.waiting.appendleft(seq)      


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, scheduler_output: SchedulerOutput, token_ids: torch.Tensor, discard_seq_ids: list[int]) -> None:
        seqs: list[Sequence] = scheduler_output.scheduled_seqs
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        token_ids = token_ids.tolist()
        for seq, token_id in zip(seqs, token_ids):
            num_computed_tokens = num_scheduled_tokens[seq.seq_id]

            if seq.seq_id in discard_seq_ids:
                seq.num_computed_tokens += num_computed_tokens
                continue

            seq.num_computed_tokens += num_computed_tokens
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)