from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int,
                 max_cached_blocks: int, block_size: int, eos: int,
                 max_model_length: int):
        if max_model_length <= 0:
            raise ValueError("max_model_length must be greater than 0")

        max_kv_cache_tokens = max_cached_blocks * block_size
        if max_model_length > max_kv_cache_tokens:
            raise ValueError(
                f"max_model_length ({max_model_length}) exceeds the KV cache "
                f"capacity ({max_kv_cache_tokens} tokens)"
            )

        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        self.max_model_length = max_model_length
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos
        # Prefill and decode use different kernels. Alternate when both queues
        # have work so neither newly arrived prompts nor decodes can starve.
        self._last_scheduled_prefill = False


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        request_max_length = sequence.max_model_length
        if request_max_length is None:
            request_max_length = self.max_model_length
        elif request_max_length > self.max_model_length:
            raise ValueError(
                f"Request max_model_length ({request_max_length}) exceeds the "
                f"engine limit ({self.max_model_length})"
            )
        elif request_max_length <= 0:
            raise ValueError("Request max_model_length must be greater than 0")

        # Prefill must leave room for at least one completion token. We reject
        # overlong prompts instead of silently dropping their prefix.
        if sequence.num_prompt_tokens >= request_max_length:
            raise ValueError(
                f"Prompt length ({sequence.num_prompt_tokens}) must be smaller "
                f"than the request context limit ({request_max_length})"
            )
        sequence.max_model_length = request_max_length
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # try schedule for prefilling from waiting queue if not exceeding limits
        try_prefill = self.waiting and (
            not self.running or not self._last_scheduled_prefill
        )
        while try_prefill and self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            num_prefill_tokens = (
                len(seq) - self.block_manager.get_num_cached_tokens(seq)
            )
            if num_prefill_tokens > self.max_num_batched_tokens:
                raise ValueError(
                    f"Request needs {num_prefill_tokens} prefill tokens, but "
                    f"max_num_batched_tokens is {self.max_num_batched_tokens}; "
                    "chunked prefill is not supported"
                )
            if self.block_manager.can_allocate(seq) and num_prefill_tokens + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq) - seq.num_cached_tokens
            else:
                break
        if scheduled_sequences:
            self._last_scheduled_prefill = True
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    self.running.appendleft(seq)
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))
            self._last_scheduled_prefill = False

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        if token_ids is None or len(seqs) != len(token_ids):
            actual = 0 if token_ids is None else len(token_ids)
            raise RuntimeError(
                f"Sampler returned {actual} tokens for {len(seqs)} sequences"
            )
        for seq, token_id in zip(seqs, token_ids):
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
