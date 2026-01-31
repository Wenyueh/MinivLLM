from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int,
                 summary_end: int = None, content_start: int = None, content_end: int = None):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos
        # sparse attention special tokens
        self.summary_end = summary_end
        self.content_start = content_start
        self.content_end = content_end


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether we can append one more token
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)

            # Special token handling: summary_end triggers dead range marking
            if self.summary_end is not None and token_id == self.summary_end:
                # Search backwards for closest content_start and content_end
                pos_content_start = None
                pos_content_end = None

                # Search from the second-to-last position (exclude the just-added summary_end)
                for i in range(len(seq.token_ids) - 2, -1, -1):
                    if seq.token_ids[i] == self.content_start and pos_content_start is None:
                        pos_content_start = i
                    if seq.token_ids[i] == self.content_end and pos_content_end is None:
                        pos_content_end = i
                    # Stop if we found both
                    if pos_content_start is not None and pos_content_end is not None:
                        break

                # Only mark if both tokens found AND content_start comes before content_end
                if pos_content_start is not None and pos_content_end is not None and pos_content_start < pos_content_end:
                    # Mark range from content_start to content_end (inclusive)
                    start = pos_content_start
                    end = pos_content_end + 1  # +1 because mark_dead uses [start, end)

                    seq.mark_dead(start, end)
                    # Evict blocks that are now fully dead
                    blocks_freed = self.block_manager.evict_dead_blocks(seq)
                    print(f"[Sparse Attention] summary_end ({self.summary_end}) at pos {len(seq.token_ids)-1}: "
                          f"marked tokens [{start}, {end}) as dead (content_start at {pos_content_start}, "
                          f"content_end at {pos_content_end}), freed {blocks_freed} blocks")

            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = 1 + seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)