from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        self.block_size = block_size # number of tokens per block
        # record sequence id
        self.seq_id = next(Sequence.counter)
        # status
        self.status = SequenceStatus.WAITING
        # token ids, need copy so that it is a new list, won't be affected by outside changes
        self.token_ids = copy(token_ids)
        # last token
        self.last_token = self.token_ids[-1] if self.token_ids else None
        # num_tokens, num_prompt_tokens
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        # num_cached_tokens = 0
        self.num_cached_tokens = 0
        # block_table
        self.block_table = []
        # sampling_params' related things
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, idx):
        return self.token_ids[idx]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - max(self.num_blocks - 1, 0) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        # Preserve every attribute instead of a hand-maintained tuple. The tuple
        # silently desynchronized when a field was added, which is exactly what
        # #89 is: the worker unpickled a Sequence without block_size and raised
        # AttributeError from num_blocks / num_cached_blocks /
        # last_block_num_tokens. Routing the whole __dict__ means a new field
        # cannot fall out of the protocol again.
        state = self.__dict__.copy()
        # Decode only needs the newest token on the worker; shipping the whole
        # token list every step would make the shared-memory hop O(seq length).
        if self.num_completion_tokens > 0:
            state["token_ids"] = [self.last_token]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
