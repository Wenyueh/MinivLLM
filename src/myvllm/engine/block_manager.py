import math
import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is still in use"
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def get_cache_hit_tokens(self, seq: Sequence):
        max_cache_hit_length = seq.num_tokens - 1
        num_need_cache_hit_blocks = int(math.floor(max_cache_hit_length / self.block_size))
        num_cached_tokens = 0  # cache hit token count
        cache_hit_blocks = []
        h = -1
        for i in range(num_need_cache_hit_blocks):
            token_ids = seq.block(i)
            # compute hash
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h)
            block_id = self.hash_to_block_id.get(h, -1)

            # cache miss
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # cache hit
            num_cached_tokens += self.block_size
            cache_hit_blocks.append(block_id)

        return cache_hit_blocks, num_cached_tokens



    def deallocate(self, seq: Sequence) -> None:
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0

    def allocate(
            self, 
            seq: Sequence, 
            num_new_tokens: int,                   # scheduler tokens
            num_new_computed_tokens: int = 0,      # cache hit tokens 
            new_computed_blocks: list[int] = None  # cache hit blocks
    ):

        # base case: judge whether there is enough space
        num_last_computed_block_tokens = seq.num_computed_tokens % self.block_size
        if num_last_computed_block_tokens != 0: 
            num_last_computed_remaining_tokens = self.block_size - num_last_computed_block_tokens
            if num_new_tokens > num_last_computed_remaining_tokens:
                num_need_allocate_new_blocks = int(math.ceil((num_new_tokens - num_last_computed_remaining_tokens) / self.block_size))
                if num_need_allocate_new_blocks > len(self.free_block_ids):
                    return False
            else:
                return True
        else: 
            num_need_allocate_new_blocks = int(math.ceil(num_new_tokens  / self.block_size))
            if num_need_allocate_new_blocks > len(self.free_block_ids):
                return False

        # cache block ref_cout + 1
        if num_new_computed_tokens != 0 and new_computed_blocks:
            seq.num_cached_tokens = num_new_computed_tokens
            seq.num_computed_tokens = num_new_computed_tokens
            for block_id in new_computed_blocks:
                block = self.blocks[block_id]
                block.ref_count += 1
                seq.block_table.append(block_id)

        # compute full block hash and determine the left scheduler tokens
        if seq.num_computed_tokens > 0:
            num_computed_blocks = int(math.ceil(seq.num_computed_tokens / self.block_size))
            pre_block_id = seq.block_table[num_computed_blocks - 1]
            pre_block = self.blocks[pre_block_id]
            if num_computed_blocks == 1:
                h = -1
            else:
                pre_pre_block_id = seq.block_table[num_computed_blocks - 2]
                h = self.blocks[pre_pre_block_id].hash
            # compute hash
            h = self.compute_hash(token_ids=seq.block(num_computed_blocks - 1), prefix_hash_value=h)
            self.hash_to_block_id[h] = pre_block.block_id
            if num_last_computed_block_tokens != 0:
                num_next_new_tokens = num_new_tokens - (self.block_size - num_last_computed_block_tokens)
            else:
                num_next_new_tokens = num_new_tokens
        else:
            # seq first allocate block and not cache hit
            num_next_new_tokens = num_new_tokens
            num_computed_blocks = 0
            h = -1

        next_new_last_block_slots = num_next_new_tokens % self.block_size
        num_next_new_blocks = int(math.ceil(num_next_new_tokens / self.block_size))

        for i in range(num_next_new_blocks):
            # middle block or last block is full
            if i != num_next_new_blocks - 1 or next_new_last_block_slots == 0:
                block = self._allocate_block(self.free_block_ids[0])
                h = self.compute_hash(token_ids=seq.block(num_computed_blocks + i), prefix_hash_value=h)
                block.update(h=h, token_ids=seq.block(num_computed_blocks + i))
                self.hash_to_block_id[h] = block.block_id
                seq.block_table.append(block.block_id)
            else:
                # last block is not full
                block = self._allocate_block(self.free_block_ids[0])
                seq.block_table.append(block.block_id) 
        
        return True
