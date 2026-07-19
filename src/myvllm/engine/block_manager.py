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
        # Multiple physical blocks can represent the same logical prefix.
        self.hash_to_block_ids: dict[int, set[int]] = {}
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

    def _register_hash(self, hash_value: int, block_id: int) -> None:
        if hash_value != -1:
            self.hash_to_block_ids.setdefault(hash_value, set()).add(block_id)

    def _unregister_hash(self, hash_value: int, block_id: int) -> None:
        block_ids = self.hash_to_block_ids.get(hash_value)
        if block_ids is None:
            return
        block_ids.discard(block_id)
        if not block_ids:
            del self.hash_to_block_ids[hash_value]

    # move this block to used list and update all information of the block
    def _allocate_new_block(self, block_id: int, hash: int, token_ids: list[int]) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        # 防止出现old_hash已经映射到其他的block上了导致误删
        # 上述情况在prefix未连续命中会发生（如prefix的前面一部分被释放了，但是后面还有没被释放的）
        # 因此后面没释放的也不能命中，要新allocate block，但是这些block会计算出hash map中已有的hash值，改变block id指向
        old_hash = block.hash
        self._unregister_hash(old_hash, block_id)
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.update(hash, token_ids)
        block.ref_count += 1
        return block

    # move this block to used list and change ref_count only
    def _allocate_hit_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        block.ref_count += 1
        return block

    # Keep block contents and hash indexes for inter-request prefix caching.
    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _get_prefix_hit_blocks(self, seq: Sequence) -> list[Block]:
        """Return the longest contiguous, reusable prefix of ``seq``.

        The last prompt token must still be computed to produce logits, so the
        block containing it is deliberately excluded from the cache hit.
        """
        h = -1
        hit_blocks = []
        max_cacheable_blocks = max(0, (seq.num_tokens - 1) // self.block_size)

        for i in range(max_cacheable_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_ids = self.hash_to_block_ids.get(h)
            if not block_ids:
                break
            matching_blocks = [
                self.blocks[block_id]
                for block_id in block_ids
                if self.blocks[block_id].token_ids == token_ids
            ]
            if not matching_blocks:  # hash collision
                break
            # Sharing an in-use block consumes no free block, so prefer it.
            block = max(matching_blocks, key=lambda candidate: candidate.ref_count > 0)
            hit_blocks.append(block)

        return hit_blocks

    # whether we can allocate blocks for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        hit_blocks = self._get_prefix_hit_blocks(seq)

        # A hit on an in-use block only increments its ref_count and therefore
        # needs no free block. A hit with ref_count == 0 is still in the free
        # queue; allocating the request removes it from that queue, so it must
        # remain part of the capacity check.
        num_shared_hit_blocks = sum(block.ref_count > 0 for block in hit_blocks)
        num_required_free_blocks = seq.num_blocks - num_shared_hit_blocks
        return len(self.free_block_ids) >= num_required_free_blocks

    def get_num_cached_tokens(self, seq: Sequence) -> int:
        """Return the reusable prefix length without mutating the sequence."""
        return len(self._get_prefix_hit_blocks(seq)) * self.block_size


    def allocate(self, seq: Sequence) -> None:
        hit_blocks = self._get_prefix_hit_blocks(seq)
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            if i < len(hit_blocks):
                block = hit_blocks[i]
                seq.num_cached_tokens += self.block_size
                # Prefix of a finished request is hit.
                if block.ref_count == 0:
                    self._allocate_hit_block(block.block_id)
                # Prefix of an on-going request is hit.
                else:
                    block.ref_count += 1
                seq.block_table.append(block.block_id)
                continue
            # cache miss
            block = self._allocate_new_block(self.free_block_ids[0], h, token_ids)
            self._register_hash(h, block.block_id)
            seq.block_table.append(block.block_id)
        
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

    # this is to check whether we can append tokens to this sequence
    # The token was already added to the sequence by postprocess(). A new
    # block is needed when that token is the first token in a block.
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 1:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]
        token_ids = seq.block(seq.num_blocks - 1)

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = token_ids, prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self._register_hash(h, block.block_id)
        # if one new block is needed
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_new_block(self.free_block_ids[0], -1, token_ids)
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
