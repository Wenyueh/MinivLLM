import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


# vocabparallelembedding
# shard over the number of vocab, not the embedding size

class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        assert num_embeddings % self.tp_size == 0, "num_embeddings must be divisible by tp_size"
        # keep the original num_embeddings
        self.num_embeddings = num_embeddings
        # this is the num_embeddings per partition in this current GPU
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data

        offset = self.tp_rank * self.num_embeddings_per_partition
        shard_size = self.num_embeddings_per_partition

        sharded_weights = loaded_weights.narrow(0, offset, shard_size)
        param_data.copy_(sharded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition)
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # need to mask again, otherwise the embedding for the out-of-range ids will be the embedding of id 0
            output = mask.unsqueeze(1) * output
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

# weight tying with embedding layer
class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    # x: [batch_size, seq_len, hidden_size]
    # weight: [vocab_size_per_partition, hidden_size]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            # cu_seqlens_q = [0, 5, 8, 12]
            # last_indices = [5, 8, 12] - 1 = [4, 7, 11]
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0
            x = x[last_token].contiguous()

        # logits: [batch_size, seq_len, vocab_size_per_partition]
        # F.linear automatically transpose the weight
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # prepare for all_gather only for GPU 0 which is the main GPU
            all_logits = [torch.empty(logits.size()) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # dist.gather collects the logits from all GPUs to GPU 0
            dist.gather(logits, gather_list=all_logits, dst=0)
            # concatenate
            if self.tp_rank == 0:
                # [batch_size, seq_len, vocab_size]
                logits = torch.cat(all_logits, dim=-1)

        return logits