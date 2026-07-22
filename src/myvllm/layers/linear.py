import torch.nn as nn 
import torch
import torch.distributed as dist
import os

class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int], # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        param_data = param.data
        # compute offset 
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data 
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result




if __name__ == "__main__":
    # how to run?
    # 1. cd src/myvllm/layers
    # 2. CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --nproc_per_node=4 linear.py # 4 GPUs

    def _init_dist(): 
        rank = int(os.environ["RANK"]) 
        world_size = int(os.environ["WORLD_SIZE"]) 
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) 
        backend = "nccl" if torch.cuda.is_available() else "gloo" 

        if torch.cuda.is_available(): 
            torch.cuda.set_device(local_rank) 
            device = torch.device("cuda", local_rank) 
            dist.init_process_group(
                backend=backend, 
                init_method="env://", 
                device_id=local_rank,
                ) 
        else: 
            device = torch.device("cpu") 
            dist.init_process_group( backend=backend, init_method="env://", ) 
        return rank, world_size, local_rank, device

    # Single linear layer and column parallel test
    @torch.no_grad()
    def test_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()  # Number of parallel GPUs

        in_features = 1024 * tp_size
        out_features = 1024 * tp_size
        batch = 4

        # Ensure that each rank gets exactly the same full input/weight
        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # reference (single GPU)
        single_layer = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single_layer.weight.weight_loader(single_layer.weight, w_full)
        single_layer.bias.weight_loader(single_layer.bias, b_full)
        y_single = single_layer(x_full)

        # TP layer (each rank stores out_features/tp)
        col_tp_layer = ColumnParallelLinear(in_features, out_features, bias=True).to(device)
        col_tp_layer.weight.weight_loader(col_tp_layer.weight, w_full)
        col_tp_layer.bias.weight_loader(col_tp_layer.bias, b_full)

        # forward
        y_col_tp = col_tp_layer(x_full)  # [batch, out_features/tp]

        # Restore full output: all_gather+concat
        y_parts = [torch.empty_like(y_col_tp) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_col_tp)
        y_full = torch.cat(y_parts, dim=-1)  # [batch, out_features]

        # Alignment check (print only at rank0)
        max_err = (y_full - y_single).abs().max().item()
        ok = torch.allclose(y_full, y_single, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[ColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # MergedColumnParallelLinear test
    @torch.no_grad()
    def test_merged_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # Make the dimension automatically adapt to tp_size to ensure divisibility
        in_features = 1024 * tp_size
        out_each = 512 * tp_size
        out_sizes = [out_each, out_each, out_each]  # Combination of Q, K and V matrices
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_q = torch.randn(out_sizes[0], in_features, generator=g)
        w_k = torch.randn(out_sizes[1], in_features, generator=g)
        w_v = torch.randn(out_sizes[2], in_features, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # Reference: single-card equivalent output (Q | K | V concat)
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # TP merged layer
        # NOTE: your MergedColumnParallelLinear defines weight_loader(param, loaded_weights, loaded_weight_id),
        # so bias loader signature doesn't match base (param, loaded_weights). Therefore bias=False here.
        merged = MergedColumnParallelLinear(in_features, out_sizes, bias=False).to(device)
        merged.weight_loader(merged.weight, w_q, 0)
        merged.weight_loader(merged.weight, w_k, 1)
        merged.weight_loader(merged.weight, w_v, 2)

        y_local = merged(x_full)  # [batch, sum(out_sizes)/tp], layout: [q_local, k_local, v_local]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        ql = out_sizes[0] // tp_size
        kl = out_sizes[1] // tp_size
        vl = out_sizes[2] // tp_size

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[MergedColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # QKVColumnParallelLinear test
    @torch.no_grad()
    def test_qkv_column_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        # make input dim divisible by tp_size
        input_size = 1024 * tp_size
        head_size = 16
        num_heads = 4 * tp_size
        num_kv_heads = 2 * tp_size
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, input_size, generator=g)
        w_q = torch.randn(head_size * num_heads, input_size, generator=g)
        w_k = torch.randn(head_size * num_kv_heads, input_size, generator=g)
        w_v = torch.randn(head_size * num_kv_heads, input_size, generator=g)

        x_full = x_full.to(device)
        w_q = w_q.to(device)
        w_k = w_k.to(device)
        w_v = w_v.to(device)

        # reference: full Q|K|V
        y_ref = torch.cat(
            [
                nn.functional.linear(x_full, w_q, None),
                nn.functional.linear(x_full, w_k, None),
                nn.functional.linear(x_full, w_v, None),
            ],
            dim=-1,
        )

        # TP QKV layer: rank output layout [q_local | k_local | v_local]
        qkv = QKVColumnParallelLinear(
            input_size=input_size,
            head_size=head_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            bias=False,
        ).to(device)

        qkv.weight_loader(qkv.weight, w_q, "q")
        qkv.weight_loader(qkv.weight, w_k, "k")
        qkv.weight_loader(qkv.weight, w_v, "v")

        y_local = qkv(x_full)  # [batch, head_size*(local_h + 2*local_kv)]

        # all_gather then re-pack to [Q_all | K_all | V_all]
        y_parts = [torch.empty_like(y_local) for _ in range(tp_size)]
        dist.all_gather(y_parts, y_local)

        ql = head_size * (num_heads // tp_size)
        kl = head_size * (num_kv_heads // tp_size)
        vl = head_size * (num_kv_heads // tp_size)

        q_full = torch.cat([p[:, :ql] for p in y_parts], dim=-1)
        k_full = torch.cat([p[:, ql : ql + kl] for p in y_parts], dim=-1)
        v_full = torch.cat([p[:, ql + kl : ql + kl + vl] for p in y_parts], dim=-1)
        y_full = torch.cat([q_full, k_full, v_full], dim=-1)

        max_err = (y_full - y_ref).abs().max().item()
        ok = torch.allclose(y_full, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[QKVColumnParallel] allclose={ok}, max_abs_err={max_err:.6f}")


    # RowParallelLinear test
    @torch.no_grad()
    def test_row_parallel(device):
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()

        in_features = 128 * tp_size
        out_features = 256
        batch = 4

        g = torch.Generator(device="cpu").manual_seed(2026)
        x_full = torch.randn(batch, in_features, generator=g)
        w_full = torch.randn(out_features, in_features, generator=g)
        b_full = torch.randn(out_features, generator=g)

        x_full = x_full.to(device)
        w_full = w_full.to(device)
        b_full = b_full.to(device)

        # reference
        single = ReplicatedLinear(in_features, out_features, bias=True).to(device)
        single.weight.weight_loader(single.weight, w_full)
        single.bias.weight_loader(single.bias, b_full)
        y_ref = single(x_full)

        # RowParallel
        row_tp = RowParallelLinear(in_features, out_features, bias=True).to(device)
        row_tp.weight.weight_loader(row_tp.weight, w_full)

        if row_tp.bias is not None:
            row_tp.bias.data.copy_(b_full / tp_size)

        shard = in_features // tp_size
        start = tp_rank * shard
        x_part = x_full.narrow(-1, start, shard)

        y_row = row_tp(x_part)

        max_err = (y_row - y_ref).abs().max().item()
        ok = torch.allclose(y_row, y_ref, rtol=1e-4, atol=1e-4)
        if tp_rank == 0:
            print(f"[RowParallel] allclose={ok}, max_abs_err={max_err:.6f}")

    rank, world_size, local_rank, device = _init_dist()
    if rank == 0:
        print(f"Running TP tests with world_size={world_size} on device={device}")

    # The test output 'allclose=True' means passed.
    test_column_parallel(device)
    test_merged_column_parallel(device)
    test_qkv_column_parallel(device)
    test_row_parallel(device)

    dist.barrier()
    dist.destroy_process_group()