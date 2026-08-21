# MinivLLM Benchmark 完整报告

> 报告更新时间：2026-08-21（UTC）  
> 测试状态：Paged Attention Decoding 与 4-GPU TPS 测试均已完成  
> 正式 TPS Job：`minivllm-tps-benchmark-4gpu`，状态 `Complete`，Pod 退出码 `0`

## 1. 执行摘要

本次测试包含两部分：Paged Attention decoding kernel 微基准，以及 Qwen3-0.6B 端到端生成 TPS 基准。

主要结论：

1. 新 Triton decoding kernel 在全部 9 组配置中通过正确性检查，并相对 Naive PyTorch 获得 `6.52x–83.12x` 加速。
2. 4-GPU 正式 TPS 测试已成功跑通 MiniVLLM 与 vLLM；Transformers 作为单 GPU baseline。
3. 在本次小批量、短序列测试中，vLLM 达到 `1463.8772 tokens/s`，MiniVLLM 达到 `64.5980 tokens/s`，前者为后者的 `22.66x`。
4. Transformers 单 GPU baseline 为 `135.3840 tokens/s`。由于 GPU 数量、输入模板、采样策略和实现路径均不完全相同，该数据只作为参考，不应视为与 4-GPU TP 的严格等价对照。
5. 正式 Job 最终状态为 `Succeeded`，三个 backend 均输出完整结果，最终日志中没有 `ERROR`、`Traceback` 或 `RuntimeError`。

## 2. 测试环境

### 2.1 硬件与集群

| 项目 | 配置 |
|---|---|
| Kubernetes namespace | `ns008` |
| 节点 | `hyperpod-i-0de6f595ddbf7f24b` |
| 实例类型 | `ml.p5en.48xlarge` |
| GPU | NVIDIA H200 |
| 节点 GPU 数 | 8 |
| TPS Job 请求 GPU 数 | 4 |
| `/dev/shm` | 16 GiB memory-backed `emptyDir` |

### 2.2 软件与镜像

| 项目 | 版本/值 |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.9.1+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Triton | 3.5.1 |
| Transformers | 4.57.3 |
| vLLM | 0.15.0 |
| 正式镜像 | `028236335745.dkr.ecr.us-east-2.amazonaws.com/galileo/minivllm:v6` |
| 镜像 digest | `sha256:0174bc9f91db03e39f33c2d86980090ba5051b487a664d95e13b09cab7a4089b` |
| ECR 压缩镜像大小 | 14,382,150,318 bytes，约 14.38 GB |

`v6` 镜像已将运行依赖 `libxcb1` 固化到镜像中，不再依赖 Pod 启动后临时安装系统包。

## 3. TPS Benchmark 方法

### 3.1 工作负载

- 模型：`Qwen/Qwen3-0.6B`
- 输入数量：3 条 prompt
- 最大新生成 token：每条 256
- 最大模型长度：512
- warmup：每个 backend 2 次完整生成
- 正式计时：warmup 后执行 1 次完整生成
- TPS 定义：`实际生成 token 总数 / 正式生成耗时`
- 计时：`time.perf_counter()`，计时结束前执行 CUDA synchronize
- MiniVLLM block size：256
- MiniVLLM `gpu_memory_utilization`：0.9
- vLLM `gpu_memory_utilization`：0.75
- vLLM speculative decoding：关闭

测试 prompt：

1. `introduce yourself`
2. `list all prime numbers within 100`
3. `give me your opinion on the impact of artificial intelligence on society`

### 3.2 GPU 使用方式

| Backend | requested_world_size | effective_gpus | 并行方式 |
|---|---:|---:|---|
| MiniVLLM | 4 | 4 | 自定义 tensor parallel |
| vLLM | 4 | 4 | `tensor_parallel_size=4` |
| Transformers | 4 | 1 | 单 GPU，无 tensor parallel |

Transformers 虽运行在请求了 4 张 GPU 的同一 Pod 中，但脚本只把模型放到默认 CUDA device，因此其 `effective_gpus=1`。

## 4. 4-GPU TPS 正式结果

正式数据来自第二轮最终成功的 Job，时间范围为 `2026-08-21 03:46:33–03:50:06 UTC`。

| Backend | 有效 GPU | 生成 token | 延迟 (s) | TPS (tokens/s) | 相对 MiniVLLM |
|---|---:|---:|---:|---:|---:|
| MiniVLLM | 4 | 629 | 9.7371 | 64.5980 | 1.00x |
| vLLM | 4 | 628 | 0.4290 | 1463.8772 | 22.66x |
| Transformers | 1 | 768 | 5.6728 | 135.3840 | 2.10x（非等价配置） |

### 4.1 MiniVLLM 与 vLLM 对比

MiniVLLM 与 vLLM 都使用 4-GPU tensor parallel，并分别生成 629 和 628 个 token，计时工作量非常接近，因此这两项具有本报告中最高的可比性。vLLM 的 TPS 是 MiniVLLM 的 `22.66x`，正式生成延迟约为 MiniVLLM 的 `1/22.70`。

日志显示 MiniVLLM 的持续 decoding 吞吐通常约为整个 batch `80 tokens/s` 左右，但端到端正式 TPS 还包含 prefill、不同序列提前结束以及调度开销，最终为 `64.5980 tokens/s`。

vLLM 使用 FlashAttention、torch.compile、CUDA Graph、异步调度和成熟的通信/调度实现。其首次 engine 初始化耗时约 54.35 秒，其中 torch.compile 约 43.68 秒；这些初始化成本发生在正式计时和 warmup 之前，因此没有计入 `0.4290 s` 的生成延迟。

### 4.2 Transformers baseline 解读

Transformers baseline 的 `135.3840 tokens/s` 高于本轮 MiniVLLM，但不能据此直接得出“单 GPU 比 4 GPU 更快”的一般性结论，原因包括：

- Transformers 实际只使用 1 张 GPU，而另外两项使用 4-GPU TP；对 0.6B 小模型而言，TP 通信开销可能超过分片计算收益。
- MiniVLLM/vLLM 使用 chat template，当前 Transformers 路径直接 tokenize 原始 prompt，输入 token 数不同。
- MiniVLLM/vLLM 使用 `temperature=0.6`；Transformers 路径未显式启用相同采样配置，采用其默认 generation 行为。
- 三个实现没有在脚本中显式统一 dtype；vLLM 日志明确显示 `bfloat16`。
- Transformers 生成了 768 个 token，而另外两项因 EOS/采样差异生成约 628–629 个 token。TPS 已按实际 token 数归一化，但延迟不能直接横向比较。

## 5. Paged Attention Decoding Benchmark

### 5.1 配置

- `num_heads=16`
- `num_kv_heads=8`
- `head_dim=128`
- `block_size=256`
- 每组 10 次 warmup、100 次计时迭代
- 所有实现先与 float32 reference 做正确性对比

### 5.2 结果

| Batch | Seq Len | Naive PyTorch (ms) | Fast PyTorch (ms) | Triton old (ms) | Triton new (ms) | Triton new 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 0.217 | 0.486 | 0.080 | 0.019 | 11.18x |
| 1 | 512 | 0.251 | 0.487 | 0.316 | 0.020 | 12.56x |
| 1 | 2048 | 0.446 | 0.524 | 1.259 | 0.068 | 6.52x |
| 8 | 128 | 0.533 | 2.512 | 0.080 | 0.019 | 27.62x |
| 8 | 512 | 0.795 | 2.537 | 0.316 | 0.020 | 40.46x |
| 8 | 2048 | 2.707 | 3.308 | 2.142 | 0.094 | 28.95x |
| 32 | 128 | 1.581 | 9.400 | 0.083 | 0.019 | 83.12x |
| 32 | 512 | 2.949 | 9.738 | 0.535 | 0.044 | 66.97x |
| 32 | 2048 | 10.494 | 12.818 | 2.124 | 0.160 | 65.67x |

### 5.3 Decoding 结论

1. 9 组配置下的 4 种实现全部通过正确性检查。
2. `Triton new` 在所有配置下都是最快实现，相对 Naive PyTorch 加速为 `6.52x–83.12x`。
3. 新 Triton kernel 在 `seq_len=128` 时从 batch 1 到 batch 32 均保持约 0.019 ms，表现出良好的 batch 扩展性。
4. `Triton old` 使用逐 token 标量循环，序列变长后性能明显下降；新 kernel 的整块 masked gather 避免了这一瓶颈。
5. `Fast PyTorch` 在本测试中慢于 Naive PyTorch，主要开销来自 padded K/V 分配、gather 和 Python 层 batch 循环。

## 6. 本轮发现并修复的问题

### 6.1 OpenCV 系统依赖缺失

原始错误：

```text
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```

原因是 `opencv-python-headless` 仍动态链接基础 `libxcb`，而原镜像未安装对应系统库。修复为在 Dockerfile 中安装 `libxcb1`，并已通过镜像内 `import cv2` smoke test。

### 6.2 Sequence 反序列化缺少 block_size

原始错误：

```text
AttributeError: 'Sequence' object has no attribute 'block_size'
```

Rank 0 通过 pickle/共享内存把 `Sequence` 发送给 worker 时，`__getstate__()` 没有包含 `block_size`，`__setstate__()` 也没有恢复该属性。修复后执行了 pickle round-trip 单元检查，4-GPU MiniVLLM 的 3 个 worker 均成功完成正式生成。

### 6.3 vLLM fork 后重复初始化 CUDA

原始错误：

```text
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

vLLM 默认 worker multiprocessing method 为 `fork`，而父进程已触发 CUDA 初始化。Kubernetes 清单现已设置：

```yaml
- name: VLLM_WORKER_MULTIPROC_METHOD
  value: spawn
```

修复后 4 个 TP rank 均完成 NCCL 初始化、模型加载、torch.compile、CUDA Graph capture 和正式生成。

### 6.4 barrier 警告

日志中的以下内容是 PyTorch warning，不是失败原因：

```text
barrier(): using the device under current context
```

可在未来通过向 `init_process_group` 显式传入 `device_id` 消除，但不影响本轮结果有效性。

## 7. 有效性与局限

### 7.1 已验证项目

- 正式 Pod phase：`Succeeded`
- 容器退出码：`0`
- MiniVLLM 成功加载 283 组参数，4 个 rank 均完成 KV cache 初始化
- vLLM 4 个 TP rank 均完成 NCCL 初始化
- 三个 backend 均完成 2 次 warmup 和 1 次正式计时
- 最终正式日志无异常堆栈
- Paged Attention 各实现已与 float32 reference 做正确性对比

### 7.2 局限

1. 每个 backend 只有 1 次正式计时，没有报告均值、P50/P95 或标准差。
2. `temperature=0.6` 会导致输出长度和路径存在随机波动；一次额外 MiniVLLM 验证运行得到 `68.5456 tokens/s`，而正式成功 Job 为 `64.5980 tokens/s`。
3. 模型只有 0.6B，4-GPU tensor parallel 很可能处于通信开销主导区间，不能外推到 7B、32B 或更大模型。
4. batch 只有 3，未覆盖高并发 continuous batching 场景。
5. vLLM 与 MiniVLLM 未显式统一 dtype、kernel、缓存策略和调度策略；当前结果是端到端系统表现，不是单变量对照实验。
6. Transformers baseline 的 prompt 处理和采样策略与另外两项不同，只适合作为参考。
7. TPS 正式计时不包含模型加载、torch.compile 和 CUDA Graph 初始化时间，反映的是 warmup 后 steady-state generation throughput。

## 8. 复现方式

构建并推送镜像：

```bash
docker build -t 028236335745.dkr.ecr.us-east-2.amazonaws.com/galileo/minivllm:v6 .
docker push 028236335745.dkr.ecr.us-east-2.amazonaws.com/galileo/minivllm:v6
```

重新运行同名 Kubernetes Job 时，需要先删除已完成的 Job，因为 Job Pod template 是不可变字段：

```bash
kubectl delete job minivllm-tps-benchmark-4gpu -n ns008 --ignore-not-found
kubectl apply -f k8s/benchmark-tps-4gpu-job.yaml
kubectl logs -n ns008 -f job/minivllm-tps-benchmark-4gpu
```

检查完成状态：

```bash
kubectl wait --for=condition=complete job/minivllm-tps-benchmark-4gpu -n ns008 --timeout=15m
```

## 9. 总结与后续建议

本轮已完成从故障定位、源码修复、镜像重建、ECR 推送、4-GPU 部署到正式结果采集的完整闭环。MiniVLLM 的多进程序列化和 4-GPU tensor parallel 路径已能稳定完成端到端生成，但其当前 steady-state TPS 与成熟 vLLM 仍有明显差距。

下一阶段若要获得更具决策价值的数据，建议固定随机种子并统一 chat template、采样参数和 dtype；每组至少重复 10 次，报告均值/P50/P95；再增加 1/2/4 GPU scaling、不同 batch/concurrency、不同 prompt/output 长度，以及 7B 以上模型。这样可以区分通信开销、kernel 性能、调度效率和模型规模对 TPS 的具体影响。
