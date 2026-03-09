import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # 推理引擎参数
    'max_num_sequences': 16,          # 最大并发处理的prompt数量（同时跑16个对话）
    'max_num_batched_tokens': 1024,   # 批处理的最大token数（显存优化）
    'max_cached_blocks': 1024,        # KV缓存的最大块数（减少显存占用）
    'block_size': 256,                # KV缓存的块大小（Qwen3的优化参数）
    'world_size': 1,                  # 分布式推理的GPU数量（1=单卡）
    'enforce_eager': True,            # 禁用PyTorch JIT编译，用eager模式运行（调试/兼容友好）
    # 模型结构参数（匹配Qwen3-0.6B的真实结构）
    # 关键1：保留框架能识别的模型名，让ModelRunner通过这个值识别Qwen3-0.6B
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    # 关键2：新增参数，存储本地模型路径
    'local_model_path': '/home/rejor/Works/CUDA/Qwen3-0.6B',
    'vocab_size': 151936,             # 词表大小（修复：原错误值151643，匹配真实Qwen3）
    'hidden_size': 1024,              # 模型隐藏层维度（Qwen3-0.6B的核心参数）
    'num_heads': 16,                  # 注意力头数
    'head_dim': 128,                  # 每个注意力头的维度（修复：原64，应为1024/8=128）
    'num_kv_heads': 8,                # KV注意力头数（GQA优化，减少显存）
    'intermediate_size': 3072,        # FFN层中间维度
    'num_layers': 28,                 # 模型的Transformer层数
    'tie_word_embeddings': True,      # 词嵌入和输出层权重共享（Qwen3默认开启）
    # 位置编码参数
    'base': 1000000,                  # RoPE位置编码的基数（修复：原10000，匹配HF的Qwen3）
    'rms_norm_epsilon': 1e-6,         # 归一化的极小值（避免除0）
    # 其他参数
    'qkv_bias': False,                # QKV线性层是否有偏置（Qwen3无）
    'scale': 1,                       # 注意力缩放系数
    'max_position': 32768,            # RoPE支持的最大位置长度
    'ffn_bias': False,                # FFN层是否有偏置（修复：Qwen3无）
    'max_num_batch_tokens': 4096,     # 批处理token数上限
    'max_model_length': 1024,          # 模型支持的总序列长度（prompt+生成内容）
    'gpu_memory_utilization': 0.9,    # GPU显存利用率上限（用90%的显存）
    'eos': 151645,                    # 生成结束符的token ID（匹配分词器的eos_token_id）
    'kv_cache_dtype': 'auto', 
}

def main():
    model_path = Path("/home/rejor/Works/CUDA/Qwen3-0.6B").resolve()

    config['model_name_or_path'] = str(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False
    )
    llm = LLM(config=config)
    
    # === 验证代码开始 ===
    print("\n=== Verifying KV Cache Type ===")
    # 获取 model_runner (rank 0)
    runner = llm.model_runner
    # 遍历模型层，找到第一个 Attention 模块
    for name, module in runner.model.named_modules():
        if hasattr(module, 'k_cache'):
            print(f"Layer: {name}")
            print(f"  k_cache dtype: {module.k_cache.dtype}")
            print(f"  k_scale dtype: {module.k_scale.dtype if hasattr(module, 'k_scale') else 'None'}")
            # 如果启用了 INT8，应该输出:
            # k_cache dtype: torch.int8
            # k_scale dtype: torch.float16
            break
    print("================================\n")
    # === 验证代码结束 ===

    # max_tokens is the max number of generated tokens
    # max_model_length is the max total length including prompt
    # both should be set in SamplingParams and help to determine when to stop generation
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024, max_model_length=1024)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs is a dict with 'text' and 'token_ids' keys
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()