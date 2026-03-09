import sys, os
# os.environ["TRITON_INTERPRET"] = "1"  Triton kernel debug时开启的参数

from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # ------- 调度相关参数 -------
    'max_num_sequences': 16,                               # 批处理seq数  
    'max_num_batched_tokens': 4096,                        # 批处理token数
    'max_cached_blocks': 1024,                             # 最大kvcache BLOCK数 (运行时动态计算的, 可以不设置) 
    'max_model_length': 1024,                              # 模型支持的 单个sequence的最大长度（上下文窗口大小）
    'long_prefill_token_threshold': 10,                    # 最大prefill token数 (chunked prefill相关)
    'block_size': 8,                                       # 一个kvcache block 中存储token的数量
    'world_size': 1,                                       # 分布式GPU数量
    'model_name_or_path': '/home/models/Qwen/Qwen3-0.6B',  # 模型路径
    'enforce_eager': True,                                 # 是否开启 CUDAGraph优化 (chunked prefill时暂未实现.. 默认True 不使用CUDAGraph)
    'gpu_memory_utilization': 0.9,                         # GPU可用显存使用率  例如 空闲GPU为8G, 则框架只能使用 8G * 0.9 = 7.2G
    
    # ------- 模型相关参数  (与Qwen3-0.6B config一致) -------
    'vocab_size': 151936,                                  # 词表大小
    'eos': 151645,                                         # eos token(结束的token id)               
    'hidden_size': 1024,                                   # 隐藏层维度
    'num_heads': 16,                                       # q头数  (attention层参数)
    'num_kv_heads': 8,                                     # kv头数 (attention层参数)
    'head_dim': 128,                                       # 头维度 (attention层参数)
    'scale': 1,                                            # 注意力分数缩放值 (attention层参数)
    'qkv_bias': False,                                     # qkv投影矩阵是否使用偏置 (attention层参数)
    'intermediate_size': 3072,                             # MLP up linear 维度  (MLP层参数)
    'ffn_bias': False,                                     # MLP 中的投影矩阵是否使用偏置 (MLP层参数)
    'num_layers': 28,                                      # DecoderLayer(attention + MLP) 层数
    'tie_word_embeddings': True,                           # embedding层和lm_head层是否共享词表权重
    'base': 1000000,                                       # RoPE base
    'max_position': 32768,                                 # RoPE max_position
    'rms_norm_epsilon': 1e-6,                              # RMSNorm epsilon              
}

def main():
    path = os.path.expanduser("/home/models/Qwen/Qwen3-0.6B")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)
    
    # temperature: 采样温度
    # max_tokens: 最大生成长度
    # max_model_length: 模型支持的 单个sequence的最大长度（上下文窗口大小）
    sampling_params = SamplingParams(temperature=1.0, max_tokens=32768, max_model_length=config['max_model_length'])
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30

    # 1. 构建 chat 格式 prompts
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,             # 是否添加生成提示 <|im_start|>assistant\n
            enable_thinking=False                   # 是否开启思考模式 <think>\n\n</think>\n
        )
        for prompt in prompts
    ]

    # 2. 推理
    outputs = llm.generate(prompts, sampling_params)

    # 3. 输出
    generated_texts = outputs['text']
    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()