import sys, os
from pathlib import Path
import torch
import torch.distributed as dist

# 设置默认精度
torch.set_default_dtype(torch.bfloat16) 

sys.path.insert(0, str(Path(__file__).parent / "src"))
from myvllm.engine.llm_engine import LLMEngine as LLM

config = {
    # === 模型结构参数 (必须完整) ===
    'vocab_size': 151936, 'hidden_size': 1024, 'num_heads': 16,
    'head_dim': 128, 'num_kv_heads': 8, 'intermediate_size': 3072,
    'num_layers': 28, 'tie_word_embeddings': True, 'base': 1000000,
    'rms_norm_epsilon': 1e-6, 'qkv_bias': False, 'scale': 1,
    'max_position': 32768, 'ffn_bias': False,
    'block_size': 256,          # <--- 之前漏了这个，补上
    
    # === 测试配置 ===
    'max_num_sequences': 1,
    'max_num_batch_tokens': 250000, # 配合大长度
    'world_size': 1,
    'enforce_eager': True,
    
    # 设置一个巨大的长度，让引擎去撞显存的“天花板”
    'max_model_length': 250000, 
    
    # 按你的要求，只用 20% 显存
    'gpu_memory_utilization': 0.2, 
    
    # 切换这里测试 'int8' vs 'auto' (bf16)
    'kv_cache_dtype': 'int8', 
    
    'model_name_or_path': '/home/rejor/Works/CUDA/Qwen3-0.6B',
    'local_model_path': '/home/rejor/Works/CUDA/Qwen3-0.6B',
}

def main():
    print(f"=== CAPACITY TEST: Mode={config['kv_cache_dtype']} ===")
    
    try:
        # 初始化引擎
        llm = LLM(config=config)
        
        # 获取结果
        num_blocks = llm.model_runner.num_available_kv_blocks
        max_tokens = num_blocks * config['block_size']
        mem_used = torch.cuda.memory_allocated() / (1024**3)
        
        print("\n" + "="*40)
        print("SUCCESS: ENGINE INITIALIZED")
        print("="*40)
        print(f"Mode:            {config['kv_cache_dtype']}")
        print(f"GPU Memory Used: {mem_used:.2f} GB")
        print(f"Max Context:     {max_tokens} tokens")
        print(f"Blocks Allocated:{num_blocks}")
        print("="*40)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()