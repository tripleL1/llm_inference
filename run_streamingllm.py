import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME, DEVICE, WIKITEXT_PATH, PG19_PATH, MAX_NEW_TOKENS, KV_CACHE_SIZE
from eval_metrics import run_evaluation

class StreamingLLM:
    """
    StreamingLLM: 只保留最近的KV cache
    原理：当序列超过缓存大小时，丢弃最早的token，只保留最近的
    """
    def __init__(self, model, cache_size=512):
        self.model = model
        self.cache_size = cache_size
        
    def compress_kv_cache(self, past_key_values):
        """
        压缩KV cache，只保留最近的token
        past_key_values: tuple of tuples, 每层有 (key, value)
        形状: [batch, num_heads, seq_len, head_dim]
        """
        if past_key_values is None:
            return past_key_values
        
        compressed_cache = []
        for layer_cache in past_key_values:
            k, v = layer_cache
            
            current_len = k.shape[2]
            if current_len > self.cache_size:
                k = k[:, :, -self.cache_size:, :]
                v = v[:, :, -self.cache_size:, :]
            
            compressed_cache.append((k, v))
        
        return tuple(compressed_cache)

def main():
    print(f"加载模型 (StreamingLLM模式)... (设备: {DEVICE})")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.cpu()
    model.eval()
    
    # 在数据集上评测
    results = run_evaluation(model, tokenizer, WIKITEXT_PATH, PG19_PATH, DEVICE, "StreamingLLM")
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n最终结果: {results}")