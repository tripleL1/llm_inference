import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME, DEVICE, WIKITEXT_PATH, PG19_PATH, MAX_NEW_TOKENS
from eval_metrics import run_evaluation

def main():
    print(f"加载基线模型... (设备: {DEVICE})")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # 在数据集上评测
    results = run_evaluation(model, tokenizer, WIKITEXT_PATH, PG19_PATH, DEVICE, "Baseline")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n最终结果: {results}")