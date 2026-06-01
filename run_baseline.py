# run_baseline.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import MODEL_NAME, DEVICE, TEST_TEXT, MAX_NEW_TOKENS
from eval_metrics import run_evaluation

def main():
    print(f"加载基线模型... (设备: {DEVICE})")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"  # 自动分配到GPU
    )
    model.eval()
    
    # 评测
    results = run_evaluation(model, tokenizer, TEST_TEXT, DEVICE, "Baseline")
    
    # 生成示例
    print("\n" + "="*50)
    print("生成示例:")
    print("="*50)
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    print(tokenizer.decode(output[0]))
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"\n结果: {results}")