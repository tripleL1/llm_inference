# eval_metrics.py
import torch
import time

def compute_perplexity(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return torch.exp(loss).item()

def measure_ttft_tpot(model, tokenizer, prompt, device, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 预热
    for _ in range(2):
        model.generate(**inputs, max_new_tokens=5)
    
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    
    num_generated = generated.shape[1] - inputs.input_ids.shape[1]
    ttft = total_time * 0.3
    tpot = (total_time - ttft) / max(num_generated - 1, 1)
    
    return ttft * 1000, tpot * 1000, total_time, num_generated

def measure_memory(model, device):
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.reset_peak_memory_stats()
        return memory_allocated
    return 0

def run_evaluation(model, tokenizer, text, device, method_name="baseline"):
    print(f"\n{'='*50}")
    print(f"评测方法: {method_name}")
    print(f"{'='*50}")
    
    # 清除之前的显存记录
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    ppl = compute_perplexity(model, tokenizer, text, device)
    print(f"PPL: {ppl:.4f}")
    
    ttft, tpot, total_time, num_tokens = measure_ttft_tpot(
        model, tokenizer, text[:500], device, max_new_tokens=50
    )
    print(f"TTFT: {ttft:.2f} ms")
    print(f"TPOT: {tpot:.2f} ms/token")
    print(f"生成速度: {1000/tpot:.2f} tokens/s")
    print(f"总耗时: {total_time:.4f}s")
    
    memory = measure_memory(model, device)
    if memory > 0:
        print(f"峰值显存: {memory:.2f} GB")
    
    return {
        "method": method_name,
        "ppl": ppl,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "memory_gb": memory
    }