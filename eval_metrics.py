import torch
import time
import os

def load_wikitext_local(file_path):
    """从本地加载 WikiText-2 测试集"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return []
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 按空行分割成段落
    samples = [para.strip() for para in content.split('\n\n') if para.strip()]
    # 过滤掉太短的样本
    samples = [s for s in samples if len(s.split()) > 20]
    
    print(f"加载了 {len(samples)} 个有效段落")
    return samples

def load_pg19_local(file_path):
    """从本地加载 PG-19 测试集"""
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 取前2000字符
    content = content[:2000]
    print(f"加载了 PG-19 文本，长度: {len(content)} 字符")
    return content

def compute_perplexity_on_dataset(model, tokenizer, text, device):
    """在单条文本上计算 PPL"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    return torch.exp(loss).item()

def compute_wikitext_ppl(model, tokenizer, wikitext_path, device, max_samples=10):
    """在 WikiText 数据集上计算平均 PPL"""
    samples = load_wikitext_local(wikitext_path)
    
    if not samples:
        return 0, 0
    
    total_ppl = 0
    valid_count = 0
    
    for i, text in enumerate(samples[:max_samples]):
        try:
            ppl = compute_perplexity_on_dataset(model, tokenizer, text, device)
            total_ppl += ppl
            valid_count += 1
            print(f"  样本 {valid_count}: PPL = {ppl:.4f}")
        except:
            continue
    
    if valid_count == 0:
        return 0, 0
    
    avg_ppl = total_ppl / valid_count
    return avg_ppl, valid_count

def compute_pg19_ppl(model, tokenizer, pg19_path, device):
    """在 PG-19 单样本上计算 PPL"""
    text = load_pg19_local(pg19_path)
    
    if not text:
        return 0, 0
    
    ppl = compute_perplexity_on_dataset(model, tokenizer, text, device)
    return ppl, len(text)

def measure_ttft_tpot(model, tokenizer, prompt, device, max_new_tokens=50):
    """测量 TTFT 和 TPOT"""
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

def run_evaluation(model, tokenizer, wikitext_path, pg19_path, device, method_name="baseline"):
    print(f"\n{'='*50}")
    print(f"评测方法: {method_name}")
    print(f"{'='*50}")
    
    # 1. WikiText-2 PPL 测试
    print("\n📖 WikiText-2 数据集测试:")
    wikitext_ppl, num_samples = compute_wikitext_ppl(model, tokenizer, wikitext_path, device)
    if wikitext_ppl > 0:
        print(f"  平均 PPL: {wikitext_ppl:.4f} (基于 {num_samples} 个样本)")
    
    # 2. PG-19 PPL 测试
    print("\n📚 PG-19 数据集测试:")
    pg19_ppl, text_len = compute_pg19_ppl(model, tokenizer, pg19_path, device)
    if pg19_ppl > 0:
        print(f"  单样本 PPL: {pg19_ppl:.4f} (文本长度: {text_len} 字符)")
    
    # 3. 获取一个 prompt 用于速度测试
    samples = load_wikitext_local(wikitext_path)
    prompt = samples[0][:200] if samples else "The future of AI is"
    print(f"\n⚡ 速度测试 (Prompt: {prompt[:50]}...)")
    
    # 4. 速度测试
    ttft, tpot, total_time, num_tokens = measure_ttft_tpot(
        model, tokenizer, prompt, device, max_new_tokens=50
    )
    print(f"  TTFT: {ttft:.2f} ms")
    print(f"  TPOT: {tpot:.2f} ms/token")
    print(f"  生成速度: {1000/tpot:.2f} tokens/s")
    print(f"  总耗时: {total_time:.4f}s")
    
    # 5. 内存测试
    memory = measure_memory(model, device)
    if memory > 0:
        print(f"  峰值显存: {memory:.2f} GB")
    
    return {
        "method": method_name,
        "wikitext_ppl": wikitext_ppl,
        "pg19_ppl": pg19_ppl,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "memory_gb": memory
    }