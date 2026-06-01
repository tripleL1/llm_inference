# config.py
import torch

MODEL_NAME = "EleutherAI/pythia-70m"

# 强制使用CPU
DEVICE = "cpu"
print(f"使用设备: {DEVICE}")

# 测试文本
TEST_TEXT = """The history of artificial intelligence began in the 1950s, 
when researchers first started exploring the possibility of creating machines 
that could think and learn like humans. Over the decades, AI has evolved 
from simple rule-based systems to complex neural networks capable of 
recognizing patterns, understanding language, and even generating creative 
content. Today, large language models represent the state of the art in AI,
demonstrating remarkable abilities in tasks ranging from translation to 
code generation. However, these models also face significant challenges,
including high computational costs, memory constraints during long-sequence 
processing, and the need for more efficient inference methods."""

MAX_NEW_TOKENS = 50

# StreamingLLM参数
KV_CACHE_SIZE = 512
RECENT_TOKENS = 256