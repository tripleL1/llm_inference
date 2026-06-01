import torch
MODEL_NAME = "EleutherAI/pythia-70m"
DEVICE = "cpu"
print(f"使用设备: {DEVICE}")
WIKITEXT_PATH = "wiki.txt"
PG19_PATH = "pg19.txt"
MAX_NEW_TOKENS = 50
KV_CACHE_SIZE = 30
RECENT_TOKENS = 256