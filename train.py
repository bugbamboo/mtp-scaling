from transformers import AutoTokenizer
import torch
from modeling import MTP_LLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")

