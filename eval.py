import torch
from modeling import MTP_LLM
from transformers import AutoTokenizer
import datasets

if __name__ == "__main__":
    model = MTP_LLM(128, 16, 4, 4, 16, 5)
    x = torch.randint(0, 100, (10, 16+5+1))
    print(model(x).shape)