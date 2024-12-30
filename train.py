

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from accelerate import Accelerator
import random
from modeling import MTP_LLM
import torch.nn.functional as F
import wandb
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
accelerator = Accelerator(dtype = 'bfloat16')
class SlimPajamaDataset(Dataset):
    """
    A simple Dataset that yields one raw text sample per item.
    """
    def __init__(self, split="train/chunk1"):
        # You can set streaming=True if desired, but then __len__ won't work.
        self.dataset = load_dataset("cerebras/SlimPajama-627B",
                                    split=split,
                                    streaming=False)
        
        # Convert to an in-memory dataset if it's not too large
        # (If very large, consider streaming or memory-mapped approaches)
        self.dataset = self.dataset.shuffle(seed=42).select(range(len(self.dataset)))  
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Return just the raw text; the collate_fn will handle tokenization.
        return self.dataset[idx]["text"]

def slim_pajama_collate_fn(batch, tokenizer, max_length):
    """
    Collate function that:
      1) Tokenizes a list of raw text samples (the 'batch').
      2) For each sample, if it's longer than max_length, randomly
         takes a contiguous chunk of size max_length.
      3) If it's shorter, pads with -100 up to max_length.
      4) Returns a tensor of shape (batch_size, max_length).
    """
    
    # batch is a list of raw strings
    # Tokenize all at once (faster than tokenizing individually):
    encoding = tokenizer(batch, 
                         truncation=False,
                         padding=False,
                         add_special_tokens=False)
    # encoding["input_ids"] is now a list of lists of token IDs
    all_chunked_ids = []
    
    for ids in encoding["input_ids"]:
        if len(ids) >= max_length:
            # Randomly select a contiguous chunk of length max_length
            start_idx = random.randint(0, len(ids) - max_length)
            chunk = ids[start_idx : start_idx + max_length]
        else:
            # If too short, pad with -100
            chunk = ids + [-100] * (max_length - len(ids))
        all_chunked_ids.append(chunk)
    
    # Convert list of lists into a PyTorch tensor
    # Shape: (batch_size, max_length)
    return torch.tensor(all_chunked_ids, dtype=torch.long)



# Instantiate the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Create the dataset
dataset = SlimPajamaDataset(split="train/chunk1")


# Hyperparameters
VOCAB_SIZE = tokenizer.vocab_size
BATCH_SIZE = 8
NUM_LAYERS = 4
HIDDEN_SIZE = 192
NUM_ATTN_HEADS = 4
NUM_MTPS = 1
MAX_LENGTH = 1024
LAMBDA = 0.1

wandb.init(project="mtp-llm", 
        config={"VOCAB_SIZE": VOCAB_SIZE, 
                "HIDDEN_SIZE": HIDDEN_SIZE, 
                "NUM_ATTN_HEADS": NUM_ATTN_HEADS, 
                "NUM_LAYERS": NUM_LAYERS, 
                "NUM_MTPS": NUM_MTPS, 
                "BATCH_SIZE": BATCH_SIZE})

model = MTP_LLM(VOCAB_SIZE, HIDDEN_SIZE, NUM_ATTN_HEADS, NUM_LAYERS, MAX_LENGTH, NUM_MTPS)

# Create the DataLoader with a custom collate_fn
training_dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: slim_pajama_collate_fn(batch, tokenizer, MAX_LENGTH+NUM_MTPS+1)
)

max_lr = 6e-4
min_lr = 6e-5
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# Calculate number of training steps
total_steps = len(training_dataloader)
warmup_steps = int(0.1 * total_steps) # 10% warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=weight_decay, betas=(beta1, beta2))
# Create learning rate scheduler with linear warmup and decay
def lr_lambda(current_step):
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # Linear decay from max_lr to min_lr over remaining steps
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress) * (max_lr - min_lr) / max_lr + min_lr / max_lr

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)


for step, token_batch in enumerate(training_dataloader):
    optimizer.zero_grad()
    tokens = token_batch
    targets = tokens[:, 1:MAX_LENGTH+1]
    mtp_targets = []
    for mtp in range(NUM_MTPS):
        mtp_targets.append(tokens[:, mtp+2:MAX_LENGTH+mtp+2].unsqueeze(1))
    mtp_targets = torch.stack(mtp_targets, dim=1)
    outputs = model(tokens) #outputs is of shape (batch_size, num_mtps+1, max_length, vocab_size)

    lm_output = F.log_softmax(outputs[:, 0, :, :], dim=-1)
    mtp_outputs = F.log_softmax(outputs[:, 1:, :, :],dim=-1)

    lm_loss = -torch.mean(lm_output.view(-1, lm_output.size(-1)).index_select(0, targets.view(-1))[targets != -100])
    mtp_loss = -torch.mean(mtp_outputs.view(-1, mtp_outputs.size(-1)).index_select(0, mtp_targets.view(-1))[mtp_targets != -100])

    
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    loss = lm_loss + LAMBDA * mtp_loss
    if step % 10 == 0:
        wandb.log({"loss": loss.item()})
        wandb.log({"lm_loss": lm_loss.item()})
        wandb.log({"mtp_loss": mtp_loss.item()})




