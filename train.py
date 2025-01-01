

import torch
from torch.utils.data import TensorDataset, DataLoader
from accelerate import Accelerator
from modeling import MTP_LLM
import torch.nn.functional as F
import wandb
import tqdm
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
accelerator = Accelerator(mixed_precision="fp16")
dataset = torch.load('fineweb_edu_10BT.pt')
dataset = TensorDataset(dataset)

# Hyperparameters
VOCAB_SIZE = 50257
BATCH_SIZE = 256
NUM_LAYERS = 12
HIDDEN_SIZE = 1280
NUM_ATTN_HEADS = 10
NUM_MTPS = 2
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
    shuffle=True
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


for step, token_batch in tqdm(enumerate(training_dataloader)):
    table = wandb.Table(columns=["prompt", "generated_text"])
    optimizer.zero_grad()
    tokens = token_batch
    targets = tokens[:, 1:MAX_LENGTH+1]
    mtp_targets = []
    for mtp in range(NUM_MTPS):
        mtp_targets.append(tokens[:, mtp+2:MAX_LENGTH+mtp+2].unsqueeze(1))
    mtp_targets = torch.stack(mtp_targets, dim=1)
    outputs = model(tokens) #outputs is of shape (batch_size, num_mtps+1, max_length, vocab_size)
    with accelerator.autocast():
        lm_output = F.log_softmax(outputs[:, 0, :, :], dim=-1)
        mtp_outputs = F.log_softmax(outputs[:, 1:, :, :],dim=-1)

        lm_loss = -torch.mean(lm_output.view(-1, lm_output.size(-1)).index_select(0, targets.view(-1))[targets != 50256])
        mtp_loss = -torch.mean(mtp_outputs.view(-1, mtp_outputs.size(-1)).index_select(0, mtp_targets.view(-1))[mtp_targets != 50256])

        loss = lm_loss + LAMBDA * mtp_loss
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    loss = lm_loss + LAMBDA * mtp_loss
    if step % 10 == 0:
        wandb.log({"loss": loss.item()})
        wandb.log({"lm_loss": lm_loss.item()})
        wandb.log({"mtp_loss": mtp_loss.item()})
        #compute top5 accuracy of model predictions on the given batch
        indices = torch.topk(lm_output, k=5, dim=-1).indices
        targets_flat = targets.view(-1)
        indices_flat = indices.view(-1, 5)
        
        valid_mask = targets_flat != -100
        valid_targets = targets_flat[valid_mask]
        valid_indices = indices_flat[valid_mask]
        
        correct = (valid_indices == valid_targets.unsqueeze(-1)).any(dim=-1)
        acc = correct.sum().item() / (valid_mask.sum().item() + 1e-8)
        wandb.log({"lm_top5_acc": acc})

        # Calculate top-5 accuracy for each MTP head
        for mtp_idx in range(NUM_MTPS):
            mtp_output = mtp_outputs[:, mtp_idx,:, :]
            mtp_target = mtp_targets[:, mtp_idx]
            
            indices = torch.topk(mtp_output, k=5, dim=-1).indices
            targets_flat = mtp_target.view(-1)
            indices_flat = indices.view(-1, 5)
            
            valid_mask = targets_flat != -100
            valid_targets = targets_flat[valid_mask]
            valid_indices = indices_flat[valid_mask]
            
            correct = (valid_indices == valid_targets.unsqueeze(-1)).any(dim=-1)
            acc = correct.sum().item() / (valid_mask.sum().item() + 1e-8)
            wandb.log({f"mtp{mtp_idx+1}_top5_acc": acc})
    if step %1000 == 0:
        prompts = ["Hello, my name is ", "Jeffery Epstein was a ", "The Eiffel Tower is in ","Wh"]
        generated_texts = []
        for prompt in prompts:
            generated_texts.append(model.generate(prompt))
        table.add_data(prompts, generated_texts)
        wandb.log({f"generated_texts_step_{step}": table})
        model.save_pretrained("mtp-llm-0.5B-1280-10-12-2")




