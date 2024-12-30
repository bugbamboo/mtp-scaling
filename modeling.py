import torch.nn as nn
import torch
import torch.nn.functional as F

class GatedMLP(nn.Module):
    def __init__(self, input_dim):
        super(GatedMLP, self).__init__()
        self.hidden_dim = input_dim*7//2
        self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        self.gate_fc = nn.Linear(input_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, input_dim)
        self.silu = nn.SiLU()

    def forward(self, x):
        #x is of shape (batch_size, seq_len, input_dim)
        gate = self.gate_fc(x)
        x = self.in_proj(x)
        gate = self.silu(gate)
        x = gate * x
        x = self.out_proj(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        scaled = normed * self.weight
        return scaled

class CausalSelfAttention(nn.Module):
    #implements Multi-Headed Self-Attention
    def __init__(self, input_dim, num_heads):
        super(CausalSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim//num_heads
        self.Q_proj = nn.Linear(input_dim, input_dim) 
        self.K_proj = nn.Linear(input_dim, input_dim)
        self.V_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        #x is of shape (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        q = self.Q_proj(x)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.K_proj(x)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.V_proj(x)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # Create causal mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        attn = attn.masked_fill(mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.out_proj(x)
        return x

class TransformerBlock(nn.Module):
    #implements a single transformer block, with pre-norm, and absolute position embedding (1024 max seq len)
    def __init__(self, input_dim, num_heads, max_seq_len=1024):
        super(TransformerBlock, self).__init__()
        self.norm1 = RMSNorm(input_dim)
        self.attn = CausalSelfAttention(input_dim, num_heads)
        self.norm2 = RMSNorm(input_dim)
        self.mlp = GatedMLP(input_dim)
        # Learned positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x):
        # Add positional embeddings
        x = x + self.pos_emb
        # Regular transformer block operations
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MTP(nn.Module):
    def __init__(self, input_dim, num_heads, max_seq_len):
        super(MTP, self).__init__()
        self.transformer = TransformerBlock(input_dim, num_heads, max_seq_len)
        self.norm1 = RMSNorm(input_dim)
        self.norm2 = RMSNorm(input_dim)
        self.linear = nn.Linear(input_dim*2, input_dim)
    def forward(self, past_hidden_states, token_embedding):
        x = self.norm1(past_hidden_states)
        y = self.norm2(token_embedding)
        x = torch.cat([x, y], dim=-1)
        x = self.linear(x)
        x = self.transformer(x)
        return x


class MTP_LLM(nn.Module):
    def __init__(self,vocab_size, input_dim, num_heads, num_layers, max_seq_len=1024, num_mtps=1):
        super(MTP_LLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, input_dim)
        self.blocks = nn.ModuleList([TransformerBlock(input_dim, num_heads, max_seq_len) for _ in range(num_layers)])
        self.mtps = nn.ModuleList([MTP(input_dim, num_heads, max_seq_len) for _ in range(num_mtps)])
        self.unembed = nn.Linear(input_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        #x is of shape (batch_size, max_seq_len+num_mtps)
        tot_embeddings = self.embed(x)
        x = tot_embeddings[:, :self.max_seq_len]
        for block in self.blocks:
            x = block(x)
        outputs = [self.unembed(x).unsqueeze(1)]
        for mtp in range(len(self.mtps)):
            x = self.mtps[mtp](x, tot_embeddings[:, mtp+1:self.max_seq_len+mtp+1])
            outputs.append(self.unembed(x).unsqueeze(1))
        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    model = MTP_LLM(50257,  192, 4, 8, num_mtps=0)
    print("total params: ", sum(p.numel() for p in model.parameters()))