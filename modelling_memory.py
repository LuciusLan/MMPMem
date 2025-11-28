import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_up):
        super().__init__()
        # in -> 2*up for gate/value, up -> in
        self.w_in = nn.Linear(d_in, 2 * d_up, bias=True)
        self.w_out = nn.Linear(d_up, d_in, bias=True)

        nn.init.kaiming_uniform_(self.w_in.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_in.bias)
        nn.init.kaiming_uniform_(self.w_out.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x):
        x = self.w_in(x)                     # [B, 2u]
        v, g = x.chunk(2, dim=-1)            # value, gate
        return self.w_out(F.silu(g) * v)     # [B, d_in]

class MemBlock(nn.Module):
    def __init__(self, d_in, d_up, norm='rms'):
        super().__init__()
        self.norm = nn.LayerNorm(d_in) if norm == 'ln' else RMSNorm(d_in)
        self.ff = SwiGLU(d_in, d_up)

    def forward(self, x):
        h = self.norm(x)
        h = self.ff(h)
        return x + h

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return self.scale * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class MemoryMLP(nn.Module):
    """
    Residual MLP memory that takes a hidden state h [B, d]
    and outputs residual logits [B, V] to be ADDED to base logits.
    """
    # TODO:
    # Bind LM head with base LM head

    def __init__(self, d_in=4096, d_up=10240, num_blocks=5, vocab_size=151936,
                 head='dense', head_rank=4096, norm='rms'):
        super().__init__()
        self.blocks = nn.ModuleList([MemBlock(d_in, d_up, norm=norm) for _ in range(num_blocks)])
        if head == 'dense':
            self.head = nn.Linear(d_in, vocab_size, bias=True)
            nn.init.zeros_(self.head.bias)
            nn.init.zeros_(self.head.weight)
        elif head == 'factorized':
            self.proj = nn.Linear(d_in, head_rank, bias=False)
            self.out = nn.Linear(head_rank, vocab_size, bias=True)
        else:
            raise ValueError("head must be 'dense' or 'factorized'")
        self.head_type = head



    def forward(self, h):
        # h: [B, d_in] (e.g., last-token hidden at chosen layer)
        for blk in self.blocks:
            h = blk(h)
        if self.head_type == 'dense':
            return self.head(h)              # residual logits
        else:
            z = self.proj(h)
            return self.out(z)               # residual logits
