import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj.SCALE_INIT = True

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.mlp[2].SCALE_INIT = True

    def forward(self, x):
        # pre-norm
        x_norm = self.ln1(x)

        # attn
        q = self.q_proj(x_norm).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x_norm).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_proj(x_norm).view(x.size(0), x.size(1), self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim) # (B, T, C)
        attn_output = self.out_proj(attn_output) # (B, T, C)
        x = x + attn_output

        # pre-norm for MLP
        x_norm = self.ln2(x)

        # mlp
        x = x + self.mlp(x_norm)

        return x


class GPT2(nn.Module):
    def __init__(self, block_size, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # weight sharing schema
        self.head.weight = self.token_embedding.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.num_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        assert T <= self.block_size, "Input sequence length exceeds model block size"
        token_embeddings = self.token_embedding(x)  # (B, T, C)
        position_embeddings = self.position_embedding(torch.arange(T, device=x.device))  # (T, C)
        x = token_embeddings + position_embeddings  # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
