import math

import torch,torch.nn.functional as F,torch.nn as nn

def exponential_decay_sum(x, base_lambda, reverse=False):
    iterator = reversed(x) if reverse else x
    total = None
    for i, v in enumerate(iterator, start=1):
        weight = base_lambda**i
        if total is None:
            total = weight * v
        else:
            total = total + weight * v

    return total / len(x)

def contrastive_cluster_loss(token_embed, cluster_emb, temperature=0.1):
    import torch.nn.functional as F
    token_embed = F.normalize(token_embed, dim=-1)
    cluster_emb = F.normalize(cluster_emb, dim=-1)
    logits = token_embed @ cluster_emb.T  # [B, B]
    logits = logits / temperature

    labels = torch.arange(token_embed.size(0), device=token_embed.device)
    loss = F.cross_entropy(logits, labels)
    return loss

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1, p=2)
        rms = norm * norm / x.size(-1)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, max_position_embeddings=4096):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, :, None, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, :, None, :], persistent=False
        )

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(x.device))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, :, None, :]
            sin = emb.sin()[None, :, None, :]
        else:
            cos = self.cos_cached[:, :seq_len, :, :]
            sin = self.sin_cached[:, :seq_len, :, :]
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# def test_apply_rotary_pos_emb():
#     q = torch.randn(16, 16, 128)
#     k = torch.randn(16, 16, 128)
#     r = RotaryEmbedding(dim=128)
#     cos1,sin1 = r(q,seq_len=16)
#     cos2,sin2 = r(k,seq_len=16)
#     qe1,ke1 = apply_rotary_pos_emb(q, k, cos1, sin1)
#     qe2,ke2 = apply_rotary_pos_emb(q, k, cos2, sin2)
#     print( qe1==qe2 )
#     print( ke1==ke2 )

class LlamaMLP(torch.nn.Module):
    #思考
    def __init__(self, hidden_size, multiple_of=256, dropout=0.0):
        super().__init__()
        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        self.gate_proj = torch.nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = torch.nn.Linear(inner_size, hidden_size, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)

class LlamaAttention(nn.Module):
    #看句子
    #计算qkv之后的语义向量
    #依赖:apply_rotary_pos_emb,RotaryEmbedding
    def __init__(
        self, hidden_size, num_heads, dropout=0.0, max_position_embeddings=4096
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim, max_position_embeddings=max_position_embeddings
        )
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

class LlamaDecoderLayer(nn.Module):
    #看句子+思考
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size, num_heads, dropout=dropout)
        self.mlp = LlamaMLP(hidden_size, dropout=dropout)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = residual + self.self_attn(x, attention_mask=attention_mask)

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x

class LlamaDecoder(nn.Module):
    #最终版
    def __init__(self, hidden_size, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(hidden_size, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def _build_causal_mask(self, seq_len, device, dtype):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        attn_mask = self._build_causal_mask(seq_len, x.device, x.dtype)
        for layer in self.layers:
            print(attn_mask)
            x = layer(x, attention_mask=attn_mask)
        return self.norm(x)

