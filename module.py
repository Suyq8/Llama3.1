import math
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    rope_theta: float,
    use_scaled_rope: bool,
    device: str = None,
) -> torch.Tensor:
    """
    Precompute the frequencies for RoPE.
    """
    assert dim % 2 == 0

    position = torch.arange(0, max_seq_len, device=device, dtype=torch.float)  # (n,)
    inv_theta = 1.0 / torch.pow(
        rope_theta, torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim
    )  # (dim//2,)

    if use_scaled_rope:
        inv_theta = apply_scaling(inv_theta)

    freqs_cis = torch.einsum("i,j->ij", position, inv_theta)  # (n, dim//2)
    freqs_cis = torch.polar(
        torch.ones_like(freqs_cis), freqs_cis
    )  # (n, dim//2) complex

    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to the input tensor.
    (x0+x1i) * (cos(theta) + sin(theta)i) = x0cos(theta) - x1sin(theta) + (x0sin(theta) + x1cos(theta))i
    """
    assert x.shape[-1] == freqs_cis.shape[-1] * 2

    output = x.float().view(*x.shape[:-1], -1, 2)  # (b, seq_len, n_head, dim//2, 2)
    output = torch.view_as_real(
        torch.einsum(
            "b n c d ..., n d -> b n c d", torch.view_as_complex(output), freqs_cis
        )
    )  # (b, seq_len, n_head, dim//2, 1), (seq_len, dim//2) -> (b, seq_len, n_head, dim//2, 2)

    return output.flatten(3).type_as(x)  # (b, seq_len, n_head, dim)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat the key and value tensors to match the head of query tensor.
    """
    if n_rep == 1:
        return x

    b, n, n_kv_head, head_dim = x.shape
    x = x.unsqueeze(3).expand(b, n, n_kv_head, n_rep, head_dim)
    return x.reshape(b, n, n_kv_head * n_rep, head_dim)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class KVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.register_buffer(
            "k",
            torch.zeros(
                batch_size, max_seq_len, n_kv_head, head_dim, dtype=dtype, device=device
            ),
        )
        self.register_buffer(
            "v",
            torch.zeros(
                batch_size, max_seq_len, n_kv_head, head_dim, dtype=dtype, device=device
            ),
        )

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int) -> torch.Tensor:
        b, n, _, _ = k.shape
        self.k[:b, start_pos : start_pos + n] = k
        self.v[:b, start_pos : start_pos + n] = v
        return self.k[:b, : n + start_pos], self.v[:b, : n + start_pos]

    def forward(self, x: torch.Tensor):
        pass


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_kv_head: Optional[int],
        use_flash: bool,
        kv_cache: Optional[KVCache] = None,
    ):
        super().__init__()
        self.n_kv_head = n_head if n_kv_head is None else n_kv_head
        self.n_head = n_head
        self.n_rep = (
            self.n_head // self.n_kv_head
        )  # the times to repeat the head of kv to match the head of q
        self.head_dim = dim // n_head
        self.use_flash = use_flash

        self.wq = nn.Linear(dim, n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(n_head * self.head_dim, dim, bias=False)

        self.kv_cache = kv_cache

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        b, n, _ = x.shape  # (b, n, dim)

        q = self.wq(x).view(b, n, self.n_head, self.head_dim)
        k = self.wk(x).view(b, n, self.n_kv_head, self.head_dim)
        v = self.wv(x).view(b, n, self.n_kv_head, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(
                k, v, start_pos
            )  # (b, n+start_pos, n_kv_head, head_dim)

        k = repeat_kv(k, self.n_rep)  # (b, n, n_head, head_dim)
        v = repeat_kv(v, self.n_rep)  # (b, n, n_head, head_dim)

        q = q.transpose(1, 2)  # (b, n_head, n, head_dim)
        k = k.transpose(1, 2)  # (b, n_head, n, head_dim)
        v = v.transpose(1, 2)  # (b, n_head, n, head_dim)

        # (b, n_head, n, head_dim)
        if self.use_flash:
            output = F.scaled_dot_product_attention(q, k, v, mask)
        else:
            scores = q@k.transpose(2, 3)/math.sqrt(self.head_dim)
            if mask is not None:
                scores += mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            output = scores@v

        # (b, n, n_head*head_dim ~ dim)
        output = output.transpose(1, 2).reshape(b, n, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim // 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        output = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: Namespace, kv_cache: Optional[KVCache] = None):
        super().__init__()
        self.n_head = args.n_head
        self.head_dim = args.dim // args.n_head
        self.attention = Attention(args.dim, args.n_head, args.n_kv_head, kv_cache, args.use_flash)
        self.feed_forward = FeedForward(
            args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier
        )

        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        output = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        output += self.feed_forward(self.ffn_norm(output))
        return output


class Transformer(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.vocab_size: int = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList(TransformerBlock(args) for _ in range(args.n_layer))

        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_head,
            args.max_seq_len * 2,
            args.rope_theta,
            args.use_scaled_rope,
            device=args.device,
        )

    def forward_inference(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        _, n = x.shape

        emb = self.tok_embeddings(x)  # (b, n, dim)
        freqs_cis = self.freqs_cis[start_pos : start_pos + n]

        mask = None
        if n > 1:
            # The mask is used to prevent the model from attending to the future tokens
            mask = torch.full((n, n), float("-inf"), device=x.device)
            mask = torch.triu(mask, 1)
            # When using kv cache, the positions before start_pos are already calculated and should not be masked
            mask = torch.hstack(
                [torch.zeros((n, start_pos), device=x.device), mask]
            ).to(torch.float)

        for layer in self.layers:
            emb = layer(emb, start_pos, freqs_cis, mask)

        output = self.output(self.norm(emb)).float()

        return output
    
    def forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, n = x.shape

        emb = self.tok_embeddings(x)  # (b, n, dim)
        freqs_cis = self.freqs_cis[: n]

        mask = torch.full((n, n), float("-inf"), device=x.device)
        mask = torch.triu(mask, 1).type_as(emb)

        for layer in self.layers:
            emb = layer(emb, -1, freqs_cis, mask)
        
        output = self.output(self.norm(emb)).float()

        loss = F.cross_entropy(output.transpose(1, 2), y, reduction="mean")

        return loss