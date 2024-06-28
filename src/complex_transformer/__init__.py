import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Tuple
import numpy as np
from jaxtyping import Complex, Float, Array, PyTree
import itertools


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.complex(
                torch.randn(out_features, in_features) / np.sqrt(in_features),
                torch.randn(out_features, in_features) / np.sqrt(in_features),
            )
        )
        self.bias = nn.Parameter(
            torch.complex(torch.zeros(out_features), torch.zeros(out_features))
        )

    def forward(
        self, x: Complex[Array, "... in_features"]
    ) -> Complex[Array, "... out_features"]:
        return F.linear(x, self.weight, self.bias)


class ComplexAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = ComplexLinear(dim, dim * 3)
        self.to_out = ComplexLinear(dim, dim)

    def forward(
        self, x: Complex[Array, "b n d"], freqs: Complex[Array, "n d"]
    ) -> Complex[Array, "b n d"]:
        b, n, _ = x.shape
        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(
            qkv, "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3
        )

        q, k = apply_rotary_pos_emb(q, k, freqs)

        dots = torch.matmul(q, k.transpose(-1, -2).conj())
        attn = F.softmax(dots.abs() * self.scale, dim=-1)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def get_rope_freqs(
    dim: int, max_seq_len: int, base: int = 10000
) -> Complex[Array, "max_seq_len dim"]:
    freqs = torch.pow(base, -torch.arange(0, dim, 2).float() / dim)
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.exp(1j * freqs)


def apply_rotary_pos_emb(
    q: Complex[Array, "b h n d"],
    k: Complex[Array, "b h n d"],
    freqs: Complex[Array, "n d"],
) -> Tuple[Complex[Array, "b h n d"], Complex[Array, "b h n d"]]:
    q_rot = q * freqs
    k_rot = k * freqs
    return q_rot, k_rot


class ComplexViT(nn.Module):
    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 7,
        num_classes: int = 10,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 512,
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size**2

        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, dim, dtype=torch.complex64)
        )
        self.patch_to_embedding = ComplexLinear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim, dtype=torch.complex64))

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ComplexAttention(dim, heads),
                        ComplexLinear(dim, mlp_dim),
                        ComplexLinear(mlp_dim, dim),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.to_cls_token = nn.Linear(dim * 2, num_classes)  # Real-valued final layer

        self.freqs = get_rope_freqs(dim // heads, num_patches + 1)

    def forward(self, img: Float[Array, "b c h w"]) -> Float[Array, "b num_classes"]:
        p = self.patch_size

        x = einops.rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x.to(torch.complex64))

        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for attn, ff1, ff2 in self.layers:
            x = attn(x, self.freqs)
            x = ff2(ff1(x))

        x = torch.cat((x[:, 0].real, x[:, 0].imag), dim=-1)
        return self.to_cls_token(x)


# Test the model
model = ComplexViT()
test_input = torch.randn(4, 1, 28, 28)
output = model(test_input)
print(f"ComplexViT output shape: {output.shape}")
assert output.shape == (4, 10), "ComplexViT output shape mismatch"
