import torch
from flash_attn import flash_attn_varlen_qkvpacked_func
from torch import nn


class SparseCoordPE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, coords, spatial_shape):
        shape = coords.new_tensor(spatial_shape).clamp_min(1)
        coords = coords / (shape - 1).clamp_min(1)
        return self.net(coords * 2.0 - 1.0)


class FlashSparseSelfAttention(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.dropout = dropout
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out = nn.Linear(channels, channels, bias=False)

    def forward(self, x, batch_ids):
        order = batch_ids.argsort(stable=True)
        undo = order.argsort()
        x = x[order]

        counts = batch_ids[order].bincount()
        cu_seqlens = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).int()
        max_seqlen = counts.max().item()

        qkv = self.qkv(x).view(x.size(0), 3, self.heads, self.head_dim)
        qkv = qkv.to(torch.float16).contiguous()
        y = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            self.dropout * self.training,
            causal=False,
        )
        y = y.reshape(y.size(0), -1).to(x.dtype)
        return self.out(y)[undo]


class SparseTransformerBlock(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        assert channels % heads == 0

        self.pos = SparseCoordPE(channels)
        self.attn = FlashSparseSelfAttention(channels, heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)

    def forward(self, x):
        features = x.features
        coords = x.indices.long()
        batch_ids = coords[:, 0]
        zyx = coords[:, 1:].float()

        tokens = features + self.pos(zyx, x.spatial_shape)
        attended = self.attn(tokens, batch_ids)
        out = self.attn_norm(features + self.dropout(attended))
        out = self.ffn_norm(out + self.dropout(self.ffn(out)))
        return x.replace_feature(out)


class SparseTransformerContextAdapter(nn.Module):
    def __init__(self, channels, heads, depth, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(
            SparseTransformerBlock(channels, heads, dropout) for _ in range(depth)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
