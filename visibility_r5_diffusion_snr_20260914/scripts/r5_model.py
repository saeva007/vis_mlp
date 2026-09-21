import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DynamicTokenizer(nn.Module):
    """Point weather sequence -> temporally positioned weather tokens."""

    def __init__(self, weather_dim=25, d_model=256, steps=12):
        super().__init__()
        self.proj = nn.Linear(weather_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, steps, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        return self.proj(weather) + self.pos[:, : weather.shape[1]]


class DynamicEncoder(nn.Module):
    """Preserves the complete token sequence: [B,T,D] -> [B,T,D]."""

    def __init__(self, d_model=256, layers=6, heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ffn_dim, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(tokens))


class PatchDynamicTokenizer(nn.Module):
    """Reserved interface: future time x spatial cells -> weather-token memory."""

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Patch input is intentionally disabled in the R5 point baseline")


class AlphaEarthEncoder(nn.Module):
    """Reserved 64-d embedding -> one context token; not instantiated in this round."""

    def __init__(self, d_model=256):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.proj(embedding).unsqueeze(1)


class HandcraftedStaticEncoder(nn.Module):
    def __init__(self, d_model=256, veg_dim=16, aerosol_dim=60, dropout=0.1):
        super().__init__()
        self.veg = nn.Embedding(32, veg_dim)
        self.static = nn.Sequential(
            nn.Linear(5 + veg_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.aerosol = nn.Sequential(
            nn.Linear(aerosol_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )

    def forward(self, static: torch.Tensor, vegetation: torch.Tensor, aerosol: torch.Tensor) -> torch.Tensor:
        veg = self.veg(vegetation.clamp(0, 31))
        static_token = self.static(torch.cat([static, veg], dim=-1))
        aerosol_token = self.aerosol(aerosol)
        return torch.stack([static_token, aerosol_token], dim=1)


class ContextEncoder(nn.Module):
    def __init__(self, d_model=256, steps=12, pm_dim=2, fe_dim=36, dropout=0.1):
        super().__init__()
        self.handcrafted = HandcraftedStaticEncoder(d_model, 16, steps * pm_dim + fe_dim, dropout)

    def forward(self, weather_tokens, static, vegetation, pm_sequence, engineered, alphaearth=None):
        aerosol = torch.cat([pm_sequence.flatten(1), engineered], dim=1)
        tokens = [weather_tokens, self.handcrafted(static, vegetation, aerosol)]
        if alphaearth is not None:
            tokens.append(alphaearth)
        return torch.cat(tokens, dim=1)


class TaskQuery(nn.Module):
    def __init__(self, d_model=256, heads=8, ffn_dim=768, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.qnorm = nn.LayerNorm(d_model)
        self.mnorm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(memory.shape[0], -1, -1)
        q = q + self.attn(self.qnorm(q), self.mnorm(memory), self.mnorm(memory), need_weights=False)[0]
        q = q + self.ffn(self.ffn_norm(q))
        return q[:, 0]


class GateHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GaussianHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 256), nn.SiLU(), nn.Linear(256, 2))

    def forward(self, x):
        mu, log_sigma = self.net(x).chunk(2, dim=-1)
        return mu.squeeze(-1), log_sigma.squeeze(-1).clamp(-5.0, 3.0)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, t):
        half = self.d_model // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        emb = torch.cat([torch.sin(t.float()[:, None] * freq), torch.cos(t.float()[:, None] * freq)], dim=1)
        return self.mlp(emb)


class CrossAttentionResidualBlock(nn.Module):
    def __init__(self, d_model=256, heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.qnorm = nn.LayerNorm(d_model)
        self.mnorm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.fnorm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model))

    def forward(self, q, memory):
        q = q + self.attn(self.qnorm(q), self.mnorm(memory), self.mnorm(memory), need_weights=False)[0]
        return q + self.ffn(self.fnorm(q))


class DiffusionHead(nn.Module):
    def __init__(self, d_model=256, heads=8, blocks=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.z_proj = nn.Linear(1, d_model)
        self.time = SinusoidalTimeEmbedding(d_model)
        self.blocks = nn.ModuleList([CrossAttentionResidualBlock(d_model, heads, ffn_dim, dropout) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, z_t, timestep, h_cont, memory):
        q = self.z_proj(z_t[:, None, None]) + self.time(timestep)[:, None] + h_cont[:, None]
        for block in self.blocks:
            q = block(q, memory)
        return self.out(q[:, 0]).squeeze(-1)


class R5Model(nn.Module):
    def __init__(self, cfg: Dict, route: str):
        super().__init__()
        self.cfg, self.route = cfg, route
        d = cfg["d_model"]
        self.dynamic_tokenizer = DynamicTokenizer(cfg["weather_vars"], d, cfg["window_size"])
        self.dynamic_encoder = DynamicEncoder(
            d, cfg["transformer_layers"], cfg["attention_heads"], cfg["transformer_ffn_dim"], cfg["dropout"])
        self.context_encoder = ContextEncoder(d, cfg["window_size"], len(cfg["pm_indices"]), cfg["fe_dim"], cfg["dropout"])
        self.gate_query = TaskQuery(d, cfg["attention_heads"], cfg["query_ffn_dim"], cfg["dropout"])
        self.continuous_query = TaskQuery(d, cfg["attention_heads"], cfg["query_ffn_dim"], cfg["dropout"])
        self.gate_head = GateHead(d)
        self.continuous_head = GaussianHead(d) if route == "R5-Gaussian" else DiffusionHead(
            d, cfg["attention_heads"], cfg["diffusion_decoder_blocks"], cfg["diffusion_ffn_dim"], cfg["dropout"]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, n = self.cfg["window_size"], self.cfg["dyn_vars"]
        dyn = x[:, : t * n].reshape(-1, t, n)
        pm_idx = self.cfg["pm_indices"]
        weather_idx = self.cfg["weather_indices"]
        static_start = t * n
        static = x[:, static_start : static_start + 5]
        vegetation = x[:, static_start + 5].long()
        engineered = x[:, static_start + 6 :]
        weather_tokens = self.dynamic_encoder(self.dynamic_tokenizer(dyn[:, :, weather_idx]))
        memory = self.context_encoder(weather_tokens, static, vegetation, dyn[:, :, pm_idx], engineered)
        return memory, self.gate_query(memory), self.continuous_query(memory)

    def forward(self, x: torch.Tensor, z_t: Optional[torch.Tensor] = None, timestep: Optional[torch.Tensor] = None):
        memory, h_gate, h_cont = self.encode(x)
        out = {"gate_logit": self.gate_head(h_gate), "memory": memory, "h_cont": h_cont}
        if self.route == "R5-Gaussian":
            out["mu_z"], out["log_sigma_z"] = self.continuous_head(h_cont)
        elif z_t is not None and timestep is not None:
            out["v_pred"] = self.continuous_head(z_t, timestep, h_cont, memory)
        return out

    def predict_v(self, z_t, timestep, h_cont, memory):
        return self.continuous_head(z_t, timestep, h_cont, memory)
