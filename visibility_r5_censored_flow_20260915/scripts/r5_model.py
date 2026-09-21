import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTokenizer(nn.Module):
    def __init__(self, weather_dim=25, d_model=256, steps=12):
        super().__init__()
        self.proj = nn.Linear(weather_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, steps, d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, weather):
        return self.proj(weather) + self.pos[:, : weather.shape[1]]


class DynamicEncoder(nn.Module):
    def __init__(self, d_model=256, layers=6, heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ffn_dim, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        return self.norm(self.encoder(tokens))


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

    def forward(self, static, vegetation, aerosol):
        static_token = self.static(torch.cat([static, self.veg(vegetation.clamp(0, 31))], dim=-1))
        return torch.stack([static_token, self.aerosol(aerosol)], dim=1)


class ContextEncoder(nn.Module):
    def __init__(self, d_model=256, steps=12, pm_dim=2, fe_dim=36, dropout=0.1):
        super().__init__()
        self.handcrafted = HandcraftedStaticEncoder(d_model, 16, steps * pm_dim + fe_dim, dropout)

    def forward(self, weather_tokens, static, vegetation, pm_sequence, engineered):
        aerosol = torch.cat([pm_sequence.flatten(1), engineered], dim=1)
        return torch.cat([weather_tokens, self.handcrafted(static, vegetation, aerosol)], dim=1)


class TaskQuery(nn.Module):
    def __init__(self, d_model=256, heads=8, ffn_dim=768, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.qnorm = nn.LayerNorm(d_model)
        self.mnorm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )

    def forward(self, memory):
        query = self.query.expand(memory.shape[0], -1, -1)
        query = query + self.attn(
            self.qnorm(query), self.mnorm(memory), self.mnorm(memory), need_weights=False
        )[0]
        return (query + self.ffn(self.ffn_norm(query)))[:, 0]


class ConditionalRationalQuadraticSpline(nn.Module):
    """One monotonic scalar RQS with identity linear tails and analytic inverse/CDF."""

    def __init__(self, context_dim, bins=16, hidden_dim=256, tail_bound=10.0,
                 min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        super().__init__()
        if min_bin_width * bins >= 1 or min_bin_height * bins >= 1:
            raise ValueError("minimum bin size is incompatible with the number of bins")
        self.bins = int(bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)
        self.conditioner = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3 * self.bins - 1),
        )

    def parameters_from_context(self, context):
        raw = self.conditioner(context)
        raw_widths, raw_heights, raw_derivatives = torch.split(
            raw, [self.bins, self.bins, self.bins - 1], dim=-1
        )
        span = 2 * self.tail_bound
        widths = self.min_bin_width + (span - self.min_bin_width * self.bins) * F.softmax(raw_widths, -1)
        heights = self.min_bin_height + (span - self.min_bin_height * self.bins) * F.softmax(raw_heights, -1)
        interior = self.min_derivative + F.softplus(raw_derivatives)
        ones = torch.ones_like(interior[:, :1])
        derivatives = torch.cat([ones, interior, ones], dim=-1)
        left_w = torch.full_like(widths[:, :1], -self.tail_bound)
        right_w = torch.full_like(widths[:, :1], self.tail_bound)
        left_h = torch.full_like(heights[:, :1], -self.tail_bound)
        right_h = torch.full_like(heights[:, :1], self.tail_bound)
        cumwidths = torch.cat([left_w, -self.tail_bound + torch.cumsum(widths, -1)[:, :-1], right_w], -1)
        cumheights = torch.cat([left_h, -self.tail_bound + torch.cumsum(heights, -1)[:, :-1], right_h], -1)
        return widths, heights, derivatives, cumwidths, cumheights

    @staticmethod
    def _gather(values, index):
        return values.gather(1, index[:, None]).squeeze(1)

    def _transform(self, inputs, params, inverse):
        outputs = inputs.clone()
        logabsdet = torch.zeros_like(inputs)
        inside = (inputs > -self.tail_bound) & (inputs < self.tail_bound)
        if not inside.any():
            return outputs, logabsdet

        value = inputs[inside]
        widths, heights, derivatives, cumwidths, cumheights = (tensor[inside] for tensor in params)
        boundaries = cumheights if inverse else cumwidths
        bin_index = (value[:, None] >= boundaries[:, 1:-1]).sum(dim=-1)
        xk = self._gather(cumwidths, bin_index)
        yk = self._gather(cumheights, bin_index)
        wk = self._gather(widths, bin_index)
        hk = self._gather(heights, bin_index)
        dk = self._gather(derivatives, bin_index)
        dk1 = self._gather(derivatives, bin_index + 1)
        delta = hk / wk

        if inverse:
            y_delta = value - yk
            common = dk + dk1 - 2 * delta
            a = y_delta * common + hk * (delta - dk)
            b = hk * dk - y_delta * common
            c = -delta * y_delta
            discriminant = torch.clamp(b.square() - 4 * a * c, min=0.0)
            quadratic_root = (2 * c) / (-b - torch.sqrt(discriminant))
            linear_root = -c / b
            theta = torch.where(a.abs() < 1e-12, linear_root, quadratic_root)
        else:
            theta = (value - xk) / wk

        theta_one_minus = theta * (1 - theta)
        denominator = delta + (dk + dk1 - 2 * delta) * theta_one_minus
        derivative_numerator = delta.square() * (
            dk1 * theta.square() + 2 * delta * theta_one_minus + dk * (1 - theta).square()
        )
        forward_logdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        if inverse:
            transformed = xk + theta * wk
            selected_logdet = -forward_logdet
        else:
            numerator = hk * (delta * theta.square() + dk * theta_one_minus)
            transformed = yk + numerator / denominator
            selected_logdet = forward_logdet
        outputs[inside] = transformed
        logabsdet[inside] = selected_logdet
        return outputs, logabsdet

    def forward_transform(self, base_u, params):
        return self._transform(base_u, params, inverse=False)

    def inverse_transform(self, z, params):
        return self._transform(z, params, inverse=True)

    def log_prob(self, z, params):
        u, log_du_dz = self.inverse_transform(z, params)
        return -0.5 * (u.square() + math.log(2 * math.pi)) + log_du_dz

    def log_cdf(self, z, params):
        u, _ = self.inverse_transform(z, params)
        return torch.special.log_ndtr(u)

    def cdf(self, z, params):
        u, _ = self.inverse_transform(z, params)
        return torch.special.ndtr(u.double())


class R5CensoredFlowModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["d_model"]
        self.dynamic_tokenizer = DynamicTokenizer(cfg["weather_vars"], d, cfg["window_size"])
        self.dynamic_encoder = DynamicEncoder(
            d, cfg["transformer_layers"], cfg["attention_heads"], cfg["transformer_ffn_dim"], cfg["dropout"]
        )
        self.context_encoder = ContextEncoder(
            d, cfg["window_size"], len(cfg["pm_indices"]), cfg["fe_dim"], cfg["dropout"]
        )
        self.flow_query = TaskQuery(d, cfg["attention_heads"], cfg["query_ffn_dim"], cfg["dropout"])
        self.flow = ConditionalRationalQuadraticSpline(
            d, cfg["spline_bins"], cfg["conditioner_hidden_dim"], cfg["spline_tail_bound"],
            cfg["spline_min_bin_width"], cfg["spline_min_bin_height"], cfg["spline_min_derivative"],
        )

    def encode(self, x):
        t, n = self.cfg["window_size"], self.cfg["dyn_vars"]
        dyn = x[:, : t * n].reshape(-1, t, n)
        static_start = t * n
        static = x[:, static_start : static_start + 5]
        vegetation = x[:, static_start + 5].long()
        engineered = x[:, static_start + 6 :]
        weather = self.dynamic_encoder(self.dynamic_tokenizer(dyn[:, :, self.cfg["weather_indices"]]))
        memory = self.context_encoder(
            weather, static, vegetation, dyn[:, :, self.cfg["pm_indices"]], engineered
        )
        return self.flow_query(memory)

    def forward(self, x):
        context = self.encode(x)
        return {"context": context, "flow_params": self.flow.parameters_from_context(context)}

    def event_probabilities(self, params):
        batch = params[0].shape[0]
        device, dtype = params[0].device, params[0].dtype
        z1 = torch.full((batch,), math.log(self.cfg["extinction_constant"]), device=device, dtype=dtype)
        z05 = torch.full((batch,), math.log(self.cfg["extinction_constant"] / 0.5), device=device, dtype=dtype)
        f1, f05 = self.flow.cdf(z1, params), self.flow.cdf(z05, params)
        return torch.stack([1 - f05, f05 - f1, f1], dim=1)

    def observed_visibility_median(self, params):
        base = torch.zeros(params[0].shape[0], device=params[0].device, dtype=params[0].dtype)
        z_median, _ = self.flow.forward_transform(base, params)
        latent_visibility = self.cfg["extinction_constant"] * torch.exp(-z_median.double())
        limit = torch.full_like(latent_visibility, self.cfg["censoring_limit_km"])
        return torch.minimum(latent_visibility, limit)
