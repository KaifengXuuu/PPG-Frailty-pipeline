from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShapeletBundle:
    info: np.ndarray
    shapelets: List[np.ndarray]
    discovery_indices: np.ndarray


@contextlib.contextmanager
def _suppress_output(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class ShapeFormerAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.scale = float(emb_size) ** -0.5
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)
        self.attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        self.attn = attn
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.to_out(out)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)].unsqueeze(0))


class ShapeBlock(nn.Module):
    def __init__(
        self,
        shapelet_info: Sequence[float],
        shapelet: np.ndarray,
        shape_embed_dim: int,
        len_ts: int,
        search_window: int,
        norm: float = 1000.0,
        max_ci: float = 3.0,
    ) -> None:
        super().__init__()
        shapelet = np.asarray(shapelet, dtype=np.float32).ravel()
        if shapelet.size < 2:
            shapelet = np.pad(shapelet, (0, max(0, 2 - shapelet.size)), mode="constant")
        self.dim = int(shapelet_info[5])
        self.shapelet = nn.Parameter(torch.tensor(shapelet, dtype=torch.float32), requires_grad=True)
        self.kernel_size = int(shapelet.size)
        self.norm = float(norm)
        self.max_ci = float(max_ci)

        start = int(shapelet_info[1])
        end = int(shapelet_info[2])
        self.start_position = max(0, start - int(search_window))
        self.end_position = min(int(len_ts), end + int(search_window))

        self.l1 = nn.Linear(self.kernel_size, shape_embed_dim)
        self.l2 = nn.Linear(self.kernel_size, shape_embed_dim)
        ci = np.sqrt(np.sum((shapelet[1:] - shapelet[:-1]) ** 2)) + 1.0 / self.norm
        self.register_buffer("ci_shapelet", torch.tensor(float(ci), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim >= x.size(1):
            return x.new_zeros((x.size(0), 1, self.l1.out_features))

        pis = x[:, self.dim, self.start_position : self.end_position]
        if pis.size(1) < self.kernel_size:
            return x.new_zeros((x.size(0), 1, self.l1.out_features))

        ci_pis = torch.square(pis[:, 1:] - pis[:, :-1])
        windows = pis.unfold(1, self.kernel_size, 1).contiguous()
        flat_windows = windows.view(-1, self.kernel_size)

        if self.kernel_size > 1:
            ci_windows = ci_pis.unfold(1, self.kernel_size - 1, 1).contiguous()
            ci_windows = ci_windows.view(-1, self.kernel_size - 1)
            ci_windows = torch.sum(ci_windows, dim=1) + 1.0 / self.norm
            ci_shapelet = self.ci_shapelet.to(x.device)
            ci_dist = torch.maximum(ci_windows, ci_shapelet) / torch.minimum(ci_windows, ci_shapelet)
            ci_dist = torch.where(torch.isfinite(ci_dist), ci_dist, torch.ones_like(ci_dist))
            ci_dist = torch.clamp(ci_dist, max=self.max_ci)
        else:
            ci_dist = torch.ones(flat_windows.size(0), device=x.device)

        dist = torch.sum(torch.square(flat_windows - self.shapelet), dim=1)
        dist = dist * ci_dist
        dist = dist / max(1, self.shapelet.size(-1))
        dist = torch.where(torch.isfinite(dist), dist, torch.ones_like(dist))
        dist = dist.view(x.size(0), -1)

        best = torch.argmin(dist, dim=1)
        selected = windows[torch.arange(x.size(0), device=x.device), best]
        out = self.l1(selected) - self.l2(self.shapelet.unsqueeze(0))
        return out.unsqueeze(1)


class PortedShapeFormer(nn.Module):
    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        n_classes: int,
        shapelets_info: np.ndarray,
        shapelets: Sequence[np.ndarray],
        len_w: int = 64,
        local_embed_dim: int = 48,
        shape_embed_dim: int = 128,
        dim_ff: int = 256,
        num_heads: int = 4,
        dropout: float = 0.30,
        shapelet_search_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(shapelets) == 0:
            raise ValueError("ShapeFormer requires at least one discovered shapelet")
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)

        self.len_w = max(4, int(len_w))
        self.pad_w = self.len_w - self.seq_len % self.len_w
        self.pad_w = 0 if self.pad_w == self.len_w else self.pad_w
        self.height = self.n_channels
        self.weight = int(math.ceil(self.seq_len / self.len_w))
        self.local_layer = nn.Linear(self.len_w, local_embed_dim)
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, local_embed_dim, kernel_size=(1, 8), padding="same"),
            nn.BatchNorm2d(local_embed_dim),
            nn.GELU(),
        )
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(local_embed_dim, local_embed_dim, kernel_size=(self.n_channels, 1), padding="valid"),
            nn.BatchNorm2d(local_embed_dim),
            nn.GELU(),
        )
        self.Fix_Position = LearnablePositionalEncoding(
            local_embed_dim,
            dropout=dropout,
            max_len=self.seq_len,
        )
        list_d: List[int] = []
        list_p: List[int] = []
        for d in range(self.height):
            for p in range(self.weight):
                list_d.append(d)
                list_p.append(p)
        list_ed = position_embedding(torch.tensor(list_d, dtype=torch.long))
        list_ep = position_embedding(torch.tensor(list_p, dtype=torch.long))
        local_pos_embedding = torch.cat((list_ed, list_ep), dim=1)
        self.register_buffer("local_pos_embedding", local_pos_embedding)
        self.local_pos_layer = nn.Linear(local_pos_embedding.shape[-1], local_embed_dim)
        self.local_ln1 = nn.LayerNorm(local_embed_dim, eps=1e-5)
        self.local_ln2 = nn.LayerNorm(local_embed_dim, eps=1e-5)
        self.local_attention_layer = ShapeFormerAttention(local_embed_dim, num_heads, dropout)
        self.local_ff = nn.Sequential(
            nn.Linear(local_embed_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, local_embed_dim),
            nn.Dropout(dropout),
        )
        self.local_gap = nn.AdaptiveAvgPool1d(1)
        self.local_flatten = nn.Flatten()

        info = np.asarray(shapelets_info, dtype=np.float32)
        self.sw = nn.Parameter(torch.tensor(info[:, 3], dtype=torch.float32), requires_grad=True)

        search_window = int(shapelet_search_window) if shapelet_search_window is not None else max(1, self.len_w // 2)
        self.shape_blocks = nn.ModuleList(
            [
                ShapeBlock(
                    shapelet_info=info[i],
                    shapelet=shapelets[i],
                    shape_embed_dim=shape_embed_dim,
                    len_ts=self.seq_len,
                    search_window=search_window,
                )
                for i in range(len(shapelets))
            ]
        )
        position = torch.tensor(info[:, [5, 1, 2]], dtype=torch.float32)
        d_position = position_embedding(position[:, 0])
        s_position = position_embedding(position[:, 1])
        e_position = position_embedding(position[:, 2])
        self.register_buffer("d_position", d_position)
        self.register_buffer("s_position", s_position)
        self.register_buffer("e_position", e_position)
        self.d_pos_embedding = nn.Linear(d_position.shape[1], shape_embed_dim)
        self.s_pos_embedding = nn.Linear(s_position.shape[1], shape_embed_dim)
        self.e_pos_embedding = nn.Linear(e_position.shape[1], shape_embed_dim)
        self.shape_ln1 = nn.LayerNorm(shape_embed_dim, eps=1e-5)
        self.shape_ln2 = nn.LayerNorm(shape_embed_dim, eps=1e-5)
        self.shape_attention_layer = ShapeFormerAttention(shape_embed_dim, num_heads, dropout)
        self.shape_ff = nn.Sequential(
            nn.Linear(shape_embed_dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, shape_embed_dim),
            nn.Dropout(dropout),
        )

        self.out = nn.Linear(shape_embed_dim + local_embed_dim, n_classes)
        self.feature_dim = int(shape_embed_dim + local_embed_dim)
        self.out2 = nn.Linear(shape_embed_dim, n_classes)
        self.local_merge = nn.Linear(local_embed_dim, int(local_embed_dim / 2))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        local_x = x.unsqueeze(1)
        local_x = self.embed_layer(local_x)
        local_x = self.embed_layer2(local_x).squeeze(2)
        local_x = local_x.permute(0, 2, 1)
        x_src_pos = self.Fix_Position(local_x)
        local_att = local_x + self.local_attention_layer(x_src_pos)
        local_att = self.local_ln1(local_att)
        local_out = local_att + self.local_ff(local_att)
        local_out = self.local_ln2(local_out)
        local_out = local_out.permute(0, 2, 1)
        local_out = self.local_gap(local_out)
        local_out = self.local_flatten(local_out)

        global_x = torch.cat([block(x) for block in self.shape_blocks], dim=1)
        d_pos = self.d_position.to(x.device).repeat(x.shape[0], 1, 1)
        s_pos = self.s_position.to(x.device).repeat(x.shape[0], 1, 1)
        e_pos = self.e_position.to(x.device).repeat(x.shape[0], 1, 1)
        d_pos_emb = self.d_pos_embedding(d_pos)
        s_pos_emb = self.s_pos_embedding(s_pos)
        e_pos_emb = self.e_pos_embedding(e_pos)

        global_x = global_x + d_pos_emb + s_pos_emb + e_pos_emb
        global_att = global_x + self.shape_attention_layer(global_x)
        global_att = global_att * self.sw.unsqueeze(0).unsqueeze(2)
        global_att = self.shape_ln1(global_att)
        global_out = global_att + self.shape_ff(global_att)
        global_out = self.shape_ln2(global_out)
        global_out = global_out * self.sw.unsqueeze(0).unsqueeze(2)
        global_out = global_out[:, 0, :]

        return torch.cat((global_out, local_out), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.forward_features(x))


def position_embedding(position_list: torch.Tensor) -> torch.Tensor:
    position_list = position_list.to(dtype=torch.long)
    max_d = int(position_list.max().item()) + 1 if position_list.numel() else 1
    identity_matrix = torch.eye(max_d)
    return identity_matrix[position_list]


def _balanced_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y), dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // max(1, len(classes)))
    chosen: List[int] = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())

    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y)), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        if remaining.size:
            fill = min(max_samples - len(chosen), remaining.size)
            chosen.extend(rng.choice(remaining, size=fill, replace=False).tolist())

    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _overlaps_existing(start: int, end: int, occupied: List[Tuple[int, int]], max_overlap: float = 0.50) -> bool:
    length = max(1, end - start)
    for old_start, old_end in occupied:
        overlap = max(0, min(end, old_end) - max(start, old_start))
        if overlap / length > max_overlap:
            return True
    return False


def discover_shapelets(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_shapelets_per_class: int = 3,
    shapelet_len: int = 128,
    stride: Optional[int] = None,
    max_discovery_windows: int = 180,
    candidates_per_class_channel: int = 8,
    seed: int = 42,
) -> ShapeletBundle:
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    if x.ndim != 3:
        raise ValueError("Shapelet discovery expects x_train with shape [N, C, T]")

    idx = _balanced_indices(y, max_discovery_windows, seed)
    x_disc = x[idx]
    y_disc = y[idx]
    n_samples, n_channels, seq_len = x_disc.shape
    length = min(max(4, int(shapelet_len)), seq_len)
    step = max(1, int(stride if stride is not None else max(1, length // 2)))
    n_shapelets_per_class = max(1, int(n_shapelets_per_class))
    candidates_per_class_channel = max(1, int(candidates_per_class_channel))

    info_rows: List[List[float]] = []
    shapelets: List[np.ndarray] = []
    rng = np.random.default_rng(seed)

    for cls in np.unique(y_disc):
        cls_idx = np.flatnonzero(y_disc == cls)
        other_idx = np.flatnonzero(y_disc != cls)
        if cls_idx.size == 0:
            continue
        if other_idx.size == 0:
            other_idx = cls_idx

        cls_mean = np.mean(x_disc[cls_idx], axis=0)
        other_mean = np.mean(x_disc[other_idx], axis=0)
        pooled_std = np.std(x_disc, axis=0) + 1e-4
        effect = np.abs(cls_mean - other_mean) / pooled_std

        candidates: List[Tuple[float, int, int, int]] = []
        for channel in range(n_channels):
            channel_candidates: List[Tuple[float, int, int, int]] = []
            for start in range(0, seq_len - length + 1, step):
                end = start + length
                score = float(np.mean(effect[channel, start:end]))
                channel_candidates.append((score, channel, start, end))
            channel_candidates.sort(key=lambda item: item[0], reverse=True)
            candidates.extend(channel_candidates[:candidates_per_class_channel])

        candidates.sort(key=lambda item: item[0], reverse=True)
        occupied: Dict[int, List[Tuple[int, int]]] = {channel: [] for channel in range(n_channels)}
        selected_for_class = 0
        for score, channel, start, end in candidates:
            if selected_for_class >= n_shapelets_per_class:
                break
            if _overlaps_existing(start, end, occupied[channel]):
                continue
            target = cls_mean[channel, start:end]
            segs = x_disc[cls_idx, channel, start:end]
            distances = np.mean(np.square(segs - target[None, :]), axis=1)
            local_pos = int(np.argmin(distances)) if distances.size else int(rng.integers(0, len(cls_idx)))
            sample_in_disc = int(cls_idx[local_pos])
            original_sample = int(idx[sample_in_disc])
            shapelet = x_disc[sample_in_disc, channel, start:end].astype(np.float32)
            info_rows.append([float(original_sample), float(start), float(end), float(score), float(cls), float(channel)])
            shapelets.append(shapelet)
            occupied[channel].append((start, end))
            selected_for_class += 1

        while selected_for_class < n_shapelets_per_class:
            sample_in_disc = int(rng.choice(cls_idx))
            channel = int(rng.integers(0, n_channels))
            start = int(rng.integers(0, max(1, seq_len - length + 1)))
            end = start + length
            original_sample = int(idx[sample_in_disc])
            shapelet = x_disc[sample_in_disc, channel, start:end].astype(np.float32)
            info_rows.append([float(original_sample), float(start), float(end), 1e-3, float(cls), float(channel)])
            shapelets.append(shapelet)
            selected_for_class += 1

    info = np.asarray(info_rows, dtype=np.float32)
    if info.size == 0:
        sample = 0
        channel = 0
        start = max(0, (seq_len - length) // 2)
        end = start + length
        info = np.asarray([[0.0, float(start), float(end), 1e-3, float(y_disc[sample]), float(channel)]], dtype=np.float32)
        shapelets = [x_disc[sample, channel, start:end].astype(np.float32)]

    scores = info[:, 3].astype(np.float64)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / max(np.sum(weights), 1e-12) * len(scores)
    info[:, 3] = weights.astype(np.float32)
    return ShapeletBundle(info=info, shapelets=shapelets, discovery_indices=idx)


def _load_original_shapelet_discover():
    root = Path("/home/trinker/Code/github/multivariate-time-series-analysis/ShapeFormer")
    if not root.exists():
        raise RuntimeError(f"Original ShapeFormer directory not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from Shapelet.mul_shapelet_discovery import ShapeletDiscover

    return ShapeletDiscover


def discover_shapelets_pisd(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_shapelets_per_class: int = 3,
    shapelet_len: int = 128,
    max_discovery_windows: int = 60,
    num_pip: float = 0.2,
    processes: int = 1,
    seed: int = 42,
    verbose: bool = False,
) -> ShapeletBundle:
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    if x.ndim != 3:
        raise ValueError("Shapelet discovery expects x_train with shape [N, C, T]")

    idx = _balanced_indices(y, max_discovery_windows, seed)
    x_disc = x[idx]
    y_disc = y[idx]
    _n_samples, n_channels, seq_len = x_disc.shape
    ShapeletDiscover = _load_original_shapelet_discover()
    discovery = ShapeletDiscover(
        window_size=max(1, int(shapelet_len)),
        num_pip=float(num_pip),
        processes=max(1, int(processes)),
        len_of_ts=int(seq_len),
        dim=int(n_channels),
    )
    with _suppress_output(enabled=not verbose):
        discovery.extract_candidate(train_data=x_disc)
        discovery.discovery(train_data=x_disc, train_labels=y_disc)
    raw_info = np.asarray(discovery.get_shapelet_info(number_of_shapelet=max(1, int(n_shapelets_per_class))), dtype=np.float32)
    if raw_info.ndim != 2 or raw_info.shape[1] < 6:
        raise RuntimeError("Original PISD shapelet discovery did not return valid shapelet metadata.")

    shapelets: List[np.ndarray] = []
    info_rows: List[np.ndarray] = []
    for row in raw_info:
        sample_idx = int(row[0])
        start = int(row[1])
        end = int(row[2])
        channel = int(row[5])
        if sample_idx < 0 or sample_idx >= len(x_disc):
            continue
        if channel < 0 or channel >= n_channels:
            continue
        start = max(0, min(start, seq_len - 1))
        end = max(start + 2, min(end, seq_len))
        shapelet = x_disc[sample_idx, channel, start:end].astype(np.float32)
        if shapelet.size < 2:
            continue
        fixed = row.copy()
        fixed[0] = float(idx[sample_idx])
        fixed[1] = float(start)
        fixed[2] = float(end)
        fixed[5] = float(channel)
        info_rows.append(fixed[:6])
        shapelets.append(shapelet)

    if not shapelets:
        return discover_shapelets(
            x_train,
            y_train,
            n_shapelets_per_class=n_shapelets_per_class,
            shapelet_len=shapelet_len,
            max_discovery_windows=max_discovery_windows,
            seed=seed,
        )

    info = np.asarray(info_rows, dtype=np.float32)
    scores = info[:, 3].astype(np.float64)
    if not np.all(np.isfinite(scores)) or np.allclose(scores, 0):
        scores = np.ones_like(scores)
    scores = scores - np.max(scores)
    weights = np.exp(scores)
    weights = weights / max(np.sum(weights), 1e-12) * len(scores)
    info[:, 3] = weights.astype(np.float32)
    return ShapeletBundle(info=info, shapelets=shapelets, discovery_indices=idx)
