from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from .io import ShapeDataset
from .transforms import PCATransform, SQRT2, fit_global_pca, isdwt_virtual, sdwt_virtual, sdwt_virtual_rows, transform_dataset
from .watermark import Watermark, ber, bit_accuracy, logistic_decrypt, nc


@dataclass
class EmbedConfig:
    q: float = 3e-2
    mcof: float = 1e6
    num_particles: int = 10
    max_iter: int = 40
    inertia: float = 0.6
    inertia_min: float = 0.0
    c1_initial: float = 1.8
    c1_min: float = 1.6
    c2_initial: float = 1.8
    c2_max: float = 2.2
    early_stop_rounds: int = 8
    early_stop_tol: float = 1e-12
    target_fitness: float = 0.995
    pca_normalize: bool = True
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)
    random_seed: int = 20260615
    verbose: bool = False
    preserve_parts: bool = False


@dataclass
class FeatureEmbedStats:
    feature_index: int
    n_vertices: int
    fitness: float
    iterations: int
    skipped: bool = False
    reason: str = ""


@dataclass
class EmbedResult:
    dataset: ShapeDataset
    stats: list[FeatureEmbedStats]
    transform: PCATransform | None = None

    @property
    def average_fitness(self) -> float:
        values = [s.fitness for s in self.stats if not s.skipped]
        return float(np.mean(values)) if values else 0.0

    @property
    def average_iterations(self) -> float:
        values = [s.iterations for s in self.stats if not s.skipped]
        return float(np.mean(values)) if values else 0.0


@dataclass
class ExtractionResult:
    watermark: np.ndarray
    nc_value: float
    bit_accuracy_value: float
    ber_value: float
    votes_total: int
    empty_bins: int
    transform_variant: str = "identity"
    details: dict[str, Any] = field(default_factory=dict)


def _watermark_bits(watermark: Watermark) -> np.ndarray:
    return np.asarray(watermark.encrypted, dtype=np.uint8).ravel()


def _ratio_and_indices(hx: np.ndarray, hy: np.ndarray, config: EmbedConfig, watermark_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.abs(hy) > np.finfo(float).eps
    ratio = np.zeros_like(hx, dtype=float)
    ratio[valid] = (hx[valid] / hy[valid]) * config.mcof
    indices = np.full(hx.shape, -1, dtype=np.int64)
    raw_ratio = np.zeros_like(hx, dtype=float)
    raw_ratio[valid] = hx[valid] / hy[valid]
    indices[valid] = np.mod(np.floor(raw_ratio[valid] * 100000.0).astype(np.int64), watermark_length)
    return ratio, indices, valid


def _random_valid_offsets(ratio: np.ndarray, target_bits: np.ndarray, valid: np.ndarray, q: float, rng: np.random.Generator, n_rows: int) -> np.ndarray:
    n = ratio.size
    positions = np.zeros((n_rows, n), dtype=float)
    if not np.any(valid):
        return positions

    r = ratio[None, :]
    half = q / 2.0
    offset = np.mod(r, half)
    increase = rng.random((n_rows, n)) * half
    decrease = -(rng.random((n_rows, n)) * half)
    choose_positive = rng.random((n_rows, n)) > 0.5

    residue = np.mod(r, q)
    already_zero = residue < half
    want_zero = target_bits[None, :] == 0
    already_correct = (want_zero & already_zero) | (~want_zero & ~already_zero)

    candidate_same = -offset + increase
    candidate_pos = half - offset + increase
    candidate_neg = -offset + decrease
    candidate_fix = np.where(choose_positive, candidate_pos, candidate_neg)
    positions = np.where(already_correct, candidate_same, candidate_fix)
    positions[:, ~valid] = 0.0
    return positions


def _fitness_batch(
    positions: np.ndarray,
    lx: np.ndarray,
    hx: np.ndarray,
    hy: np.ndarray,
    ratio: np.ndarray,
    indices: np.ndarray,
    valid: np.ndarray,
    encrypted_bits: np.ndarray,
    config: EmbedConfig,
) -> np.ndarray:
    if positions.size == 0 or not np.any(valid):
        return np.zeros(positions.shape[0], dtype=float)

    modified = ratio[None, :] + positions
    hx_mod = np.tile(hx[None, :], (positions.shape[0], 1))
    hx_mod[:, valid] = (modified[:, valid] * hy[None, valid]) / config.mcof

    x_embedded = (lx[None, :] + hx_mod) / np.sqrt(2.0)
    _, hx_cvv = sdwt_virtual_rows(x_embedded)
    h_cvv = np.zeros_like(modified)
    h_cvv[:, valid] = (hx_cvv[:, valid] / hy[None, valid]) * config.mcof

    label_wvv = np.mod(modified, config.q) >= (config.q / 2.0)
    label_cvv = np.mod(h_cvv, config.q) >= (config.q / 2.0)
    compare_valid = valid.copy()
    compare_valid[-1] = False
    if not np.any(compare_valid):
        return np.zeros(positions.shape[0], dtype=float)

    return np.mean(label_wvv[:, compare_valid] == label_cvv[:, compare_valid], axis=1)


def _project_to_qim_interval(values: np.ndarray, target_bits: np.ndarray, q: float) -> np.ndarray:
    out = values.copy()
    residue = np.mod(out, q)
    want_one = target_bits.astype(bool)
    wrong_zero = (~want_one) & (residue >= q / 2.0)
    wrong_one = want_one & (residue < q / 2.0)
    out[wrong_zero] += q / 2.0 - residue[wrong_zero] + q / 4.0
    out[wrong_one] += q / 2.0 - residue[wrong_one] + q / 4.0
    return out


def embed_coordinates(
    xy: np.ndarray,
    watermark: Watermark,
    config: EmbedConfig,
    feature_index: int = 0,
) -> tuple[np.ndarray, FeatureEmbedStats]:
    if xy.shape[0] < 2:
        return xy.copy(), FeatureEmbedStats(feature_index, xy.shape[0], 0.0, 0, True, "too_few_vertices")

    x = xy[:, 0].astype(float)
    y = xy[:, 1].astype(float)
    lx, hx = sdwt_virtual(x)
    _, hy = sdwt_virtual(y)
    if hx.size == 0 or not np.any(np.abs(hy) > np.finfo(float).eps):
        return xy.copy(), FeatureEmbedStats(feature_index, xy.shape[0], 0.0, 0, True, "no_valid_coefficients")

    encrypted_bits = _watermark_bits(watermark)
    ratio, indices, valid = _ratio_and_indices(hx, hy, config, watermark.length)
    target_bits = np.zeros_like(indices, dtype=np.uint8)
    target_bits[valid] = encrypted_bits[indices[valid]]

    rng = np.random.default_rng(config.random_seed + feature_index * 1009)
    positions = _random_valid_offsets(ratio, target_bits, valid, config.q, rng, config.num_particles)
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pbest_fitness = np.full(config.num_particles, -np.inf)
    gbest = positions[0].copy()
    gbest_fitness = -np.inf
    unchanged_rounds = 0
    inertia = config.inertia
    actual_iter = 0

    for iteration in range(1, config.max_iter + 1):
        actual_iter = iteration
        c1 = config.c1_initial * (1 - iteration / config.max_iter) + config.c1_min * (iteration / config.max_iter)
        c2 = config.c2_initial * (1 - iteration / config.max_iter) + config.c2_max * (iteration / config.max_iter)
        inertia = max(inertia - (0.5 / config.max_iter), config.inertia_min)

        fitness = _fitness_batch(positions, lx, hx, hy, ratio, indices, valid, encrypted_bits, config)
        improved = fitness > pbest_fitness
        pbest[improved] = positions[improved]
        pbest_fitness[improved] = fitness[improved]

        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > gbest_fitness + config.early_stop_tol:
            gbest_fitness = float(fitness[best_idx])
            gbest = positions[best_idx].copy()
            unchanged_rounds = 0
        else:
            unchanged_rounds += 1

        if gbest_fitness >= config.target_fitness or unchanged_rounds >= config.early_stop_rounds:
            break

        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = inertia * velocities + c1 * r1 * (pbest - positions) + c2 * r2 * (gbest[None, :] - positions)
        proposed = positions + velocities
        same_interval = np.zeros_like(proposed, dtype=bool)
        same_interval[:, valid] = (np.mod(ratio[None, valid] + proposed[:, valid], config.q) >= config.q / 2.0) == target_bits[None, valid].astype(bool)
        repaired = _random_valid_offsets(ratio, target_bits, valid, config.q, rng, config.num_particles)
        positions = np.where(same_interval, proposed, repaired)
        positions[:, ~valid] = 0.0

    embedded = ratio.copy()
    embedded[valid] = ratio[valid] + gbest[valid]
    embedded[valid] = _project_to_qim_interval(embedded[valid], target_bits[valid], config.q)
    hx_mod = hx.copy()
    hx_mod[valid] = (embedded[valid] * hy[valid]) / config.mcof
    x_embedded = isdwt_virtual(lx, hx_mod)
    out = xy.copy()
    out[:, 0] = x_embedded

    return out, FeatureEmbedStats(feature_index, xy.shape[0], float(max(gbest_fitness, 0.0)), actual_iter)


def _embed_one_shape(shape: Any, watermark: Watermark, config: EmbedConfig, index: int) -> tuple[Any, FeatureEmbedStats]:
    new_shape = copy.deepcopy(shape)
    if not shape.points:
        return new_shape, FeatureEmbedStats(index, 0, 0.0, 0, True, "empty_shape")
    xy = np.asarray(shape.points, dtype=float)
    if not config.preserve_parts:
        embedded_xy, stats = embed_coordinates(xy, watermark, config, index)
    else:
        embedded_xy = xy.copy()
        bounds = list(shape.parts) + [len(xy)]
        substats = []
        for part_id, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            part = xy[start:end]
            closed = shape.shapeType in {5, 15, 25} and len(part) > 1 and np.array_equal(part[0], part[-1])
            unique = part[:-1] if closed else part
            coords, st = embed_coordinates(unique, watermark, config, index + part_id * 1000003)
            embedded_xy[start:end] = np.vstack([coords, coords[0]]) if closed else coords
            substats.append(st)
        usable = [st for st in substats if not st.skipped]
        stats = FeatureEmbedStats(index, len(xy), float(np.mean([st.fitness for st in usable])) if usable else 0., max([st.iterations for st in usable], default=0), not usable)
    new_shape.points = embedded_xy.tolist()
    return new_shape, stats


def _restore_ring_closure(dataset: ShapeDataset, reference: ShapeDataset) -> None:
    if dataset.shape_type not in {5, 15, 25}:
        return
    for shape, original in zip(dataset.shapes, reference.shapes):
        bounds = list(original.parts) + [len(original.points)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end-start > 1 and np.array_equal(original.points[start], original.points[end-1]):
                shape.points[end-1] = list(shape.points[start])


def embed_dataset(dataset: ShapeDataset, watermark: Watermark, config: EmbedConfig) -> EmbedResult:
    working = dataset.copy()
    transform = None
    if config.pca_normalize:
        transform = fit_global_pca(working)
        working = transform_dataset(working, transform, inverse=False)

    if config.preserve_parts:
        _restore_ring_closure(working, dataset)

    if config.n_jobs == 1:
        pairs = [_embed_one_shape(shape, watermark, config, i) for i, shape in enumerate(working.shapes)]
    else:
        pairs = Parallel(n_jobs=config.n_jobs, prefer="processes")(
            delayed(_embed_one_shape)(shape, watermark, config, i) for i, shape in enumerate(working.shapes)
        )
    working.shapes = [shape for shape, _ in pairs]
    stats = [stats for _, stats in pairs]

    if config.pca_normalize and transform is not None:
        working = transform_dataset(working, transform, inverse=True)
    if config.preserve_parts:
        _restore_ring_closure(working, dataset)
    working.source_path = dataset.source_path
    return EmbedResult(dataset=working, stats=stats, transform=transform)


def _extract_votes(
    dataset: ShapeDataset,
    watermark: Watermark,
    config: EmbedConfig,
    transform: PCATransform | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    votes_one = np.zeros(watermark.length, dtype=np.int64)
    votes_total = np.zeros(watermark.length, dtype=np.int64)

    point_blocks = []
    starts = []
    ends = []
    cursor = 0
    for shape in dataset.shapes:
        if not shape.points:
            continue
        xy = np.asarray(shape.points, dtype=float)
        bounds = list(shape.parts) + [len(xy)] if config.preserve_parts else [0, len(xy)]
        for start, end in zip(bounds[:-1], bounds[1:]):
            part = xy[start:end]
            if config.preserve_parts and shape.shapeType in {5, 15, 25} and len(part)>1 and np.array_equal(part[0],part[-1]):
                part = part[:-1]
            if len(part) < 2:
                continue
            starts.append(cursor)
            cursor += len(part)
            ends.append(cursor)
            point_blocks.append(part)

    if not point_blocks:
        return votes_one, votes_total, 0

    xy_all = np.vstack(point_blocks)
    if transform is not None:
        xy_all = transform.forward(xy_all)

    next_idx = np.arange(xy_all.shape[0]) + 1
    next_idx[np.asarray(ends, dtype=np.int64) - 1] = np.asarray(starts, dtype=np.int64)

    x = xy_all[:, 0]
    y = xy_all[:, 1]
    hx = (x - 0.5 * (x + x[next_idx])) / SQRT2
    hy = (y - 0.5 * (y + y[next_idx])) / SQRT2
    valid = np.abs(hy) > np.finfo(float).eps
    if not np.any(valid):
        return votes_one, votes_total, 0

    raw_ratio = hx[valid] / hy[valid]
    p = np.mod(np.floor(raw_ratio * 100000.0).astype(np.int64), watermark.length)
    ratio_scaled = raw_ratio * config.mcof
    bits = (np.mod(ratio_scaled, config.q) >= config.q / 2.0).astype(np.int64)
    np.add.at(votes_one, p, bits)
    np.add.at(votes_total, p, 1)
    return votes_one, votes_total, int(np.sum(votes_total))


def _result_from_votes(votes_one: np.ndarray, votes_total: np.ndarray, watermark: Watermark, variant: str) -> ExtractionResult:
    encrypted = np.zeros(watermark.length, dtype=np.uint8)
    nonempty = votes_total > 0
    encrypted[nonempty] = (votes_one[nonempty] / votes_total[nonempty] >= 0.5).astype(np.uint8)
    encrypted[~nonempty] = 0
    encrypted_2d = encrypted.reshape(watermark.original.shape)
    decrypted = logistic_decrypt(encrypted_2d)
    return ExtractionResult(
        details={"votes_per_bit": votes_total.tolist(), "one_votes_per_bit": votes_one.tolist(), "ties": int(np.sum(nonempty & (2 * votes_one == votes_total)))},
        watermark=decrypted,
        nc_value=nc(decrypted, watermark.original),
        bit_accuracy_value=bit_accuracy(decrypted, watermark.original),
        ber_value=ber(decrypted, watermark.original),
        votes_total=int(np.sum(votes_total)),
        empty_bins=int(np.sum(~nonempty)),
        transform_variant=variant,
    )


def _pca_candidate_transforms(dataset: ShapeDataset) -> list[tuple[str, PCATransform]]:
    transform = fit_global_pca(dataset)
    candidates: list[tuple[str, PCATransform]] = []
    for name, signs in (
        ("pca_++", np.array([1.0, 1.0])),
        ("pca_-+", np.array([-1.0, 1.0])),
        ("pca_+-", np.array([1.0, -1.0])),
        ("pca_--", np.array([-1.0, -1.0])),
    ):
        variant = PCATransform(transform.center, transform.basis * signs[None, :])
        candidates.append((name, variant))
    return candidates


def extract_dataset(
    dataset: ShapeDataset,
    watermark: Watermark,
    config: EmbedConfig,
    try_pca_variants: bool = True,
    reference_transform: PCATransform | None = None,
) -> ExtractionResult:
    candidates: list[tuple[str, PCATransform | None]] = [("identity", None)]
    if reference_transform is not None:
        candidates.append(("embed_pca_key", reference_transform))
    if config.pca_normalize and try_pca_variants:
        candidates.extend(_pca_candidate_transforms(dataset))
    elif config.pca_normalize:
        candidates.append(("pca", fit_global_pca(dataset)))

    results = []
    for variant, transform in candidates:
        votes_one, votes_total, total_votes = _extract_votes(dataset, watermark, config, transform=transform)
        result = _result_from_votes(votes_one, votes_total, watermark, variant)
        result.details["total_votes"] = total_votes
        results.append(result)
    return max(results, key=lambda r: r.nc_value)
