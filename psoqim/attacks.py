from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from .io import ShapeDataset


def _parts(shape) -> list[tuple[int, int]]:
    parts = list(shape.parts) + [len(shape.points)]
    return list(zip(parts[:-1], parts[1:]))


def _is_polygon_shape(shape_type: int) -> bool:
    return shape_type in {5, 15, 25, 31}


def _fix_part(points: np.ndarray, shape_type: int) -> np.ndarray:
    if points.size == 0:
        return points
    if _is_polygon_shape(shape_type):
        if len(points) < 3:
            return points
        if not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
    return points


def _replace_parts(shape, new_parts: list[np.ndarray], shape_type: int):
    new_shape = copy.deepcopy(shape)
    points = []
    parts = []
    for part in new_parts:
        if part.size == 0:
            continue
        part = _fix_part(part, shape_type)
        if len(part) == 0:
            continue
        parts.append(len(points))
        points.extend(part.tolist())
    if not points:
        points = shape.points[:]
        parts = list(shape.parts)
    new_shape.points = points
    new_shape.parts = parts if parts else [0]
    return new_shape


def vertex_delete(dataset: ShapeDataset, ratio: float, seed: int = 1) -> ShapeDataset:
    rng = np.random.default_rng(seed)
    out = dataset.copy()
    for idx, shape in enumerate(dataset.shapes):
        new_parts = []
        for start, end in _parts(shape):
            pts = np.asarray(shape.points[start:end], dtype=float)
            if len(pts) <= 2:
                new_parts.append(pts)
                continue
            keep = rng.random(len(pts)) >= ratio
            if _is_polygon_shape(dataset.shape_type):
                keep[0] = True
                keep[-1] = True
                if np.sum(keep) < 4:
                    keep[:] = False
                    keep[[0, max(1, len(pts) // 2), -1]] = True
            else:
                keep[0] = True
                keep[-1] = True
            new_parts.append(pts[keep])
        out.shapes[idx] = _replace_parts(shape, new_parts, dataset.shape_type)
    return out


def vertex_add(dataset: ShapeDataset, ratio: float, strength: float = 1.0, tolerance: float = 1e-6, seed: int = 233337) -> ShapeDataset:
    rng = np.random.default_rng(seed)
    out = dataset.copy()
    for idx, shape in enumerate(dataset.shapes):
        new_parts = []
        for start, end in _parts(shape):
            pts = np.asarray(shape.points[start:end], dtype=float)
            if len(pts) <= 1:
                new_parts.append(pts)
                continue
            new_pts = [pts[0]]
            for j in range(1, len(pts)):
                if rng.random() < ratio:
                    t = rng.random()
                    new_pt = pts[j - 1] + t * (pts[j] - pts[j - 1])
                    new_pt = new_pt + np.array([strength * tolerance, -strength * tolerance])
                    new_pts.append(new_pt)
                new_pts.append(pts[j])
            new_parts.append(np.asarray(new_pts, dtype=float))
        out.shapes[idx] = _replace_parts(shape, new_parts, dataset.shape_type)
    return out


def vertex_noise(dataset: ShapeDataset, ratio: float, strength: float, seed: int = 2024) -> ShapeDataset:
    rng = np.random.default_rng(seed)
    out = dataset.copy()
    for idx, shape in enumerate(dataset.shapes):
        new_parts = []
        for start, end in _parts(shape):
            pts = np.asarray(shape.points[start:end], dtype=float)
            if len(pts) == 0:
                new_parts.append(pts)
                continue
            mask = rng.random(len(pts)) < ratio
            noise = rng.uniform(-strength, strength, size=pts.shape)
            pts2 = pts.copy()
            pts2[mask] += noise[mask]
            new_parts.append(pts2)
        out.shapes[idx] = _replace_parts(shape, new_parts, dataset.shape_type)
    return out


def object_delete(dataset: ShapeDataset, ratio: float, seed: int = 2025) -> ShapeDataset:
    rng = np.random.default_rng(seed)
    n = len(dataset.shapes)
    keep = rng.random(n) >= ratio
    if not np.any(keep):
        keep[rng.integers(0, n)] = True
    out = dataset.copy()
    out.shapes = [shape for shape, k in zip(out.shapes, keep) if k]
    out.records = [record for record, k in zip(out.records, keep) if k]
    return out


def object_crop(dataset: ShapeDataset, axis: str = "X", keep_low: bool = True) -> ShapeDataset:
    all_points = np.vstack([np.asarray(shape.points, dtype=float) for shape in dataset.shapes if shape.points])
    dim = 0 if axis.upper() == "X" else 1
    threshold = float(np.median(all_points[:, dim]))
    out = dataset.copy()
    for idx, shape in enumerate(dataset.shapes):
        new_parts = []
        for start, end in _parts(shape):
            pts = np.asarray(shape.points[start:end], dtype=float)
            if keep_low:
                mask = pts[:, dim] <= threshold
            else:
                mask = pts[:, dim] >= threshold
            if np.sum(mask) >= 2:
                new_parts.append(pts[mask])
            else:
                new_parts.append(pts)
        out.shapes[idx] = _replace_parts(shape, new_parts, dataset.shape_type)
    return out


def geometric(dataset: ShapeDataset, angle_deg: float = 0.0, scale: float = 1.0, x_shift: float = 0.0, y_shift: float = 0.0) -> ShapeDataset:
    theta = math.radians(angle_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=float)
    out = dataset.copy()
    for shape in out.shapes:
        if not shape.points:
            continue
        pts = np.asarray(shape.points, dtype=float)
        pts2 = (pts @ rot.T) * scale + np.array([x_shift, y_shift])
        shape.points = pts2.tolist()
    return out


def reverse_vertices(dataset: ShapeDataset) -> ShapeDataset:
    out = dataset.copy()
    for idx, shape in enumerate(dataset.shapes):
        new_parts = []
        for start, end in _parts(shape):
            pts = np.asarray(shape.points[start:end], dtype=float)
            new_parts.append(pts[::-1])
        out.shapes[idx] = _replace_parts(shape, new_parts, dataset.shape_type)
    return out


def reverse_objects(dataset: ShapeDataset) -> ShapeDataset:
    out = dataset.copy()
    out.shapes = list(reversed(out.shapes))
    out.records = list(reversed(out.records))
    return out


def compound(dataset: ShapeDataset) -> ShapeDataset:
    out = vertex_delete(dataset, 0.3)
    out = object_delete(out, 0.3)
    out = vertex_add(out, 0.1, strength=1.0, tolerance=0.01)
    out = vertex_noise(out, 0.2, strength=0.6)
    out = geometric(out, angle_deg=0.0, scale=1.0, x_shift=10.0, y_shift=10.0)
    out = geometric(out, angle_deg=0.0, scale=0.5, x_shift=0.0, y_shift=0.0)
    out = reverse_vertices(out)
    out = reverse_objects(out)
    return out


@dataclass
class AttackCase:
    name: str
    attack_index: int
    dataset: ShapeDataset
