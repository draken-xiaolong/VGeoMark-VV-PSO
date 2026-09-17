from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import ShapeDataset

SQRT2 = np.sqrt(2.0)


def sdwt_virtual(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Virtual-vertex Haar DWT used by the Matlab implementation.

    Matlab first interleaves every real vertex with the midpoint to the next
    vertex, applies Haar DWT, and later keeps the odd reconstructed positions.
    This vectorized form is algebraically equivalent.
    """

    x = np.asarray(vertices, dtype=float)
    if x.size == 0:
        return np.empty(0), np.empty(0)
    virtual = 0.5 * (x + np.roll(x, -1))
    low = (x + virtual) / SQRT2
    high = (x - virtual) / SQRT2
    return low, high


def isdwt_virtual(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Inverse of the real-vertex component of sdwt_virtual."""

    return (np.asarray(low, dtype=float) + np.asarray(high, dtype=float)) / SQRT2


def sdwt_virtual_rows(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise virtual-vertex Haar DWT for a particle matrix."""

    x = np.asarray(vertices, dtype=float)
    virtual = 0.5 * (x + np.roll(x, -1, axis=1))
    low = (x + virtual) / SQRT2
    high = (x - virtual) / SQRT2
    return low, high


@dataclass
class PCATransform:
    center: np.ndarray
    basis: np.ndarray

    def forward(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.center) @ self.basis

    def inverse(self, xy_canonical: np.ndarray) -> np.ndarray:
        return xy_canonical @ self.basis.T + self.center


def collect_points(dataset: ShapeDataset) -> np.ndarray:
    points = []
    for shape in dataset.shapes:
        if shape.points:
            points.extend(shape.points)
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def fit_global_pca(dataset: ShapeDataset) -> PCATransform:
    """Fit a deterministic right-handed PCA frame to all map vertices."""

    xy = collect_points(dataset)
    if xy.shape[0] < 2:
        return PCATransform(np.zeros(2), np.eye(2))
    center = xy.mean(axis=0)
    centered = xy - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    if np.linalg.det(basis) < 0:
        basis[:, 1] *= -1.0
    return PCATransform(center=center, basis=basis)


def transform_dataset(dataset: ShapeDataset, transform: PCATransform, inverse: bool = False) -> ShapeDataset:
    new_dataset = dataset.copy()
    for shape in new_dataset.shapes:
        if not shape.points:
            continue
        xy = np.asarray(shape.points, dtype=float)
        xy_new = transform.inverse(xy) if inverse else transform.forward(xy)
        shape.points = xy_new.tolist()
    return new_dataset
