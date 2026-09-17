from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Watermark:
    original: np.ndarray
    encrypted: np.ndarray

    @property
    def length(self) -> int:
        return int(self.original.size)


def load_binary_watermark(path: str | Path, threshold: int = 127) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    return (arr > threshold).astype(np.uint8)


def logistic_permutation(length: int, init: float = 0.98) -> np.ndarray:
    seq = np.zeros(length, dtype=float)
    seq[0] = init
    for i in range(1, length):
        seq[i] = 1.0 - 2.0 * seq[i - 1] * seq[i - 1]
    return np.argsort(seq, kind="mergesort")


def logistic_encrypt(bits_2d: np.ndarray, init: float = 0.98) -> np.ndarray:
    flat = np.asarray(bits_2d, dtype=np.uint8).ravel()
    perm = logistic_permutation(flat.size, init=init)
    out = np.empty_like(flat)
    out[:] = flat[perm]
    return out.reshape(bits_2d.shape)


def logistic_decrypt(bits_2d: np.ndarray, init: float = 0.98) -> np.ndarray:
    flat = np.asarray(bits_2d, dtype=np.uint8).ravel()
    perm = logistic_permutation(flat.size, init=init)
    out = np.empty_like(flat)
    out[perm] = flat
    return out.reshape(bits_2d.shape)


def load_watermark(path: str | Path, init: float = 0.98) -> Watermark:
    original = load_binary_watermark(path)
    encrypted = logistic_encrypt(original, init=init)
    return Watermark(original=original, encrypted=encrypted)


def nc(mark_get: np.ndarray, mark_prime: np.ndarray) -> float:
    a = np.asarray(mark_get, dtype=float)
    b = np.asarray(mark_prime, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Input watermark arrays must have the same shape")
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def bit_accuracy(mark_get: np.ndarray, mark_prime: np.ndarray) -> float:
    a = np.asarray(mark_get, dtype=np.uint8)
    b = np.asarray(mark_prime, dtype=np.uint8)
    if a.shape != b.shape:
        raise ValueError("Input watermark arrays must have the same shape")
    return float(np.mean(a == b))


def ber(mark_get: np.ndarray, mark_prime: np.ndarray) -> float:
    return 1.0 - bit_accuracy(mark_get, mark_prime)
