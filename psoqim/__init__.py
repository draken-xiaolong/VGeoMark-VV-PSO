"""Optimized Python implementation of the PSO-QIM vector watermarking method."""

from .algorithm import EmbedConfig, ExtractionResult, embed_dataset, extract_dataset
from .io import ShapeDataset, read_shapefile, write_shapefile

__all__ = [
    "EmbedConfig",
    "ExtractionResult",
    "ShapeDataset",
    "embed_dataset",
    "extract_dataset",
    "read_shapefile",
    "write_shapefile",
]
