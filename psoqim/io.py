from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import shapefile


@dataclass
class ShapeDataset:
    """In-memory representation of a shapefile with records and fields."""

    shape_type: int
    fields: list
    records: list
    shapes: list
    source_path: Path | None = None

    def copy(self) -> "ShapeDataset":
        return ShapeDataset(
            shape_type=self.shape_type,
            fields=copy.deepcopy(self.fields),
            records=copy.deepcopy(self.records),
            shapes=copy.deepcopy(self.shapes),
            source_path=self.source_path,
        )


def read_shapefile(path: str | Path, encoding: str = "utf-8") -> ShapeDataset:
    """Read a shapefile using pyshp, preserving records and field metadata."""

    path = Path(path)
    try:
        reader = shapefile.Reader(str(path), encoding=encoding)
    except UnicodeDecodeError:
        reader = shapefile.Reader(str(path), encoding="gbk")

    try:
        dataset = ShapeDataset(
            shape_type=reader.shapeType,
            fields=copy.deepcopy(reader.fields[1:]),
            records=[list(rec) for rec in reader.records()],
            shapes=[copy.deepcopy(shape) for shape in reader.shapes()],
            source_path=path,
        )
    finally:
        reader.close()
    return dataset


def write_shapefile(dataset: ShapeDataset, path: str | Path) -> Path:
    """Write a ShapeDataset to disk and copy .prj/.cpg metadata when available."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(path), shapeType=dataset.shape_type)
    try:
        for field in dataset.fields:
            writer.field(*field)
        for shape, record in zip(dataset.shapes, dataset.records):
            writer.shape(shape)
            writer.record(*list(record))
    finally:
        writer.close()

    if dataset.source_path is not None:
        for suffix in (".prj", ".cpg"):
            src = dataset.source_path.with_suffix(suffix)
            dst = path.with_suffix(suffix)
            if src.exists():
                shutil.copy2(src, dst)
    return path


def iter_parts(shape: shapefile.Shape) -> Iterable[tuple[int, int, list]]:
    """Yield part index ranges and point lists for a pyshp Shape."""

    parts = list(shape.parts) + [len(shape.points)]
    for start, end in zip(parts[:-1], parts[1:]):
        yield start, end, shape.points[start:end]


def replace_shape_points(shape: shapefile.Shape, points: list, parts: list[int] | None = None) -> shapefile.Shape:
    """Return a copy of shape with replaced points and optionally replaced parts."""

    new_shape = copy.deepcopy(shape)
    new_shape.points = points
    if parts is not None:
        new_shape.parts = parts
    return new_shape


def copy_sidecar_files(src_base: str | Path, dst_base: str | Path) -> None:
    """Copy common shapefile sidecars if they exist."""

    src_base = Path(src_base)
    dst_base = Path(dst_base)
    for suffix in (".prj", ".cpg", ".qmd"):
        src = src_base.with_suffix(suffix)
        dst = dst_base.with_suffix(suffix)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
