#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from psoqim import EmbedConfig, embed_dataset, extract_dataset, read_shapefile, write_shapefile
from psoqim.attacks import geometric
from psoqim.watermark import load_watermark


def save_pca_key(transform, path: Path, config: EmbedConfig) -> None:
    if transform is None:
        return
    payload = {
        "kind": "global_pca_reference_frame",
        "description": "Secret synchronization parameter for PCA-normalized extraction.",
        "center": transform.center.tolist(),
        "basis": transform.basis.tolist(),
        "q": config.q,
        "mcof": config.mcof,
        "random_seed": config.random_seed,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PSO-QIM Python watermarking test.")
    parser.add_argument("--dataset", default="data/Railways.shp")
    parser.add_argument("--watermark", default="data/M.png")
    parser.add_argument("--out", default="outputs/single_embed.shp")
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--rotation", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    dataset_path = (base / args.dataset).resolve()
    watermark_path = (base / args.watermark).resolve()
    out_path = (base / args.out).resolve()

    dataset = read_shapefile(dataset_path)
    watermark = load_watermark(watermark_path)
    config = EmbedConfig(
        num_particles=args.particles,
        max_iter=args.iters,
        n_jobs=args.jobs,
        pca_normalize=not args.no_pca,
        verbose=False,
    )

    t0 = time.perf_counter()
    embedded = embed_dataset(dataset, watermark, config)
    embed_seconds = time.perf_counter() - t0
    write_shapefile(embedded.dataset, out_path)
    key_path = out_path.with_suffix(".pca_key.json")
    save_pca_key(embedded.transform, key_path, config)

    no_attack = extract_dataset(embedded.dataset, watermark, config, reference_transform=embedded.transform)
    rotated = geometric(embedded.dataset, angle_deg=args.rotation)
    rotation_result = extract_dataset(rotated, watermark, config, try_pca_variants=True, reference_transform=embedded.transform)
    identity_after_rotation = extract_dataset(rotated, watermark, EmbedConfig(pca_normalize=False), try_pca_variants=False)

    summary = {
        "dataset": str(dataset_path),
        "output": str(out_path),
        "pca_key": str(key_path) if embedded.transform is not None else "",
        "features": len(dataset.shapes),
        "embed_seconds": embed_seconds,
        "average_fitness": embedded.average_fitness,
        "average_iterations": embedded.average_iterations,
        "no_attack_nc": no_attack.nc_value,
        "rotation_deg": args.rotation,
        "rotation_nc_with_pca": rotation_result.nc_value,
        "rotation_variant": rotation_result.transform_variant,
        "rotation_nc_without_pca": identity_after_rotation.nc_value,
        "empty_bins_no_attack": no_attack.empty_bins,
        "empty_bins_rotation": rotation_result.empty_bins,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
