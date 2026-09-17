#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from psoqim import EmbedConfig, embed_dataset, extract_dataset, read_shapefile, write_shapefile
from psoqim import attacks
from psoqim.watermark import load_watermark


DATASETS = ["Railways", "Building", "Landuse", "Boundary", "Road", "Lake"]


def save_pca_key(transform, path: Path, dataset_name: str, config: EmbedConfig) -> None:
    if transform is None:
        return
    payload = {
        "dataset": dataset_name,
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
    parser = argparse.ArgumentParser(description="Run optimized Python PSO-QIM experiments.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--watermark", default="data/M.png")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--skip-write", action="store_true")
    return parser.parse_args()


def attack_suite(embedded):
    cases = []
    cases.append(("no_attack", 0, embedded))
    cases.append(("vertex_delete_30pct", 1, attacks.vertex_delete(embedded, 0.3)))
    cases.append(("vertex_delete_50pct", 2, attacks.vertex_delete(embedded, 0.5)))
    cases.append(("object_delete_30pct", 3, attacks.object_delete(embedded, 0.3)))
    cases.append(("object_delete_90pct", 31, attacks.object_delete(embedded, 0.9)))
    cases.append(("vertex_add_10pct", 4, attacks.vertex_add(embedded, 0.1, strength=1.0, tolerance=1e-6)))
    cases.append(("vertex_add_30pct", 5, attacks.vertex_add(embedded, 0.3, strength=1.0, tolerance=1e-6)))
    cases.append(("vertex_add_50pct", 51, attacks.vertex_add(embedded, 0.5, strength=1.0, tolerance=1e-6)))
    cases.append(("noise_10pct_0p6tau", 6, attacks.vertex_noise(embedded, 0.1, strength=0.6)))
    cases.append(("noise_50pct_1p4tau", 7, attacks.vertex_noise(embedded, 0.5, strength=1.4)))
    cases.append(("crop_half_x", 8, attacks.object_crop(embedded, axis="X", keep_low=True)))
    cases.append(("translation_10", 9, attacks.geometric(embedded, angle_deg=0, scale=1, x_shift=10, y_shift=10)))
    cases.append(("scale_0p5", 10, attacks.geometric(embedded, angle_deg=0, scale=0.5, x_shift=0, y_shift=0)))
    for angle in [45, 90, 135, 180, 225, 270, 315]:
        cases.append((f"rotation_{angle}", 100 + angle, attacks.geometric(embedded, angle_deg=angle, scale=1, x_shift=0, y_shift=0)))
    cases.append(("reverse_vertices", 11, attacks.reverse_vertices(embedded)))
    cases.append(("reverse_objects", 12, attacks.reverse_objects(embedded)))
    cases.append(("compound", 13, attacks.compound(embedded)))
    return cases


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    data_dir = (base / args.data_dir).resolve()
    watermark_path = (base / args.watermark).resolve()
    out_dir = (base / args.out_dir).resolve()
    results_dir = (base / args.results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    watermark = load_watermark(watermark_path)
    config = EmbedConfig(num_particles=args.particles, max_iter=args.iters, n_jobs=args.jobs, pca_normalize=True)
    identity_config = EmbedConfig(pca_normalize=False, n_jobs=1)

    rows = []
    summaries = []
    for dataset_name in args.datasets:
        shp_path = data_dir / f"{dataset_name}.shp"
        dataset = read_shapefile(shp_path)

        t0 = time.perf_counter()
        embedded_result = embed_dataset(dataset, watermark, config)
        embed_seconds = time.perf_counter() - t0
        embedded = embedded_result.dataset

        key_path = ""
        if not args.skip_write:
            embedded_path = out_dir / "embedded" / f"Embed_python_pca_pso_{args.particles}_{args.iters}_{dataset_name}.shp"
            write_shapefile(embedded, embedded_path)
            key_path_obj = embedded_path.with_suffix(".pca_key.json")
            save_pca_key(embedded_result.transform, key_path_obj, dataset_name, config)
            key_path = str(key_path_obj)

        summary = {
            "dataset": dataset_name,
            "features": len(dataset.shapes),
            "embed_seconds": embed_seconds,
            "avg_feature_fitness": embedded_result.average_fitness,
            "avg_feature_iterations": embedded_result.average_iterations,
            "particles": args.particles,
            "max_iter": args.iters,
            "pca_key": key_path,
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

        for attack_name, attack_index, attacked in attack_suite(embedded):
            result = extract_dataset(
                attacked,
                watermark,
                config,
                try_pca_variants=True,
                reference_transform=embedded_result.transform,
            )
            row = {
                "dataset": dataset_name,
                "attack": attack_name,
                "attack_index": attack_index,
                "nc": result.nc_value,
                "bit_accuracy": result.bit_accuracy_value,
                "ber": result.ber_value,
                "votes_total": result.votes_total,
                "empty_bins": result.empty_bins,
                "pca_variant": result.transform_variant,
                "embed_seconds": embed_seconds,
                "avg_feature_fitness": embedded_result.average_fitness,
                "avg_feature_iterations": embedded_result.average_iterations,
            }
            if attack_name.startswith("rotation_"):
                no_pca = extract_dataset(attacked, watermark, identity_config, try_pca_variants=False)
                row["nc_without_pca"] = no_pca.nc_value
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))

    results_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summaries)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"optimized_python_attack_results_{timestamp}.csv"
    summary_path = results_dir / f"optimized_python_embedding_summary_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    results_df.to_csv(results_dir / "optimized_python_attack_results_latest.csv", index=False)
    summary_df.to_csv(results_dir / "optimized_python_embedding_summary_latest.csv", index=False)
    print(json.dumps({"results": str(results_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
