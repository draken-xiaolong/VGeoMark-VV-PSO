# Figure Data CSV Notes

This folder contains cleaned CSV data extracted from the table screenshots in `图片数据`.

## Corrected source-file numbering

The source screenshots were shifted one figure number forward:

- `FigNP.png` -> `Figure_11_negative_probability_data.png`
- `Fig11.png` -> `Figure_12_pso_iteration_nc_data.png`
- `Fig12-a.png`--`Fig12-f.png` -> `Figure_13a`--`Figure_13f` vertex-deletion data
- `Fig13-a.png`--`Fig13-b.png` -> `Figure_14a`--`Figure_14b` object-removal data
- `Fig14-a.png`--`Fig14-c.png` -> `Figure_15a`--`Figure_15c` geometric-attack data

## Unit normalization

- `Figure_11_negative_probability.csv`: NP values were shown as percentages in the source screenshot and are stored here as probabilities from 0 to 1.
- `Figure_12_pso_iteration_nc.csv`: NC values were shown as percentages in the source screenshot and are stored here as NC values from 0 to 1.
- `Figure_13`--`Figure_15`: NC values were already in 0 to 1 form in the source screenshots.
- Obvious OCR/source decimal-loss cells such as `96.6880` were normalized to `0.96688`, because NC must be bounded by 0 and 1.

## Baseline-method mapping

The source screenshots use old column codes. They were mapped to the method labels in the existing figure legends as follows:

- `E4` -> `yan_2017` -> Yan et al. (2017)
- `E2_30` -> `li_2021` -> Li et al. (2021)
- `E5_35` -> `zhang_2025` -> Zhang et al. (2025)
- `E3_36` -> `lin_2018` -> Lin et al. (2018)
- `E1_37` -> `xi_2022` -> Xi et al. (2022)
- `Proposed` -> `proposed`

The mapping is also saved in `Figure_method_mapping.csv`.

## Row semantics

- `Figure_12_pso_iteration_nc.csv`: each row is one attack index from the manuscript caption for Fig. 12.
- `Figure_13_vertex_deletion_nc.csv`: each row is one dataset and vertex-deletion ratio. Panels a--f are Railways, Building, Landuse, Boundary, Road, and Lake.
- `Figure_14_object_removal_nc.csv`: panel a is Landuse; panel b is the average over all six datasets.
- `Figure_15_geometric_attacks_nc.csv`: panels a--c are rotation, scaling, and translation, respectively. Scaling rows use `display_x` as the plotted percentage label and `semantic_x` as the actual scale factor.
