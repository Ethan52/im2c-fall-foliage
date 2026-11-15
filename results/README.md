# Results and figures

Every table and figure is written by a script. Rerun a script to rebuild what it produced.

## Retrospective pipeline

| Output | Script |
|---|---|
| `results/retrospective/fit_summary.tsv`, `lr_coefficients.tsv`, `r3_test_scores.tsv`, `r3_site_onsets.tsv`; `figures/retrospective/r3_site_curves.png`, `gam_partial_effects.png` | `src/tasks/retrospective.py` (data/pipeline session) |
| `figures/retrospective/r3_brownness_maps_<year>.png`: observed and GAM brownness at four composites, and their difference, 2024 and 2025 | `src/tasks/retrospective_report.py` |
| `figures/retrospective/r3_onset_maps_<year>.png`, `r3_onset_scatter_<year>.png`: observed and GAM onset; the script asserts that the recomputed GAM onset bias, MAE and cell count equal `r3_test_scores.tsv` | `src/tasks/retrospective_report.py` |
| `results/retrospective/dacf_peak_weeks_vs_modis_onset.tsv`: DACF peak report date against the median observed MODIS onset of each zone, 2020 to 2026 | `src/tasks/retrospective_report.py` |
| `results/retrospective/split_fit_summary.tsv`, `split_lr_coefficients.tsv`, `split_by_year/split_scores_<year>.tsv`: fit on the training cells of a within-year 80/20 cell split (20% of cells held out, the same cells every year, seed in `retrospective_config.json`), scored on test and training cells in every year 2001 to 2025 | `src/tasks/retrospective.py`, stages `split_fit` and `split_evaluate` |
| `results/retrospective/split_scores.tsv`, `split_scores_summary.tsv`, `figures/retrospective/split_scores_by_year.png`, `brownness_and_onset_panels_<years>.png` (split fit) | `src/tasks/retrospective_report.py` |
| `results/retrospective/preprocessing_counts.tsv`: grid, forest-mask and U and L counts, and undefined onsets per year | `src/tasks/retrospective_report.py` |

## Prospective pipeline

| Output | Script |
|---|---|
| `results/prospective/prospective_scores.tsv`, `prospective_site_onsets.tsv`; `figures/prospective/onset_mae_by_issue_date.png` | `src/tasks/prospective.py` (data/pipeline session) |
| `figures/prospective/forecast_onset_maps_<year>_<mapping>.png`: ensemble median forecast onset at three issue dates, its error, and the observed onset | `src/tasks/prospective_report.py` |
| `figures/prospective/assimilation_onset_change_maps_<year>.png`: assimilated minus fit forecast onset at the same issue dates | `src/tasks/prospective_report.py` |
| `figures/prospective/scores_by_issue_date.png`, `results/prospective/assimilation_gain.tsv`: onset MAE and brownness RMSE against issue date, and the change from assimilation | `src/tasks/prospective_report.py` |
| `figures/prospective/assimilation_beta_change.png`: relative change of the coefficients and observations assimilated | `src/tasks/prospective_report.py` |
| `figures/prospective/site_onset_fans.png`: forecast onset percentiles at the three site cells against issue date | `src/tasks/prospective_report.py` |
| `figures/prospective/dacf_category_maps_<year>.png`, `results/prospective/dacf_category_confusion_<year>.tsv`, `dacf_peak_weeks_vs_forecast_onset.tsv`: DACF categories and peak weeks against the forecast | `src/tasks/prospective_report.py` |
| `figures/prospective/forecast_onset_maps_2026.png`: only if `forecast_2026-08-01.npz` exists | `src/tasks/prospective_report.py` |

## Manuscript summary

| Output | Script |
|---|---|
| `results/prospective/aug1_error_summary.tsv`; `figures/manuscript/aug1_error_histograms.png`, `dacf_report_minus_onset.png`, `sites_2026.png` | `src/tasks/manuscript_figures.py` (reads outputs of the scripts above and `prospective_calibration.py`) |

## DACF comparison conventions

- **Zones:** each forest cell takes the DACF zone of its nearest named town (`data/raw/maine_foliage_reports/zone_towns.tsv`). This approximates the zone map in the reports.
- **Onset:** onsets use the brownness threshold 0.6 of D1.
- **Category maps:** these read the DACF percent ranges as brownness fractions: Very Low below 0.1, Low 0.1 to 0.3, Moderate 0.3 to 0.5, High 0.5 to 0.7, Peak at least 0.7. This is an assumption. Brownness cannot distinguish Past Peak, so the confusion tables merge Peak and Past Peak.
