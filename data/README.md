# Data Information

This file documents all external data used in the IM2C fall-foliage project.
Raw downloaded data are excluded from Git; this manifest makes the workflow reproducible.

## Primary foliage-response data

### MODIS Terra Surface Reflectance

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | NASA LP DAAC / AppEEARS                                                     |
| Product                 | MOD09A1.061                                                                 |
| Description             | Terra MODIS Surface Reflectance Bands 1-7                                   |
| Native resolution       | 500 m                                                                       |
| Temporal resolution     | 8-day composite                                                             |
| Native projection       | MODIS Sinusoidal                                                            |
| Requested output format | GeoTIFF                                                                     |
| Requested dates         | 2001-01-01 through 2025-12-31                                               |
| Study region            | Maine State Boundary Polygon (see below), buffered by 5 km and simplified with a 500 m tolerance, in `configs/modis_request_region.geojson`; final state and forest masks applied locally |
| Requested projection    | AppEEARS `native` (MODIS Sinusoidal)                                        |
| Download script         | `src/data_download/dataload_modis.py`, SLURM job `src/shell_scripts/data_download/dataload_modis.sh` |
| Local path              | `data/raw/modis/`                                                           |
| AppEEARS request name   | `maine-foliage`                                                             |
| AppEEARS task ID        | `f8f1e965-5759-4bcb-8d75-14fefea2c61a` (submitted 2026-11-14 23:09 UTC)     |
| Download date           | 2026-11-15 (SLURM job 61844450; 5747 files, 3.48 GB, all sizes and sha256 match the AppEEARS listing, record in `data/raw/modis/download_record.tsv`) |

### Requested MODIS layers

| Layer                  | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `sur_refl_b01`         | Red surface reflectance                                     |
| `sur_refl_b02`         | Near-infrared surface reflectance                           |
| `sur_refl_qc_500m`     | Per-band quality-control flags                              |
| `sur_refl_state_500m`  | Cloud, shadow, snow, aerosol, and land/water flags          |
| `sur_refl_day_of_year` | Actual selected observation day within each 8-day composite |

### MODIS Terra Surface Reflectance, 2026 (comparison only)

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Product and layers      | Same as above: MOD09A1.061, the five layers listed below, same request region, GeoTIFF, native projection |
| Requested dates         | 2026-01-01 through 2026-11-15                                               |
| Use                     | Only to measure the observed 2026 onset with the fixed U and L; not used for training or the 2024 and 2025 assimilation tests |
| Download script         | `src/data_download/dataload_modis_2026.py`, SLURM job `src/shell_scripts/data_download/dataload_modis_2026.sh` |
| Local path              | `data/raw/modis_2026/`                                                      |
| AppEEARS request name   | `maine-foliage-2026`                                                        |
| AppEEARS task ID        | `bfe034e3-0d10-4980-8a53-955304ff7149` (submitted 2026-11-15)               |
| Download date           | 2026-11-15 (SLURM job 61847906, after job 61847739 lost its AppEEARS session; 207 files, 40 composites from 2025-361 to 2026-305, record in `data/raw/modis_2026/download_record.tsv`) |

## Weather data

### Daymet V4R1 daily (training weather)

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | NASA ORNL DAAC, through NASA Earthdata OPeNDAP                              |
| Product                 | Daymet: Daily Surface Weather Data on a 1-km Grid for North America, Version 4 R1, https://doi.org/10.3334/ORNLDAAC/2129, CMR collection `C2532426483-ORNL_CLOUD` |
| Variables and units     | `tmax`, `tmin` (degrees C), `prcp` (mm/day), `srad` (W/m2, mean over daylight hours), `vp` (Pa), `dayl` (s/day); float32, fill -9999 |
| Years                   | 2001 through 2025                                                           |
| Calendar                | 365 days in every year; leap years keep February 29 and drop December 31; `yearday` is the true day of year |
| Grid                    | 1 km, Lambert conformal conic (central meridian -100, latitude of origin 42.5, standard parallels 25 and 60, WGS84 ellipsoid, meters) |
| Subset                  | Box around `configs/modis_request_region.geojson` plus 2 km: x index 6674 to 6995, y index 4031 to 4541 (322 by 511 cells) |
| Format                  | One classic NetCDF-3 file per variable and year, read with `scipy.io.netcdf_file` |
| Download script         | `src/data_download/dataload_daymet.py`, SLURM job `src/shell_scripts/data_download/dataload_daymet.sh` |
| Local path              | `data/raw/daymet/`                                                          |
| Download date           | 2026-11-15 (SLURM job 61847731; per-file checks in `data/raw/daymet/download_record.tsv`) |

### gridMET daily (2026 weather, corrected to Daymet)

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | Climatology Lab, University of California Merced, THREDDS NCSS at `http://thredds.northwestknowledge.net:8080/thredds/ncss/MET/` |
| Variables and units     | `tmmx`, `tmmn` (K), `pr` (mm/day), `srad` (W/m2, 24 hour mean), `vpd` (kPa); packed unsigned 16 bit with scale and offset, fill 32767 |
| Years                   | 2001 through 2026; 2026 runs to 2026-11-13, with days after the permanent slice marked provisional or early in the record |
| Grid                    | 1/24 degree latitude and longitude, latitude descending; the grid ends near -67.06 degrees, west of the easternmost tip of Maine |
| Format                  | One classic NetCDF-3 file per variable and year; array order is day, latitude, longitude (the `dimensions` attribute in the files is wrong) |
| Download script         | `src/data_download/dataload_gridmet.py`, SLURM job `src/shell_scripts/data_download/dataload_gridmet.sh` |
| Local path              | `data/raw/gridmet/`                                                         |
| Download date           | 2026-11-15 (SLURM job 61847744; record in `data/raw/gridmet/download_record.tsv`) |

## Land surface data

### NASADEM elevation

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | NASA LP DAAC / AppEEARS                                                     |
| Product and layer       | `NASADEM_NC.001`, layer `NASADEM_HGT` (meters, int16, fill -32768), 1 arc second (about 30 m) |
| Region and format       | `configs/modis_request_region.geojson`, GeoTIFF, native geographic projection |
| Download script         | `src/data_download/dataload_dem.py`, SLURM job `src/shell_scripts/data_download/dataload_dem.sh` |
| Local path              | `data/raw/dem/`                                                             |
| AppEEARS task ID        | `6e85917f-c1a9-402b-b39c-e91f7b479c60` (submitted 2026-11-15)               |
| Download date           | 2026-11-15 (SLURM job 61847738; record in `data/raw/dem/download_record.tsv`) |

### NLCD Land Cover L48 (forest mask)

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | Multi-Resolution Land Characteristics Consortium (MRLC), GeoServer WCS 2.0.1 at `https://www.mrlc.gov/geoserver/ows`, coverage `mrlc_download__NLCD_<year>_Land_Cover_L48` |
| Years                   | 2001, 2004, 2006, 2008, 2011, 2013, 2016, 2019, 2021                        |
| Values                  | NLCD class codes, uint8, nodata 0; forest classes 41 deciduous, 42 evergreen, 43 mixed |
| Grid                    | EPSG:5070 (CONUS Albers), 30 m, pixel edges at 15 m modulo 30 m             |
| Subset                  | Region bounds plus about 2 km, requested in 100 km tiles and merged         |
| Download script         | `src/data_download/dataload_nlcd.py`, SLURM job `src/shell_scripts/data_download/dataload_nlcd.sh` |
| Local path              | `data/raw/nlcd/`                                                            |
| Download date           | 2026-11-15 (SLURM job 61847762; record in `data/raw/nlcd/download_record.tsv`) |

### USDA Forest Service TreeMap (species composition)

| Year | Archive ID     | Local path                     |
| ---- | -------------- | ------------------------------ |
| 2016 | RDS-2021-0074  | `data/raw/treemap/2016/`       |
| 2020 | RDS-2025-0031  | `data/raw/treemap/2020/`       |
| 2022 | RDS-2025-0032  | `data/raw/treemap/2022/`       |
| 2023 | RDS-2026-0038  | `data/raw/treemap/2023/`       |

Each folder holds the CONUS 30 m raster of imputed plot IDs, its tree table (CSV), and the data dictionary. They were obtained from the USDA Forest Service Research Data Archive (https://www.fs.usda.gov/rds/archive/). The files were downloaded by hand; their modification times are 2026-11-14.

## Validation data

### Maine DACF weekly fall foliage reports

| Field                   | Value                                                                       |
| ----------------------- | --------------------------------------------------------------------------- |
| Provider                | Maine Department of Agriculture, Conservation and Forestry, Maine Forest Service |
| Archive page            | https://www1.maine.gov/dacf/mfs/projects/fall_foliage/report/past.shtml (saved as `past.shtml.html`; includes the 2020 to 2024 peak weeks by zone) |
| Reports                 | 18 weekly reports, 2024 to 2026, listed with URLs in `report_urls.tsv`      |
| How obtained            | Archive page and URL list by `src/data_download/dataload_foliage_reports.py`; the weekly reports refuse scripted requests and are saved by hand as PDF (see `MANUAL_STEPS.md`) |
| Local path              | `data/raw/maine_foliage_reports/`                                           |
| Download date           | 2026-11-15 (archive page by script; the 18 report PDFs saved by hand as `<date>.pdf`) |
| License                 | Public web pages without an explicit open-data license; cited, not redistributed |

Hand-transcribed tables in the same folder (weather-model session, 2026-11-15), read by `src/reporting/dacf.py`:

| File | Content | Source and check |
|---|---|---|
| `zone_categories.tsv` | report date, zone (1 to 7, with N and S halves where a report splits a zone), color change, leaf drop, note; 121 rows from the 18 weekly reports | the "Week of ..." table of each report PDF; rows per report counted against the PDF table (7 to 10 each, all equal). The 2026-09-23 table is malformed in the report itself (zone 4 has no values); its zone 4 color change is read from the report map (Moderate) and its leaf drop left empty, see the note column |
| `peak_weeks_archive.tsv` | year, zone, peak report date, flag for peak after the last report; 2020 to 2024 | the "Historical Peak Foliage Weeks 2020 - 2024" table of `past.shtml.html`. Check: the 2024 peak dates derived from the 2024 weekly reports (first report labeled Peak, or Past Peak if Peak was never reported) equal the archive in all 11 sub-zones |
| `zone_towns.tsv` | town, zone, latitude, longitude | towns named in the reports' zone descriptions and map labels; the halves of zones 1, 3, 4 and 5 follow the report maps of the dates given in the source column. Zone 6 names no towns, so two places in northwestern Maine (Allagash, Clayton Lake) are added as anchors. Coordinates were typed by the agent to about 0.001 degree and not checked against a gazetteer, except Fort Kent, Rangeley and Bar Harbor, which agree with `configs/sites.csv` to 0.004 degree. Each forest cell takes the zone of its nearest town, an approximation of the DACF zone map |

## Boundary data

### Maine State Boundary Polygon

| Field             | Value                                                                       |
| ----------------- | --------------------------------------------------------------------------- |
| Provider          | Maine GeoLibrary                                                            |
| Description       | Authoritative state boundary polygon for Maine, mapped at 1:24,000 scale   |
| Publication date  | 2021-08-02                                                                  |
| CRS               | NAD83 / UTM zone 19N (`EPSG:26919`)                                         |
| Contents          | 6204 polygons: mainland, 6201 islands, and 2 zero-area records with `LAND = 'n'` |
| Local path        | `data/raw/maine_boundary_polygon/Maine_State_Boundary_Polygon_Feature.shp`  |
| Download date     | `ADD`                                                                       |
| Used for          | MODIS request region; exact Maine mask in preprocessing                     |

Above is for sure, rest is GPTed rn, revisit/upate later

### Processing plan

1. Apply the MODIS reflectance scale factor of `0.0001`.
2. Exclude fill values, poor-quality, cloud/shadow, snow/ice, water, and nonforest pixels.
3. Overlay the exact Maine boundary and a deciduous/mixed-forest mask.
4. Aggregate valid 500 m pixels to fixed 1 km analysis cells.
5. Calculate EVI2:

   `EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4 * Red + 1)`

6. Fit an autumn EVI2 decline curve for each cell-year.
7. Convert the fitted curve to normalized brownness:

   `B(t) = (EVI2_summer - EVI2(t)) / (EVI2_summer - EVI2_dormant)`

8. Define satellite-derived onset of peak foliage as the first date for which `B(t) >= 0.60`.

## Planned supporting datasets

| Dataset               | Intended use                           | Status         |
| --------------------- | -------------------------------------- | -------------- |
| NLCD Land Cover L48   | Forest and deciduous/mixed-forest mask | Downloaded, see above |
| Daymet                | Historical weather covariates          | Downloaded, see above |
| gridMET               | 2026 weather, corrected to Daymet      | Downloaded, see above |
| NOAA GEFS or AIFS ENS | Prospective weather forecasts          | Not downloaded; the choice of weather model is open (O9 in `references/project_decisions.md`) |
| USDA TreeMap          | Tree-species composition               | Downloaded, see above |
| NASADEM               | Elevation                              | Downloaded, see above |
| Maine foliage reports | Qualitative external validation        | Archive page saved; weekly reports by hand |

## Geographic conventions

| Item                    | Value                                    |
| ----------------------- | ---------------------------------------- |
| Final study region      | Maine                                    |
| Raw MODIS CRS           | MODIS Sinusoidal                         |
| Analysis CRS            | NAD83 / UTM zone 19N (`EPSG:26919`)      |
| Analysis-unit size      | 1 km × 1 km                              |
| Demonstration locations | Fort Kent, Rangeley, Acadia / Bar Harbor |

## Repository policy

- Raw and processed rasters are stored locally under `data/` and excluded from Git.
- Code, configuration files, source metadata, small tables, and final figures are committed.
- Every download or manually obtained dataset must be added to this manifest.

## Processed data

Derived files in `data/processed/` are excluded from Git and rebuilt by the scripts below. Arrays are NumPy `.npz`. Decisions and results are recorded in `references/project_decisions.md` (D8, D11, D17, D18, D19, D23, V13, V14, V16).

| Path | Contents | Script, SLURM job |
| ---- | -------- | ----------------- |
| `grid/analysis_grid.npz` | 1 km analysis grid in EPSG:26919, 84,001 cells whose centers are in Maine: `cell_id`, `x`, `y`, `lon`, `lat`, `row`, `col`, `id_raster` | `src/data_preprocess/build_grid.py` |
| `forest_mask/forest_mask.npz`, `forest_fraction_500m.tif` | Stable deciduous or mixed forest fraction on the MODIS grid, MODIS pixel to cell map, kept pixels, the 55,284 forest cells (D11) | `src/data_preprocess/build_forest_mask.py` |
| `modis/evi2_<year>.npz` (2001 to 2026) | Cell mean EVI2, mean observation date and pixel count, shape (forest cell, composite), every composite of the year, quality rules D18 | `src/data_preprocess/preprocess_modis.py` |
| `brownness/u_l.npz`, `brownness/brownness_<year>.npz` | Fixed U and L per cell (D2) and observed brownness, shape (forest cell, composite) | `src/data_preprocess/build_brownness.py` |
| `brownness/onset_provisional.npz`, `preprocessing_summary.tsv`, `site_candidates.tsv` | Provisional onset (O2 open), counts and checks, candidate site cells | `src/checks/check_preprocessing.py` |
| `daymet/daymet_<year>.npz` (2001 to 2025) | Daily Daymet `tmax`, `tmin`, `prcp`, `srad`, `vp`, `dayl` on the forest cells, nearest Daymet cell (D19), shape (forest cell, day), Daymet units and calendar | `src/data_preprocess/regrid_daymet.py` |
| `static/elevation.npz` | NASADEM mean elevation of each forest cell over all 30 m pixels in the 1 km cell, `elevation_m` (D7) | `src/data_preprocess/build_elevation.py`, job 61851847 |
| `static/species.npz` | Live-tree basal-area shares of hardwoods, red maple, sugar maple, birches and aspens per forest cell from TreeMap 2016, 2020, 2022, 2023; `share_edition` (group, cell, edition) and `share_year` (group, cell, year 2001 to 2026, D7 interpolation) (D23) | `src/data_preprocess/build_species.py`, job 61852003 |
| `gridmet_daymet/gridmet_daymet_2026.npz`, `heldout_check_2024_2025.tsv` | 2026 gridMET corrected to Daymet on the forest cells, June 1 to November 13, `tmax`, `tmin`, `prcp`, `srad`, `vp` and climatological `dayl`, shape (forest cell, day), Daymet units; per-cell slopes and intercepts; held-out error table (D8, O7) | `src/data_preprocess/correct_gridmet.py`, job 61851849 |
| `phenology/phenology_dates.npz` | Observed onset (O2) and spring green-up (D17) of each forest cell and year 2001 to 2026, fractional day of year, NaN where undefined, shape (forest cell, year) | `src/data_preprocess/build_phenology_dates.py`, run interactively 2026-11-15 |
| `foliage/rows/rows_<year>.npz` (2001 to 2025) | R1 rows: observed brownness `y` and the 13 covariates `X` (row, covariate, names in `covariates`) at each composite observation from Aug 1 to Nov 30, with cell position, composite and dates (D25) | `src/tasks/retrospective.py --stage rows`, job 61853305 |
| `foliage/fits.npz` | R2 fits of F: linear regression and GAM coefficients, noise variances, prior precisions for data assimilation, GAM smooth definitions and GCV grid (D25, D26) | `src/tasks/retrospective.py --stage fit`, job 61853393 |
| `foliage/dayl_climatology.npz` | Daymet daylength climatology 2001 to 2025 by calendar day (cell, day of a leap year), seconds | `src/models/foliage/features.py` |
| `foliage/assimilation_<year>.npz` (2024, 2025) | Kalman filter means of beta at each issue date for both models (D26) | `src/tasks/prospective.py --stage assimilate`, job 61853486 |
| `foliage/forecasts/forecast_<issue date>.npz` | Forecast onset day of year (mapping, model, variant, member, cell; infinite where the curve does not reach 0.6 by the last forecast day) and the ensemble median of predicted brownness at observed composites after the issue date | `src/tasks/prospective.py --stage forecast`, job 61853556 |

The MODIS steps run together with `src/shell_scripts/data_preprocess/preprocess_modis.sh` (job 61848308). Daymet runs with `src/shell_scripts/data_preprocess/regrid_daymet.sh` (job 61848535). Elevation, species, gridMET and phenology dates run one step at a time with `src/shell_scripts/data_preprocess/preprocess_covariates.sh <step>`. The 2026 gridMET days after day 257 are provisional or early releases (file attributes `last_permanent_slice` and `last_provisional_slice`) and may change.
