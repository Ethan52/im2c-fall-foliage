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
| Requested dates         | 2017-01-01 through 2025-12-31                                               |
| Study region            | Simplified Maine-area polygon; final state and forest masks applied locally |
| AppEEARS request name   | `maine-foliage`                                                             |
| AppEEARS task ID        | `ADD AFTER REQUEST COMPLETES`                                               |
| Download date           | `ADD AFTER DOWNLOAD`                                                        |

### Requested MODIS layers

| Layer                  | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `sur_refl_b01`         | Red surface reflectance                                     |
| `sur_refl_b02`         | Near-infrared surface reflectance                           |
| `sur_refl_qc_500m`     | Per-band quality-control flags                              |
| `sur_refl_state_500m`  | Cloud, shadow, snow, aerosol, and land/water flags          |
| `sur_refl_day_of_year` | Actual selected observation day within each 8-day composite |

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
| Annual NLCD           | Forest and deciduous/mixed-forest mask | Not downloaded |
| Daymet / gridMET      | Historical weather covariates          | Not downloaded |
| NOAA GEFS or AIFS ENS | Prospective weather forecasts          | Not downloaded |
| USDA TreeMap          | Tree-species composition               | Not downloaded |
| USGS 3DEP             | Elevation, slope, aspect               | Not downloaded |
| Maine foliage reports | Qualitative external validation        | Not downloaded |

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
