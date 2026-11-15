import sys
import logging

import numpy as np

logger = logging.getLogger(__name__)

# checks how close autumn daylength is to a straight line in time, the evidence for decision D4 in references/project_decisions.md
DOY = np.arange(213, 335)  # August 1 to November 30 in a non-leap year
SITE_LATITUDES = [44.0, 45.5, 47.3]
MAINE_LATITUDES = np.linspace(43.0, 47.5, 50)


def daylength(lat_deg: np.ndarray, doy: np.ndarray):
    '''daylength in hours from the CBM model (Forsythe et al. 1995), sunrise and sunset at the top of the solar disk'''
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))  # solar declination
    p = 0.8333  # degrees of the sun below the horizon at sunrise and sunset
    lat = np.deg2rad(lat_deg)
    arg = (np.sin(np.deg2rad(p)) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    return 24 - (24 / np.pi) * np.arccos(arg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    for lat in SITE_LATITUDES:
        s = daylength(np.full(DOY.shape, lat), DOY)  # day of year
        r = np.corrcoef(DOY, s)[0, 1]
        resid = s - np.polyval(np.polyfit(DOY, s, 1), DOY)
        logger.info(
            "latitude %.1f: daylength %.2f to %.2f hours, correlation with day %.5f, largest gap from a line %.1f minutes",
            lat, s.max(), s.min(), r, 60 * np.abs(resid).max(),
        )
    lat_grid, doy_grid = np.meshgrid(MAINE_LATITUDES, DOY, indexing="ij")  # latitude, day of year
    s = daylength(lat_grid, doy_grid)
    logger.info("pooled over Maine latitudes: correlation of day and daylength %.5f", np.corrcoef(doy_grid.ravel(), s.ravel())[0, 1])


if __name__ == "__main__":
    main()
