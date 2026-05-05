"""ESMValTool CMORizer for CALIOP data.

Tier
    Tier 2

"""

import copy
import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import xarray as xr
from esmvalcore.cmor.table import CMOR_TABLES

from esmvaltool.cmorizers.data import utilities as utils

logger = logging.getLogger(__name__)

band550 = {"name": "green_558nm", "lambda": 558}


def _extract_variable(short_name, var, cfg, in_dir, out_dir):
    attrs = copy.deepcopy(cfg["attributes"])
    attrs["mip"] = var["mip"]
    ver = attrs["version"]
    files = attrs["files"]
    raw_var = var.get("raw_name", short_name)

    cmor_table = CMOR_TABLES[attrs["project_id"]]
    cmor_info = cmor_table.get_variable(var["mip"], short_name)

    logger.info("CMORizing variable '%s' from file(s) '%s'", short_name, files)

    # CALIOP has three sets of files: AllSky_Night, CloudFree_Day, and CloudFree_Night
    # I presume that the best picture of od550aer would include all three of these, but I should ask Ruth.

    """Extract variable."""
    # load data
    for filepath in Path(os.path.join(in_dir, ver)).glob(files):
        xrds = xr.open_dataset(filepath, group="Aerosol_Parameter_Average")
        xrvar = xrds.sel(Band=band550["name"], Optical_Depth_Range="all")[
            raw_var
        ]

        # change order of latitude and longitude coordinates
        xrvar = xrvar.transpose()

        # Add additional coordinates before converting to an iris cube, as this is easier with xarray

        # Time not present in source data, needs to be added manually
        # Determine time from filename:
        fileparts = str(filepath).split("_")
        year = int(fileparts[-3])
        monthstr = fileparts[-4]
        month = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ].index(monthstr) + 1
        days_since_1999 = dt.date(year, month, 15) - dt.date(1999, 1, 1)
        lb_since_1999 = dt.date(year, month, 1) - dt.date(1999, 1, 1)
        if month == 12:
            ub_since_1999 = (
                dt.date(year + 1, 1, 1)
                - dt.date(1999, 1, 1)
                - dt.timedelta(days=1)
            )
        else:
            ub_since_1999 = (
                dt.date(year, month + 1, 1)
                - dt.date(1999, 1, 1)
                - dt.timedelta(days=1)
            )

        xrvar = xrvar.assign_coords(time=days_since_1999.days)
        xrvar = xrvar.expand_dims("time", axis=2)
        xrvar["time"].attrs["units"] = "days since 1999-01-01"

        # timeco = iris.coords.DimCoord(days_since_1999.days, standard_name='time', units='days since 1999-01-01')
        # cube.add_aux_coord(timeco)

        if short_name in ["od550aer", "abs550aer"]:
            xrvar = xrvar.assign_coords(radiation_wavelength=band550["lambda"])
            xrvar["radiation_wavelength"].attrs["units"] = "nm"

        cube = xrvar.to_iris()

        # Fix metadata
        cube.coord("Geodetic Latitude").rename("latitude")
        cube.coord("Geodetic Longitude").rename("longitude")

        # add time bounds
        cube.coord("time").bounds = np.array(
            [ub_since_1999.days, lb_since_1999.days]
        )

        utils.fix_var_metadata(cube, cmor_info)
        utils.set_global_atts(cube, attrs)

        utils.fix_dim_coordnames(cube)

        # When Dask tries to roll this cube, it fails because it can't chunk this properly
        # So here we replicate the part of fix_coords that does that, except with numpy.roll
        # instead of dask.roll.
        cube_coord = cube.coord("longitude")
        logger.info("Fixing longitude...")
        if cube_coord.ndim == 1:
            if cube_coord.points[0] < 0.0 and cube_coord.points[-1] < 181.0:
                cube_coord.points = cube_coord.points + 180.0
                cube.attributes["geospatial_lon_min"] = 0.0
                cube.attributes["geospatial_lon_max"] = 360.0
                nlon = len(cube_coord.points)
                (shift, axis) = (nlon // 2, -1)
                cube.data = np.roll(cube.core_data(), shift, axis=axis)

        utils.fix_coords(cube)

        # fix the wavelength coordinate information.
        if short_name in ["od550aer", "abs550aer"]:
            cube.coord("radiation_wavelength").var_name = "wavelength"
            cube.coord("wavelength").standard_name = "radiation_wavelength"

        utils.set_global_atts(cube, attrs)

        # Save variable
        utils.save_variable(
            cube, short_name, out_dir, attrs, unlimited_dimensions=["time"]
        )


def cmorization(in_dir, out_dir, cfg, cfg_user, start_date, end_date):
    """Run CMORizer for MISR."""
    cfg.pop("cmor_table")

    for short_name, var in cfg["variables"].items():
        _extract_variable(short_name, var, cfg, in_dir, out_dir)
