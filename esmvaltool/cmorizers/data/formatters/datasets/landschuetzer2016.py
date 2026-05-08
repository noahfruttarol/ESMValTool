"""ESMValTool CMORizer for Landschuetzer2016 data.

Tier
   Tier 2: other freely-available dataset.

Source
   https://www.nodc.noaa.gov/archive/arc0105/0160558/3.3/data/0-data/

Last access
   20190308

Download and processing instructions
   Download the file spco2_1982-2015_MPI_SOM-FFN_v2016.nc

Modification history
   20190227-lovato_tomas: written.
"""

import datetime
import logging
import os
from warnings import catch_warnings, filterwarnings

import iris
from dask import array as da

from esmvaltool.cmorizers.data.utilities import (
    add_scalar_depth_coord,
    constant_metadata,
    fix_coords,
    fix_var_metadata,
    save_variable,
    set_global_atts,
)

logger = logging.getLogger(__name__)


def _fix_data(cube, var):
    """Specific data fixes for different variables."""
    logger.info("Fixing data ...")
    with constant_metadata(cube) as metadata:
        if var == "fgco2":
            # Assume standard year 365_day
            cube *= -12.01 / 1000.0 / (86400.0 * 365.0)
            metadata.attributes["positive"] = "down"
        elif var == "spco2":
            cube *= 101325.0 / 1.0e06
    return cube


# pylint: disable=unused-argument
def _fix_fillvalue(cube, field, filename):
    """Create masked array from missing_value."""
    if hasattr(field.cf_data, "missing_value"):
        # fix for bad missing value definition
        cube.data = da.ma.masked_equal(
            cube.core_data(),
            field.cf_data.missing_value,
        )

def _fix_scalar_coords(cube, cmor_var):
    """Fix scalar coordinates."""
    if cmor_var in ["fgco2", "spco2"]:
        add_scalar_depth_coord(cube)

def extract_variable(var_info, raw_info, out_dir, attrs):
    """Extract to all vars."""
    var = var_info.short_name
    with catch_warnings():
        filterwarnings(
            action="ignore",
            message="Ignoring netCDF variable .* invalid units .*",
            category=UserWarning,
            module="iris",
        )
        cubes = iris.load(raw_info["file"], callback=_fix_fillvalue)
    rawvar = raw_info["name"]

    for cube in cubes:
        if cube.var_name == rawvar:
            _fix_scalar_coords(cube, var)
            fix_var_metadata(cube, var_info)
            index_year = 2023
            timeepoch = datetime.datetime(1950, 1, 1)
            time_points = [0] * 12
            for month in range(1, 13):
                month_stamp = datetime.datetime(index_year, month, 15)
                month_stamp = month_stamp - timeepoch

                time_points[month - 1] = month_stamp.days
            cube.coord("time").points = time_points
            cube.coord('time').units = 'days since 1950-01-01 00:00:00'

            # ensure auxiliary depth coord uses CMOR var name 'depth'
            fix_coords(cube)

            # ensure auxiliary depth coord uses CMOR var name 'depth'
            for coord in cube.coords():
                if coord.var_name == 'lev' and (
                    coord.name().lower().startswith('depth')
                    or getattr(coord, 'standard_name', '') == 'depth'
                ):
                    coord.var_name = 'depth'
                    coord.standard_name = 'depth'
                    coord.long_name = 'depth'
                    break

            _fix_data(cube, var)
            set_global_atts(cube, attrs)
            save_variable(
                cube,
                var,
                out_dir,
                attrs,
                local_keys=["positive"],
                unlimited_dimensions=["time"],
            )
            _fix_data(cube, var)
            set_global_atts(cube, attrs)
            save_variable(
                cube,
                var,
                out_dir,
                attrs,
                local_keys=["positive"],
                unlimited_dimensions=["time"],
            )


def cmorization(in_dir, out_dir, cfg, cfg_user, start_date, end_date):
    """Cmorization func call."""
    cmor_table = cfg["cmor_table"]
    glob_attrs = cfg["attributes"]

    # run the cmorization
    for var, vals in cfg["variables"].items():
        inpfile = os.path.join(in_dir, vals["file"])
        logger.info("CMORizing var %s from file %s", var, inpfile)
        var_info = cmor_table.get_variable(vals["mip"], var)
        raw_info = {"name": vals["raw"], "file": inpfile}
        glob_attrs["mip"] = vals["mip"]
        with catch_warnings():
            filterwarnings(
                action="ignore",
                message=(
                    "WARNING: missing_value not used since it\n"
                    "cannot be safely cast to variable data type"
                ),
                category=UserWarning,
                module="iris",
            )
            extract_variable(var_info, raw_info, out_dir, glob_attrs)
