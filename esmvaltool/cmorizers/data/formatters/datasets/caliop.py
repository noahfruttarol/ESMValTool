"""ESMValTool CMORizer for CALIOP data.

Tier
    Tier 2

"""

import datetime
import logging
from pathlib import Path

import iris
import numpy as np
from dask import array as da
from esmvalcore.cmor.table import CMOR_TABLES
from pyhdf.SD import SD, SDC

from esmvaltool.cmorizers.data import utilities as utils

logger = logging.getLogger(__name__)


def get_year_from_filepath(filename: str) -> int:
    """get year from first four characters of last section of basename"""

    offset = -8
    year = int(filename[offset - 4 : offset])
    return year


def get_month_from_filepath(filename: str) -> int:
    """get month from timerange section of basename"""

    offset = -5
    month = int(filename[offset - 2 : offset])
    return month


def group_files_by_year(filepaths: list) -> dict:
    """group filepaths by year using get_year_from_filepath function"""
    years_D = dict()
    for filepath in filepaths:
        filename = str(filepath)
        year = get_year_from_filepath(filename)

        if year not in years_D:
            years_D[year] = [filepath]
        else:
            years_D[year].append(filepath)

    return years_D


def read_hdf(
    filepath: str,
    year: int,
    month: int,
    raw_name: str,
    short_name: str,
    **extras,
):
    """Read HDF file and build iris cube with auxiliary data for special variables."""
    f = SD(filepath, SDC.READ)

    data_obj = f.select(raw_name)
    data = data_obj.get().astype(np.float32)

    # mask fill values
    if hasattr(data_obj, "fill_value"):
        data[data == np.float32(data_obj.fill_value)] = np.nan

    # mask values outside valid range
    if hasattr(data_obj, "valid_range"):
        valid_range = data_obj.valid_range
        data[
            (data < np.float32(valid_range[0:2]))
            | (data > np.float32(valid_range[-2:]))
        ] = np.nan

    # apply scale factor and add offset if they exist
    if hasattr(data_obj, "scale_factor"):
        scale_factor = np.float32(data_obj.scale_factor)
        data *= scale_factor
    if hasattr(data_obj, "add_offset"):
        add_offset = np.float32(data_obj.add_offset)
        data += add_offset

    # mask all null values (nan, inf)
    data = np.ma.masked_invalid(data)
    # Handle units - convert invalid units to '1'
    units_str = data_obj.attributes().get("units", "1")
    if units_str in ["none", "None", "", "NoUnits", None]:
        units_str = "1"

    # Create time coordinate for the start of the month, with bounds covering the whole month
    time_point = datetime.datetime(
        year=year, month=month, day=15
    )  # use 15th as representative time for the month
    time_bounds_lower = datetime.datetime(year=year, month=month, day=1)
    time_bounds_upper = datetime.datetime(
        year=year + (month == 12), month=month + 1 - (month == 12) * 12, day=1
    ) - datetime.timedelta(
        days=1
    )  # end of month is the day before the first day of the next month

    # Convert time to days since 1850-01-01
    time_units = datetime.datetime(1850, 1, 1)
    delta_1850 = (time_point - time_units).days
    # After creating time delta, add bounds:
    delta_bounds_lower = (time_bounds_lower - time_units).days
    delta_bounds_upper = (time_bounds_upper - time_units).days

    time_coord = iris.coords.DimCoord(
        points=[delta_1850],
        standard_name="time",
        units="days since 1850-01-01 00:00:00",
        bounds=[[delta_bounds_lower, delta_bounds_upper]],
    )
    data = data[np.newaxis, :, :]  # Add time dimension at start

    lats = f.select("Latitude_Midpoint").get().astype(np.float64)[0]
    lat_coord = iris.coords.DimCoord(
        lats,
        standard_name="latitude",
        units="degrees_north",
        bounds=[
            (  # calculate bounds as midpoint between adjacent latitudes, assuming regular grid
                lats[i] - 0.5 * np.abs(lats[1] - lats[0]),
                lats[i] + 0.5 * np.abs(lats[1] - lats[0]),
            )
            for i in range(len(lats))
        ],
    )

    lons = f.select("Longitude_Midpoint").get().astype(np.float64)[0]
    lon_coord = iris.coords.DimCoord(
        lons,
        standard_name="longitude",
        units="degrees_east",
        bounds=[
            (  # calculate bounds as midpoint between adjacent longitudes, assuming regular grid
                lons[i] - 0.5 * np.abs(lons[1] - lons[0]),
                lons[i] + 0.5 * np.abs(lons[1] - lons[0]),
            )
            for i in range(len(lons))
        ],
    )

    if short_name in ["od550aer"]:
        # add auxiliary coordinate for wavelength (550 nm for this variable, which is AOD at 550 nm)
        wavelength_coord = iris.coords.AuxCoord(
            [532],
            standard_name="radiation_wavelength",
            var_name="wavelength",
            units="nm",
        )
        aux_coords_and_dims = [
            (
                wavelength_coord,
                None,
            ),  # wavelength is a scalar auxiliary coordinate
        ]
    else:
        aux_coords_and_dims = (
            None  # no auxiliary coordinates for other variables
        )

    cube = iris.cube.Cube(
        data,
        long_name=data_obj.attributes().get("long_name", raw_name),
        var_name=short_name,
        units=units_str,
        dim_coords_and_dims=[
            (time_coord, 0),
            (lat_coord, 1),
            (lon_coord, 2),
        ],
        aux_coords_and_dims=aux_coords_and_dims,
    )
    return cube


def convert(
    cube: iris.cube.Cube,
    cmor_info,
    attrs,
) -> iris.cube.Cube:
    """Convert cube to CMOR standards based on data type and cmor_info."""

    if attrs.get("reference") is None:
        attrs["reference"] = cmor_info.attributes["reference"]

    cube.convert_units(cmor_info.units)

    if cube.coord("longitude").points.min() < 0:
        # convert from [-180, 180] to [0, 360]
        cube.coord("longitude").points = cube.coord("longitude").points + 180.0
        # roll the data as part of the longitude conversion to maintain correct order (assuming regular grid and that longitude is the last dimension)
        nlon = len(cube.coord("longitude").points)
        dim_lon = 2
        cube.data = da.roll(cube.core_data(), int(nlon / 2), axis=dim_lon)

    if np.diff(cube.coord("latitude").points)[0] < 0:
        # convert [90,-90] to [-90,90]
        cube.coord("latitude").points = cube.coord("latitude").points[::-1]
        # flip the data
        cube.data = cube.data[:, ::-1, :]  # latitude is axis=1

    utils.set_global_atts(cube, attrs)
    utils.fix_var_metadata(cube, cmor_info)
    utils.fix_dim_coordnames(cube)

    utils.fix_coords(cube)

    return cube


def _extract_variable(short_name, var, cfg, in_dir, out_dir):
    attrs = cfg["attributes"]
    attrs["mip"] = var["mip"]
    files = attrs["files"]
    raw_name = var.get("raw_name", short_name)
    logger.error(attrs)
    cmor_table = CMOR_TABLES[attrs["project_id"]]
    cmor_info = cmor_table.get_variable(var["mip"], short_name)

    logger.info("CMORizing variable '%s' from file(s) '%s'", short_name, files)
    # CALIOP has three sets of files: AllSky_Night, CloudFree_Day, and CloudFree_Night
    # I presume that the best picture of od550aer would include all three of these, but I should ask Ruth.

    """Extract variable."""
    # load data
    filepaths = list(Path(in_dir).glob(files))

    years_D = group_files_by_year(filepaths)

    for year, filepaths in years_D.items():
        cubes = iris.cube.CubeList()
        for filepath in filepaths:
            filepath = str(filepath)
            month = get_month_from_filepath(filepath)
            logger.info(f"Year: {year}, Month: {month}, Filepath: {filepath}")
            cube = read_hdf(
                in_dir=in_dir,
                filepath=filepath,
                year=year,
                month=month,
                raw_name=raw_name,
                short_name=short_name,
            )
            cubes.append(cube)

        cube = cubes.concatenate_cube()
        cube = convert(cube, cmor_info, attrs)
        cube_attrs = cube.attributes.globals
        cube_attrs["dataset_id"] = attrs["dataset_id"]
        logger.debug(f"Saving variable {cube} with attributes: {cube_attrs}")

        utils.save_variable(
            cube,
            short_name,
            out_dir,
            cube_attrs,
            unlimited_dimensions=["time"],
        )


def cmorization(in_dir, out_dir, cfg, cfg_user, start_date, end_date):
    """Run CMORizer for MISR."""
    cfg.pop("cmor_table")

    for short_name, var in cfg["variables"].items():
        _extract_variable(short_name, var, cfg, in_dir, out_dir)
