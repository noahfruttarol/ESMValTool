"""ESMValTool CMORizer for WOA data.

Tier
   Tier 2: other freely-available dataset.

Source
   WOA v2023: https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/
   WOA v2018: https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA
   WOA v2013_v2: https://www.ncei.noaa.gov/data/oceans/woa/WOA13/DATAv2

Last access
   WOA18: 20210311
   WOA23: 20240911

Download and processing instructions
   All handled by the script (download only if local data are missing)

   Alternatively, download the following files:
     temperature/netcdf/decav81B0/1.00/woa18_decav81B0_t00_01.nc
     salinity/netcdf/decav81B0/1.00/woa18_decav81B0_s00_01.nc
     oxygen/netcdf/all/1.00/woa18_all_o00_01.nc
     nitrate/netcdf/all/1.00/woa18_all_n00_01.nc
     phosphate/netcdf/all/1.00/woa18_all_p00_01.nc
     silicate/netcdf/all/1.00/woa18_all_i00_01.nc
   (To get WOA13, replace filenames prefix woa18 with woa13)
   (To get WOA23, replace filenames prefix woa18 with woa23)


Modification history
   20250911-webb_kristi: fix calculation of thetao, so, add bigthetao
   20250603-webb_kristi: fix WOA23 cmorization
   20240911-webb_kristi: handle WOA18/WOA13/WOA23, raw data download, use OBS6
   20210311-lovato_tomas: handle WOA18/WOA13, raw data download, use OBS6
   20200911-bock_lisa: extend to WOA18
   20190328-lovato_tomas: cmorizer revision
   20190131-predoi_valeriu: adapted to v2.
   20190131-demora_lee: written.
"""

import glob
import logging
import os
from copy import deepcopy
from pprint import pformat
from warnings import catch_warnings, filterwarnings

import cftime
import iris
import numpy as np
from cf_units import Unit

from esmvaltool.cmorizers.data.utilities import (
    fix_coords,
    fix_var_metadata,
    save_variable,
    set_global_atts,
)

logger = logging.getLogger(__name__)


def _fix_data(cube, cmor_var_info=None, custom_units=None):
    """Specific data fixes for different variables."""
    logger.info("Fixing data ...")
    logger.info(
        f"Current units: {cube.units}, cmor units: {cmor_var_info.units}. Specified units: {custom_units}"
    )

    if custom_units is not None:
        logger.info(f">>> Manually setting units to {custom_units}")
        cube.units = custom_units

    if (cmor_var_info is not None) and (cube.units != cmor_var_info.units):
        logger.info(
            f'>>> converting from units "{cube.units}" to "{cmor_var_info.units}"'
        )
        try:
            cube.convert_units(cmor_var_info.units)
        except ValueError:
            if cube.units == "micromole / kilogram":
                # apply conversion factor manually
                density_seawater = 1000  # kg/m3
                cube *= density_seawater
                cube.units = "micromole / m3"
                cube.convert_units(cmor_var_info.units)

    return cube


def convert(
    return_var, salinity, insitu_temp, depth, grav=9.81, rho_bsq=1035.0
):
    """
    Source: https://teos-10.github.io/GSW-Python/gsw_flat.html

    SP
    Practical Salinity (PSS-78), unitless

    p
    Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    lon
    Longitude, -360 to 360 degrees

    lat
    Latitude, -90 to 90 degrees

    Note: Following McDougall et al. (2021; GMD) our TEOS-10 based model's native salinity
    variable should be interpreted as Preformed Salinity = S* = Sstar. Thus we should compare
    our salinity against observations converted to S*, not to the Absolute Salinity variable
    # obtained from the McDougall et al. (2012) algorithm encapsulated in SA_from_SP.

    Note: Geoff Stanley's email correspondence with Trevor McDougall on 2025-09-05 confirms that
    we should use Sstar in the functions gsw.pt0_from_t and gsw.CT_from_pt.

    Note: gsw works best with xarray datasets

    """

    import gsw

    # SP : Practical Salinity (PSS-78), unitless
    salinity.convert_units("1")

    # iT : In-situ temperature (ITS-90), degrees C
    insitu_temp.convert_units("degC")

    # P : Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    # Pressure at each depth level (same lat assumed for all in WOA grid)
    sea_pressure = depth_to_pressure(depth, grav=grav, rho_bsq=rho_bsq)

    lons = salinity.coord("longitude").points
    lats = salinity.coord("latitude").points
    # Make broadcastable 3D fields
    lon2d, lat2d = np.meshgrid(lons, lats)
    shape = insitu_temp.shape
    ndim = len(shape)
    if ndim == 3:  # depth × lat × lon
        ntime, nz, ny, nx = 1, *shape
    elif ndim == 4:  # time × depth × lat × lon
        ntime, nz, ny, nx = shape
    else:
        raise ValueError(f"Unexpected input shape {shape}")
    sea_pressure4d = np.broadcast_to(sea_pressure[:, None, None], shape)

    # get data arrays from cubes
    salinity_data = salinity.core_data()
    insitu_temp_data = insitu_temp.core_data()

    # Expand lon/lat to 4D if needed
    lon4d = np.broadcast_to(lon2d, (ntime, nz, ny, nx))
    lat4d = np.broadcast_to(lat2d, (ntime, nz, ny, nx))

    # Sstar: preformed salinity, g/kg
    Sstar = gsw.Sstar_from_SP(salinity_data, sea_pressure4d, lon4d, lat4d)

    # pt: potential temperature with reference sea peressure (p_ref) = 0 dbar, degC
    pt0 = gsw.pt0_from_t(Sstar, insitu_temp_data, sea_pressure4d)

    # CT: conservative temperature, degC
    CT = gsw.CT_from_pt(Sstar, pt0)

    comment = f"Converted by GSW v{gsw.__version__} from objectively analyzed mean fields for sea_water_temperature and sea_water_practical_salinity at standard depth levels converted to Pressure using a Boussinesq reference density of {rho_bsq:.4f} kg/m3 and gravitational acceleration of {grav:.4f} m/s2."

    # Build output cubes

    if return_var == "thetao":
        cube = insitu_temp.copy(data=pt0)
        cube.standard_name = "sea_water_potential_temperature"
        cube.long_name = "Sea water potential temperature"
        cube.var_name = "thetao"
        cube.units = "degC"
        cube.attributes["comment"] = comment

    elif return_var == "bigthetao":
        cube = insitu_temp.copy(data=CT)
        cube.standard_name = "sea_water_conservative_temperature"
        cube.long_name = "Sea water conservative temperature"
        cube.var_name = "bigthetao"
        cube.units = "degC"
        cube.attributes["comment"] = comment

    elif return_var == "so":
        cube = insitu_temp.copy(data=Sstar)
        cube.standard_name = "sea_water_salinity"
        cube.long_name = "Sea water preformed salinity"
        cube.var_name = "so"
        cube.units = "g/kg"
        cube.attributes["comment"] = comment
    else:
        raise Exception(f'Cannot return "{return_var}"')

    return cube


def depth_to_pressure(depth, grav=9.81, rho_bsq=1035.0):
    """
    Convert depth (Z, in metres) to pressure (P, in dbar) using hydrostatic balance
    with fixed Boussinesq reference density and gravitational acceleration.

    Parameters
    ----------
    depth : array-like
        Depth, metres (positive and increasing down)

    grav: 9.81 m/s2, gravitational accelleration
    rho_bsq: 1035.0 kg.m-3, Boussinesq reference density

    Returns
    -------
    P : array-like
        Pressure, dbar
    """
    Pa2dbar = 1e-4  # conversion factor, dbar.Pa-1
    depth_m = depth.points if isinstance(depth, iris.coords.Coord) else depth
    return rho_bsq * grav * depth_m * Pa2dbar  # dbar


def collect_files(in_dir, var_info, cfg):
    """Compose input file list and download if missing."""

    attrs = cfg["attributes"]

    # follow ESMValTool directory structure convention
    # rootpath / Tier{tier}/{dataset}/{version}/{frequency}/{short_name}
    # in_dir  = rootpath / Tier{tier}/{dataset}
    in_dir = os.path.join(
        in_dir, attrs["version"], var_info["frequency"], var_info["name"]
    )
    fname = f"{attrs['short_name'].lower()}_{var_info['file']}*.nc"

    in_file = os.path.join(in_dir, fname)

    files = sorted(glob.glob(in_file))
    if len(files) == 0:
        raise Exception(f"No files found, {in_file}")

    logger.info(f"Input files: {','.join(files)}")

    return files


def infer_reference_year(cube, files, var):
    """
    infer reference year from time start and duration

    If multiple files merged, handle the case where attributes are not propagated due to mismatch.

    Handle case where either duration or end is specified
    """

    cube_attrs = cube.attributes.globals
    start = cube_attrs.get("time_coverage_start")
    duration = cube_attrs.get("time_coverage_duration")
    end = cube_attrs.get("time_coverage_duration")

    def get_from_multiple_files(key):
        # read attribute from all input files
        values = [
            iris.load_cube(f, var).attributes.globals.get(key) for f in files
        ]
        # extract just the year
        years = [int(v.split("-")[0]) for v in values]
        if np.unique(years).size == 1:
            return years[0]
        else:
            raise Exception(
                "Cannot infer reference year for files which have different start years."
            )

    # if start is null, likely because different between files merged
    if start is None:
        start_year = get_from_multiple_files("time_coverage_start")
    else:
        start_year = int(start.split("-")[0])

    try:
        duration_year = int(duration.removeprefix("P").removesuffix("Y"))
    except ValueError:
        # infer from end
        try:
            end_year = int(end.split("-")[0])
        except ValueError:
            end_year = get_from_multiple_files("time_coverage_end")

        duration_year = end_year - start_year

    ref_year = start_year + int(duration_year / 2)
    logger.info(f">>> setting reference year to {ref_year}")

    return ref_year


def extract_variable(infiles_d, out_dir, attrs, raw_var_info, cmor_table):
    """Extract variables and create OBS dataset."""

    short_name = raw_var_info["short_name"]
    cmor_var_info = cmor_table.get_variable(raw_var_info["mip"], short_name)

    cubes_d = {}
    for var, var_d in infiles_d.items():
        with catch_warnings():
            filterwarnings(
                action="ignore",
                message="Ignoring netCDF variable .* invalid units .*",
                category=UserWarning,
                module="iris",
            )
            cubes = iris.load(var_d["files"], var_d["raw_var"])
        iris.util.equalise_attributes(cubes)
        cube = cubes.concatenate_cube()

        # apply fixes to data
        # do this BEFORE updating metadata, need to manually resdolve unknown units
        # do before conversions
        _fix_data(cube, cmor_var_info, var_d.get("units"))

        cubes_d[var] = cube

    # derive variables
    if short_name in ["thetao", "bigthetao", "so"]:
        cube = convert(
            short_name,
            salinity=cubes_d["salinity"],
            insitu_temp=cubes_d["temperature"],
            depth=cubes_d["temperature"].coord("depth").points,
            grav=raw_var_info.get("grav"),
            rho_bsq=raw_var_info.get("rho_bsq"),
        )

    cube_attrs = cube.attributes.globals
    cube_attrs.update(attrs)
    cube_attrs0 = deepcopy(cube_attrs)  # make copy of original

    ref_year = infer_reference_year(cube, var_d["files"], var_d["raw_var"])

    # set reference time
    cube.coord("time").climatological = False
    # if cmor_var_info.frequency == 'mon':
    if raw_var_info.get("frequency") == "mon":
        dates = [
            cftime.DatetimeGregorian(ref_year, m, 1) for m in range(1, 13)
        ]
    # elif cmor_var_info.frequency == 'yr':
    elif raw_var_info.get("frequency") == "yr":
        dates = [cftime.DatetimeGregorian(ref_year, 6, 15)]

    calendar = "gregorian"
    cube.coord("time").points = Unit(
        "days since 1950-01-01 00:00:00", calendar=calendar
    ).date2num(dates)
    cube.coord("time").units = Unit(
        "days since 1950-01-01 00:00:00", calendar=calendar
    )

    fix_var_metadata(cube, cmor_var_info)
    fix_coords(cube)
    set_global_atts(cube, cube_attrs0)

    # add back key attrs
    for key in (
        "time_coverage_start",
        "time_coverage_duration",
        "time_coverage_resolution",
    ):
        if key in cube_attrs0:
            cube.attributes.globals[key] = cube_attrs0[key]

    save_variable(
        cube, short_name, out_dir, cube_attrs0, unlimited_dimensions=["time"]
    )

    # derive ocean surface
    if raw_var_info.get("srf_var"):
        cmor_var_info = cmor_table.get_variable(
            raw_var_info["mip"], raw_var_info["srf_var"]
        )
        logger.info("Extract surface OBS for %s", raw_var_info["srf_var"])
        level_constraint = iris.Constraint(cube.var_name, depth=0)
        cube_os = cube.extract(level_constraint)
        fix_var_metadata(cube_os, cmor_var_info)
        save_variable(
            cube_os,
            raw_var_info["srf_var"],
            out_dir,
            cube_attrs0,
            unlimited_dimensions=["time"],
        )


def cmorization(in_dir, out_dir, cfg, cfg_user, start_date, end_date):
    """Cmorization func call."""
    cmor_table = cfg["cmor_table"]
    glob_attrs = cfg["attributes"]

    # run the cmorization
    for var, var_info in cfg["variables"].items():
        if glob_attrs.get("version") not in var:
            continue

        logger.info(
            "CMORizing var %s from input set %s", var, var_info["name"]
        )

        var_info.update(cfg["custom"])
        logger.info("\n" + pformat(var_info))

        try:
            # organize input files
            in_files_d = {
                var_info["name"]: {
                    "files": collect_files(in_dir, var_info, cfg),
                    "short_name": var_info["short_name"],
                    "raw_var": var_info["raw_var"],
                    "units": var_info.get("units"),
                },
            }

            for supp_var_info in var_info.get("supplimentary_vars", []):
                supp_var_info["frequency"] = var_info.get("frequency")
                in_files_d[supp_var_info["name"]] = {
                    "files": collect_files(in_dir, supp_var_info, cfg),
                    "raw_var": supp_var_info["raw_var"],
                    "units": supp_var_info.get("units"),
                }

            glob_attrs["mip"] = var_info["mip"]

            extract_variable(
                in_files_d, out_dir, glob_attrs, var_info, cmor_table
            )

        except Exception as e:
            logger.error(e)
