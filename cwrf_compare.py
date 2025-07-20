import numpy as np
import os
import sys
import time
import datetime
import netCDF4
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

from helpers import *


def CONUS_mask(lon2d, lat2d, states=[]):
    reader = shpreader.natural_earth(resolution='110m',
                                     category='cultural',
                                     name='admin_1_states_provinces_lakes_shp')

    records = shpreader.Reader(reader).records()

    # List of contiguous US states (excluding Alaska, Hawaii, and territories)
    if len(states) == 0:
        conus_states = set([
            'Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
            'Delaware', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
            'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
            'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming'
        ])
    else:
        conus_states = set(states)

    # Combine geometries for CONUS
    conus_geom = sgeom.MultiPolygon([
        rec.geometry for rec in records if rec.attributes['name'] in conus_states
    ])

    # Flatten grid
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    # Build mask
    mask_flat = np.array([
        conus_geom.contains(sgeom.Point(lon, lat)) for lon, lat in zip(lon_flat, lat_flat)
    ])
    return mask_flat.reshape(lon2d.shape)


# makes maps of CWRF - OBS extreme precip for 4 specific years, in provided season, and on provided states
def map_CWRF_diffs(FIGURE_FOLDER, years, season, data_type, obs_set, percentile, states=[], map_extent=[]):
    if data_type == "frequencies":
        data_label = "freq"
        data_key = "EX-PR-FRQ"
        data_title = "Frequency"
        data_units = "# events"
    elif data_type == "intensities":
        data_label = "avg-intensity"
        data_key = "EX-PR-INTENS"
        data_title = "Average Intensity"
        data_units = "mm/day"
    else:
        print("ERROR: invalid data_type passed")
        return
    
    per_title = f"{obs_set}{str(percentile).replace(".","-")}"
    if obs_set == "MSWEP" or obs_set == "DMET":
        obs_path = f"./{data_type}/{obs_set}-on-{obs_set}/{per_title}/"
        cwrf_path = f"./{data_type}/CWRF-on-{obs_set}/{per_title}/"
    else:
        print("ERROR: invalid obs set passed")
        return
    
    # set up data parse
    print("Reading in data and making mask...")
    init_file = f"{obs_path}{years[0]}_{season}_{data_label}.nc"
    ds = netCDF4.Dataset(init_file, m='r')
    lon2d = ds.variables["LONG"][:,:]
    lat2d = ds.variables["LAT"][:,:]
    ds.close()
    mask = CONUS_mask(lon2d, lat2d, states)
    data = np.ma.masked_all((len(years), lon2d.shape[0], lon2d.shape[1]))
    # read in data
    for i, year in enumerate(years):
        ds = netCDF4.Dataset(f"{obs_path}{year}_{season}_{data_label}.nc")
        obs = ds.variables[data_key][:,:]
        ds.close()
        ds = netCDF4.Dataset(f"{cwrf_path}{year}_{season}_{data_label}.nc")
        cwrf = ds.variables[data_key][:,:]
        ds.close()

        diff = cwrf - obs
        diff = np.ma.masked_where(~mask, diff)
        data[i] = diff

    print("Plotting data ...")
    # Set up figure and axes
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    fig.suptitle(f"Top {percentile} Percentile {season} Precip {data_title} Gridwise Difference (CWRF - {obs_set})")
    abs_max = np.max(np.abs(data))
    axes = axes.flatten()
    if len(years) != len(axes):
        print(f"ERROR: must pass exactly {len(axes)} years")
        return

    # plot years
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(lon2d, lat2d, data[i], cmap="coolwarm", transform=ccrs.PlateCarree(), vmin=-abs_max, vmax=abs_max)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_title(years[i] + season)
        if len(map_extent) == 4:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(f"Extreme Precip {data_title} ({data_units})")

    fig_path = f"./{FIGURE_FOLDER}/{obs_set}{per_title}"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    plt.savefig(f"{fig_path}/cwrf_diff_{str.join("-", years)}.png")
    plt.clf()

# makes time series of avg CWRF - OBS extreme precip for year range on provided states
def time_series_CWRF_diffs(FIGURE_FOLDER, year_range, data_type, percentile, obs_set, states=[]):
    if data_type == "frequencies":
        data_label = "freq"
        data_key = "EX-PR-FRQ"
        data_title = "Frequency"
        data_units = "# events"
    elif data_type == "intensities":
        data_label = "avg-intensity"
        data_key = "EX-PR-INTENS"
        data_title = "Average Intensity"
        data_units = "mm/day"
    else:
        print("ERROR: invalid data_type passed")
        return
    
    per_title = f"{obs_set}{str(percentile).replace(".","-")}"
    if obs_set == "MSWEP" or obs_set == "DMET":
        obs_path = f"./{data_type}/{obs_set}-on-{obs_set}/{per_title}/"
        cwrf_path = f"./{data_type}/CWRF-on-{obs_set}/{per_title}/"
    else:
        print("ERROR: invalid obs set passed")
        return
    
    init_file = f"{obs_path}{year_range[0]}_DJF_{data_label}"
    ds = netCDF4.Dataset(init_file, m='r')
    lon2d = ds.variables["LONG"][:,:]
    lat2d = ds.variables["LAT"][:,:]
    ds.close()
    mask = CONUS_mask(lon2d, lat2d, states)

    for year in range(year_range):
        pass


#########################################################################################################

FIG_FOLDER = "figures"

# make diff maps
years = [1981, 1993, 2005, 2017]
season = "DJF"
data_type = "intensities"
obs_set = "MSWEP"
percentile = 5.0
states = []
map_extent = []
map_CWRF_diffs(FIG_FOLDER, years, season, data_type, obs_set, percentile, states, map_extent)


