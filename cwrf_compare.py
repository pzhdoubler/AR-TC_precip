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
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from helpers import *


def CONUS_mask(lon2d, lat2d, states=[]):
    reader = shpreader.natural_earth(resolution='110m',
                                     category='cultural',
                                     name='admin_1_states_provinces')

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

    print("Getting geometries ...")
    # Combine geometries for CONUS
    polygons = []
    for rec in records:
        if rec.attributes['name'] in conus_states:
            geom = rec.geometry
        else:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(geom.geoms)

    conus_geom = unary_union(polygons)

    # Flatten grid
    lon_flat = lon2d.flatten()
    lat_flat = lat2d.flatten()

    print("Building mask ...")
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
        mesh = ax.pcolormesh(lon2d, lat2d, data[i], cmap="seismic", transform=ccrs.PlateCarree(), vmin=-abs_max, vmax=abs_max)
        ax.add_feature(cfeature.BORDERS, linewidth=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
        ax.set_title(str(years[i]) + season)
        if len(map_extent) == 4:
            ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    cbar = fig.colorbar(mesh, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(f"Extreme Precip {data_title} ({data_units})")

    # save figure
    fig_path = f"./{FIGURE_FOLDER}/{per_title}"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fname = f"cwrf_{season}-{data_label}_diff_{str.join("-", [str(y) for y in years])}.png"
    plt.savefig(f"{fig_path}/{fname}")
    plt.clf()
    print(f"Saved {fname}.")

# makes time series of avg CWRF - OBS extreme precip for year range on provided states
def annual_time_series_CWRF_diffs(FIGURE_FOLDER, year_range, data_type, obs_set, percentile, states=[], mask_name="CONUS"):
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
    
    print("Reading in data and making mask...")
    init_file = f"{obs_path}{year_range[0]}_DJF_{data_label}.nc"
    ds = netCDF4.Dataset(init_file, m='r')
    lon2d = ds.variables["LONG"][:,:]
    lat2d = ds.variables["LAT"][:,:]
    ds.close()
    mask = CONUS_mask(lon2d, lat2d, states)
    season_labels = ["DJF", "MAM", "JJA", "SON"]
    years = np.arange(year_range[0], year_range[1], 1)
    season_data = np.zeros((2,4,len(years)))

    # read in data
    for y, year in enumerate(years):
        for s, season in enumerate(season_labels):
            season_data[0,s,y] = year + s*0.25

            ds = netCDF4.Dataset(f"{obs_path}{year}_{season}_{data_label}.nc")
            obs = ds.variables[data_key][:,:]
            ds.close()
            ds = netCDF4.Dataset(f"{cwrf_path}{year}_{season}_{data_label}.nc")
            cwrf = ds.variables[data_key][:,:]
            ds.close()

            diff = cwrf - obs
            diff = np.ma.masked_where(~mask, diff)
            season_data[1,s,y] = np.mean(diff)

    season_colors = ["darkblue", "hotpink", "lime", "darkgoldenrod"]
    print("Plotting data ...")
    # Set up figure and axes
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10),
        constrained_layout=True
    )
    fig.suptitle(f"Seasonal Average Difference (CWRF - {obs_set}) of Top {percentile} Percentile Precip {data_title} over {mask_name}, {str.join("-", [str(y) for y in year_range])}")
    axes = axes.flatten()

    # plot years
    for i, ax in enumerate(axes):
        # plot other seasons faintly
        for j in range(len(season_labels)):
            if j != i:
                ax.plot(season_data[0,j,:], season_data[1,j,:], color=season_colors[j], alpha=0.1)

        # plot focus season
        ax.plot(season_data[0,i,:], season_data[1,i,:], color=season_colors[i])

        # best fit for focus season
        m, b = np.polyfit(season_data[0,i,:], season_data[1,i,:], 1)
        x = season_data[0,i,:]
        ax.plot(x, m*x + b, color="black", linestyle="--", label=f"{m}x + {b}")
        ax.set_title(season_labels[i])
        ax.set_xlabel("Years")
        ax.set_ylabel(data_units)
        ax.legend()

    # save figure
    fig_path = f"./{FIGURE_FOLDER}/{per_title}"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    fname = f"{mask_name}_cwrf_{data_label}_diff_timeseries.png"
    plt.savefig(f"{fig_path}/{fname}")
    plt.clf()
    print(f"Saved {fname}.")


def seasonal_time_series_CWRF_diffs(FIGURE_FOLDER, year, season, data_type, obs_set, percentile, states=[], mask_name="CONUS"):
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

#########################################################################################################

FIG_FOLDER = "figures"

# make diff maps
years = [1981, 1993, 2005, 2017]
seasons = ["DJF", "MAM", "JJA", "SON"]
data_type = "intensities"
obs_set = "MSWEP"
percentile = 5.0

states = [
            'Alabama', 'Arizona', 'Arkansas', 'Colorado', 'Connecticut',
            'Delaware', 'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
            'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 
            'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
            'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma',
            'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
            'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
            'West Virginia', 'Wisconsin', 'Wyoming'
        ]
mask_name = "Not-West-Coast"
map_extent = [-125, -66, 24, 51]

annual_time_series_CWRF_diffs(FIG_FOLDER, [1981,2020], data_type, obs_set, percentile, states, mask_name)

# for season in seasons:
#     map_CWRF_diffs(FIG_FOLDER, years, season, data_type, obs_set, percentile, states, map_extent)


