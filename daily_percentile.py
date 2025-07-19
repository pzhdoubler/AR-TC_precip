import numpy as np
import os
import sys
import time
import datetime
import netCDF4
import pandas as pd

from helpers import *

def MSWEP_get_top_percentiles(files, lon, lat, percentiles, title = "", target_season=""):
    # parse lon, lat
    # assume they are passed sorted
    lon[1] += 0.1
    lat[0] -= 0.1
    lon_labels = np.arange(lon[0], lon[1], 0.1)
    lat_labels = np.arange(lat[1], lat[0], -0.1)
    lon2d, lat2d = np.meshgrid(lon_labels, lat_labels)
    lon = sorted([round((l + 179.95) * 10) for l in lon])
    lat = sorted([round((-1*l + 89.95) * 10) for l in lat])   # lat must be flipped

    print(lon2d.shape)
    print()

    # parse percentiles
    max_percentile = max(percentiles)
    max_percentile_len = int(len(files) * (max_percentile/100))
    # find indices of each percentile
    percentile_indexes = []
    for p in percentiles:
        if p == max_percentile:
            percentile_indexes.append(1)
        else:
            i = int(np.ceil(max_percentile_len * (1 - (p/max_percentile))))
            if i < max_percentile_len + 1:
                percentile_indexes.append(i)
            else:
                print(f"Data not large enough to take top {p} percentile. Quitting...")
                return

    # create grid to sort highest percnentiles
    percentile_grid = np.zeros((lon2d.shape[0], lon2d.shape[1], max_percentile_len + 1), dtype=np.float32)
    start = time.time()
    init = True
    print("Getting top percentiles ...")
    # get top percentiles
    for f in range(len(files)):
        try:
            # print progress
            if len(files) > 100 and f % (len(files) // 100) == 0:
                print((f/len(files))*100, "% complete")

            data = netCDF4.Dataset(files[f],mode='r')

            # get lons and lats of region
            precip = data.variables["precipitation"][:][0][lat[0]:lat[1],lon[0]:lon[1]]
            precip = np.ma.masked_where(precip <= -9999.0, precip)
            
            # put new data in extra slot, sort, next run will replace smallest data
            percentile_grid[:,:,0] = precip
            percentile_grid.sort()

            # print approximate runtime
            if init:
                end = time.time()
                print(f"One file took {end - start} seconds")
                print(f"All files estimated to take {len(files) * (end - start)} seconds")
                init = False
        except Exception as e:
            print(f"Error with file {files[f]}")
            print(e)

    path = "percentiles/"
    if not os.path.exists(path):
        os.makedirs(path)

    # get and save percentiles for each grid point
    for p in range(len(percentiles)):
        percentile_slice = percentile_grid[:,:,percentile_indexes[p]]
        percentile_str = str(percentiles[p]).replace(".","-")
        fname = f"{title}_{target_season}_top_{percentile_str}_percentile.nc"
        try:
            save_NETCDF(lon2d, lat2d, percentile_slice, 
                        "PRTHRESH", 
                        "mm/day", 
                        f"Top {percentile_str} Extreme Precip Threshold", 
                        path, fname, 
                        f"Top {percentile_str} Percentile Extreme Precipitation"
                    )
        except Exception as e:
            print(f"Error saving {title}_{target_season}_top_{percentile_str}_percentile.csv")
            print(e)


def MSWEP(FOLDER, target_season):
    # get list of all files
    files = []

    # divide into seasons
    # DJF 0, MAM 1, JJA 2, SON 3
    season = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
    season_labels = ["DJF", "MAM", "JJA", "SON"]

    for r,d,f in os.walk(FOLDER):
        for i in range(len(f)):
            dt = parse_precip_filename(f[i])
            if season[dt.month] == target_season:
                files.append(f"{r}/{f[i]}")

    # US lat/lon box
    LON = [-138.95,-52.95]
    LAT = [15.95,57.05]
    title = f"MSWEP"
    percentiles = [5.0, 4.0, 3.0, 2.0, 1.0, 0.1, 0.01]

    # world lat/lon box
    # LON = [-179.95, 179.95]
    # LAT = [-89.95, 89.95]
    # title = "WORLD"
    # percentiles = [1.0, 0.1, 0.01]

    MSWEP_get_top_percentiles(files, LON, LAT, percentiles, title, season_labels[target_season])


def DMET_get_top_percentiles(files, lon, lat, percentiles, target_season, title = ""):
    # time is days since 1900-01-01
    # find how many days are in target_season in range
    ref_time = datetime.datetime(1900,1,1)
    season_labels = ["DJF", "MAM", "JJA", "SON"]
    season = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
    season_index = season_labels.index(target_season)
    season_days = 0
    for i, f in enumerate(files):
        data = netCDF4.Dataset(f,mode='r')
        times = data.variables["time"]
        times = np.ma.masked_invalid(times)
        for t in times:
            dt = ref_time + datetime.timedelta(days=t)
            if season[dt.month] == season_index:
                season_days += 1

    # parse percentiles
    max_percentile = max(percentiles)
    # assumes ~91 days in a season
    max_percentile_len = int(season_days * (max_percentile/100))
    # find indices of each percentile
    percentile_indexes = []
    for p in percentiles:
        if p == max_percentile:
            percentile_indexes.append(1)
        else:
            i = int(np.ceil(max_percentile_len * (1 - (p/max_percentile))))
            if i < max_percentile_len + 1:
                percentile_indexes.append(i)
            else:
                print(f"Data not large enough to take top {p} percentile. Quitting...")
                return

    # create grid to sort highest percentiles
    percentile_grid = np.zeros((lon.shape[0], lon.shape[1], max_percentile_len + 1), dtype=np.float32)
    start = time.time()
    init = True
    print("Getting top percentiles ...")
    # get top percentiles
    for f in range(len(files)):
        try:
            # print progress
            if len(files) > 100 and f % (len(files) // 100) == 0:
                print((f/len(files))*100, "% complete")

            data = netCDF4.Dataset(files[f],mode='r')
            times = data.variables["time"]
            times = np.ma.masked_invalid(times)
            for i, t in enumerate(times):
                # calc time and only include for interested seasons
                dt = ref_time + datetime.timedelta(days=t)
                if season[dt.month] == season_index:
                    # get lons and lats of region
                    precip = data.variables["PRAVG"][0,:,:]
                    precip = np.ma.masked_invalid(precip)

                    # put new data in extra slot, sort, next run will replace smallest data
                    percentile_grid[:,:,0] = precip
                    percentile_grid.sort()

            # print approximate runtime
            if init:
                end = time.time()
                print(f"One file took {end - start} seconds")
                print(f"All files estimated to take {len(files) * (end - start)} seconds")
                init = False
        except Exception as e:
            print(f"Error with file {files[f]}")
            print(e)


    path = "percentiles/"
    if not os.path.exists(path):
        os.makedirs(path)

    # get and save percentiles for each grid point
    for p in range(len(percentiles)):
        percentile_slice = percentile_grid[:,:,percentile_indexes[p]]
        percentile_str = str(percentiles[p]).replace(".","-")
        fname = f"{title}_{target_season}_top_{percentile_str}_percentile.nc"
        try:
            save_NETCDF(lat, lon, percentile_slice, 
                        "PRTHRESH", 
                        "mm/day", 
                        "Extreme Precip Threshold", 
                        path, fname, 
                        f"Top {percentile_str} Percentile Extreme Precipitation"
                    )
        except Exception as e:
            print(f"Error saving {fname}")
            print(e)

def DMET(FOLDER, config_file, target_season):
    # get list of all files
    files = [FOLDER + f for f in os.listdir(FOLDER)]

    # divide into seasons
    # DJF 0, MAM 1, JJA 2, SON 3
    season_labels = ["DJF", "MAM", "JJA", "SON"]

    # get lat lon
    ds = netCDF4.Dataset(config_file,mode='r')
    LON = ds.variables["XLONG"][0,:,:]
    LAT = ds.variables["XLAT"][0,:,:]

    title = f"DMET"
    percentiles = [5.0, 4.0, 3.0, 2.0, 1.0, 0.1, 0.01]

    print(f"Getting top percentiles for {season_labels[target_season]} ...")
    DMET_get_top_percentiles(files, LON, LAT, percentiles, season_labels[target_season], title)




for s in range(4):
    MSWEP("../MSWEP_daily", s)
    #DMET("../OBS/", "/ocean/projects/ees210011p/shared/zafix5/wrfout_d01_1979-12-31_00:00:00", s)
