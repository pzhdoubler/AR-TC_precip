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
    lon = sorted([round((l + 179.95) * 10) for l in lon])
    lat = sorted([round((-1*l + 89.95) * 10) for l in lat])   # lat must be flipped
    lon_range = lon[1] - lon[0]
    lat_range = lat[1] - lat[0]


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

    # create grid to sort highest percentiles
    percentile_grid = np.zeros((lat_range, lon_range, max_percentile_len + 1), dtype=np.float32)
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


    # get and save percentiles for each grid point
    for p in range(len(percentiles)):
        percentile_slice = percentile_grid[:,:,percentile_indexes[p]]
        percentile_str = str(percentiles[p]).replace(".","-")
        try:
            print(f"Saving ./percentiles/{title}_{target_season}_top_{percentile_str}_percentile.csv...")
            df = pd.DataFrame(percentile_slice, index=lat_labels, columns=lon_labels)
            df.index.name="LAT/LON"
            df.to_csv(f"./percentiles/{title}_{target_season}_top_{percentile_str}_percentile.csv")
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
    LON = [-133.95,-56.95]
    LAT = [20.95,54.05]
    title = f"USA"
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


    if not os.path.exists("percentiles"):
        os.makedirs("percentiles")

    # get and save percentiles for each grid point
    for p in range(len(percentiles)):
        percentile_slice = percentile_grid[:,:,percentile_indexes[p]]
        percentile_str = str(percentiles[p]).replace(".","-")
        try:
            fname = f"{title}_{target_season}_top_{percentile_str}_percentile.nc"
            print(f"Saving ./percentiles/{fname}...")
            ds = netCDF4.Dataset(f"percentiles/{fname}", 'w', format='NETCDF4')
            # Define dimensions
            ds.createDimension('south_north', lon.shape[0])
            ds.createDimension('west_east', lon.shape[1])

            # Define variables
            lat_var = ds.createVariable('LAT', 'f4', ('south_north', 'west_east'), zlib=True)
            lon_var = ds.createVariable('LONG', 'f4', ('south_north', 'west_east'), zlib=True)
            precip_var = ds.createVariable('PRTHRESH', 'f4', ('south_north', 'west_east'), zlib=True)

            # Set attributes to match original
            lat_var.units = "degree_north"
            lat_var.description = "latitude, south is negative"
            lat_var.MemoryOrder = "XY"
            lat_var.FieldType = 104
            lat_var.stagger = ""

            lon_var.units = "degree_east"
            lon_var.description = "longitude, west is negative"
            lon_var.MemoryOrder = "XY"
            lon_var.FieldType = 104
            lon_var.stagger = ""

            precip_var.units = "mm/day"
            precip_var.long_name = "Extreme Precip Threshold"

            # Write data
            lat_var[:, :] = lat
            lon_var[:, :] = lon
            precip_var[:, :] = percentile_slice

            # Global attributes (optional)
            ds.title = f"Top {percentile_str} Percentile Extreme Precipitation"
            ds.source = "Generated by custom script"
            ds.close()
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


#MSWEP("../MSWEP_daily", 3)

for s in range(4):
    DMET("../OBS/", "/ocean/projects/ees210011p/shared/zafix5/wrfout_d01_1979-12-31_00:00:00", s)
