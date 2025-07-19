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

from helpers import *

def count_MSWEP_on_MSWEP_percentiles(PERCENTILE_FOLDER, FREQ_FOLDER, INTENSITY_FOLDER, all_files, percentiles):
    # get percentile files
    percentile_title = "MSWEP"
    per_files = [f for f in os.listdir(PERCENTILE_FOLDER) if f.startswith(percentile_title)]
    # get expected shape of  percentile files
    ds = netCDF4.Dataset(f"./percentiles/{per_files[0]}")
    lon2d = ds.variables["LONG"][:,:]
    lat2d = ds.variables["LAT"][:,:]

    # (lat, lon, percentile, season)
    percentile_grids = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles), 4))

    season_labels = ["DJF", "MAM", "JJA", "SON"]
    for p, percentile in enumerate(percentiles):
        p_label = str(percentile).replace(".","-")
        percentile_files = [f for f in per_files if p_label in f]
        for s, season in enumerate(season_labels):
            season_file = next(f for f in percentile_files if season in f)
            ds = netCDF4.Dataset(f"./percentiles/{season_file}")
            percentile_grids[:,:,p,s] = ds.variables["PRTHRESH"][:,:]
            print(f"Read in {season_file} ...")

    # set data recording
    frequency_count = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.int64)
    intensity_sum = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.float64)
    lon_slice = sorted([round((l + 179.95) * 10) for l in [np.min(lon2d), np.max(lon2d) + 0.1]])
    lat_slice = sorted([round((-1*l + 89.95) * 10) for l in [np.max(lat2d), np.min(lat2d) - 0.1]])   # lat must be flipped

    # set up seasons and file order
    season = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
    all_files.sort(key=parse_precip_MSWEP_filename)
    cur_season = season[parse_precip_MSWEP_filename(all_files[0]).month]

    init = True
    for year in range(1979,2021):
        print(f"Counting frequencies for {year} ...")

        year_files = [f for f in all_files if str(year) in f]

        for f, file in enumerate(year_files):
            try:
                dt = parse_precip_MSWEP_filename(file)

                # calc runtime
                start = time.time()

                # save current frequencies if switched to a new season
                if season[dt.month] != cur_season:
                    for p, percentile in enumerate(percentiles):
                        percentile_str = percentile_title + str(percentile).replace(".","-")
                        # frequency path
                        f_name = f"{year}_{season_labels[cur_season]}_freq.csv"
                        f_path = f"./{FREQ_FOLDER}/{percentile_str}/"
                        if not os.path.exists(f_path):
                            os.makedirs(f_path)
                        try:
                            save_NETCDF(lon2d, lat2d, frequency_count[:,:,p], 
                                "EX-PR-FRQ", 
                                "exceedences/season", 
                                f"Top {percentile_str} Extreme Precip Freq", 
                                f_path, f_name, 
                                f"{season_labels[cur_season]} Freq of Top {percentile_str} Percentile Extreme Precipitation"
                            )
                        except Exception as e:
                            print(f"Error saving {f_name}")
                            print(e)

                        # intensity path
                        i_name = f"{year}_{season_labels[cur_season]}_avg-intensity.csv"
                        i_path = f"./{INTENSITY_FOLDER}/{percentile_str}/"
                        if not os.path.exists(i_path):
                            os.makedirs(i_path)
                        try:
                            frequency_count[:,:,p] += frequency_count[:,:,p] == 0    # remove 0 valued counts to avoid div by 0
                            intens_data = intensity_sum[:,:,p] / frequency_count[:,:,p]
                            save_NETCDF(lon2d, lat2d, intens_data, 
                                "EX-PR-INTENS", 
                                "mm/day", 
                                f"Top {percentile_str} Extreme Precip Average Intensity", 
                                i_path, i_name, 
                                f"{season_labels[cur_season]} Avg Intensity of Top {percentile_str} Percentile Extreme Precipitation"
                            )
                        except Exception as e:
                            print(f"Error saving {i_name}")
                            print(e)

                    # reset data
                    frequency_count = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.int64)
                    intensity_sum = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.float64)
                    cur_season = season[dt.month]

                # print progress
                if f % (len(year_files) // 100) == 0:
                    print((f/len(year_files))*100, "% complete", end="\r")

                data = netCDF4.Dataset(year_files[f],mode='r')

                precip = data.variables["precipitation"][:][0][lat_slice[0]:lat_slice[1],lon_slice[0]:lon_slice[1]]
                precip = np.ma.masked_where(precip <= -9999.0, precip)

                # count frequency of each percentile and sum intensities of exceedences
                for p in range(len(percentiles)):
                    # do not count 0 events as exceedences
                    zero_precip_grid = precip > 0.0 
                    exceedence_grid = (precip >= percentile_grids[:,:,p,cur_season]) * zero_precip_grid 
                    frequency_count[:,:,p] += exceedence_grid
                    intensity_sum[:,:,p] += precip*exceedence_grid

                # print approximate runtime
                if init:
                    end = time.time()
                    print(f"One file took {end - start} seconds")
                    print(f"All files estimated to take {(len(all_files) - f) * (end - start)} seconds")
                    init = False

            except Exception as e:
                print(f"Error with file {year_files[f]}")
                print(e)
                continue

def count_DMET_on_DMET_percentiles(PERCENTILE_FOLDER, FREQ_FOLDER, INTENSITY_FOLDER, all_files, percentiles):
    # get percentile files
    percentile_title = "DMET"
    per_files = [f for f in os.listdir(PERCENTILE_FOLDER) if f.startswith(percentile_title)]
    # get expected shape of  percentile files
    ds = netCDF4.Dataset(f"./percentiles/{per_files[0]}")
    lon2d = ds.variables["LONG"][:,:]
    lat2d = ds.variables["LAT"][:,:]

    # (lat, lon, percentile, season)
    percentile_grids = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles), 4))

    season_labels = ["DJF", "MAM", "JJA", "SON"]
    for p, percentile in enumerate(percentiles):
        p_label = str(percentile).replace(".","-")
        percentile_files = [f for f in per_files if p_label in f]
        for s, season in enumerate(season_labels):
            season_file = next(f for f in percentile_files if season in f)
            ds = netCDF4.Dataset(f"./percentiles/{season_file}")
            percentile_grids[:,:,p,s] = ds.variables["PRTHRESH"][:,:]
            print(f"Read in {season_file} ...")

    # set data recording
    frequency_count = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.int64)
    intensity_sum = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.float64)

    # set up seasons and file order
    # files are full years
    # time is days since 1900-01-01
    ref_time = datetime.datetime(1900,1,1)
    season = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
    all_files.sort(key=parse_precip_DMET_filename)
    cur_season = season[parse_precip_DMET_filename(all_files[0]).month]

    init = True
    # calc runtime
    start = time.time()
    for f, file in enumerate(all_files):
        dt = parse_precip_DMET_filename(file)
        year = dt.year
        print(f"Counting frequencies for {dt.year} ...")

        data = netCDF4.Dataset(file,mode='r')
        times = data.variables["time"]
        times = np.ma.masked_invalid(times)

        for i, time in enumerate(times):
            try:
                dt = ref_time + datetime.timedelta(days=time)
                # save current frequencies if switched to a new season
                if season[dt.month] != cur_season:
                    for p, percentile in enumerate(percentiles):
                        percentile_str = percentile_title + str(percentile).replace(".","-")
                        # frequency path
                        f_name = f"{year}_{season_labels[cur_season]}_freq.csv"
                        f_path = f"./{FREQ_FOLDER}/{percentile_str}/"
                        if not os.path.exists(f_path):
                            os.makedirs(f_path)
                        try:
                            save_NETCDF(lon2d, lat2d, frequency_count[:,:,p], 
                                "EX-PR-FRQ", 
                                "exceedences/season", 
                                f"Top {percentile_str} Extreme Precip Freq", 
                                f_path, f_name, 
                                f"{season_labels[cur_season]} Freq of Top {percentile_str} Percentile Extreme Precipitation"
                            )
                        except Exception as e:
                            print(f"Error saving {f_name}")
                            print(e)

                        # intensity path
                        i_name = f"{year}_{season_labels[cur_season]}_avg-intensity.csv"
                        i_path = f"./{INTENSITY_FOLDER}/{percentile_str}/"
                        if not os.path.exists(i_path):
                            os.makedirs(i_path)
                        try:
                            frequency_count[:,:,p] += frequency_count[:,:,p] == 0    # remove 0 valued counts to avoid div by 0
                            intens_data = intensity_sum[:,:,p] / frequency_count[:,:,p]
                            save_NETCDF(lon2d, lat2d, intens_data, 
                                "EX-PR-INTENS", 
                                "mm/day", 
                                f"Top {percentile_str} Extreme Precip Average Intensity", 
                                i_path, i_name, 
                                f"{season_labels[cur_season]} Avg Intensity of Top {percentile_str} Percentile Extreme Precipitation"
                            )
                        except Exception as e:
                            print(f"Error saving {i_name}")
                            print(e)

                    # reset data
                    frequency_count = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.int64)
                    intensity_sum = np.zeros((lon2d.shape[0], lon2d.shape[1], len(percentiles)), dtype=np.float64)
                    cur_season = season[dt.month]

                precip = data.variables["PRAVG"][i, :, :]
                precip = np.ma.masked_invalid(precip)

                # count frequency of each percentile and sum intensities of exceedences
                for p in range(len(percentiles)):
                    # do not count 0 events as exceedences
                    zero_precip_grid = precip > 0.0 
                    exceedence_grid = (precip >= percentile_grids[:,:,p,cur_season]) * zero_precip_grid 
                    frequency_count[:,:,p] += exceedence_grid
                    intensity_sum[:,:,p] += precip*exceedence_grid

            except Exception as e:
                print(f"Error with file {file}, time {time}")
                print(e)
                continue
    
        # print approximate runtime
        if init:
            end = time.time()
            print(f"One file took {end - start} seconds")
            print(f"All files estimated to take {(len(all_files) - f) * (end - start)} seconds")
            init = False

def count_CWRF_on_MSWEP_percentiles(PERCENTILE_FOLDER, FREQ_FOLDER, INTENSITY_FOLDER, all_files, percentiles):
    pass

def count_CWRF_on_DMET_percentiles(PERCENTILE_FOLDER, FREQ_FOLDER, INTENSITY_FOLDER, all_files, percentiles):
    pass

def setup_freq_count():
    pass

percentiles = [5.0, 1.0, 0.1, 0.01]

# MSWEP on MSWEP percentiles
FOLDER = "../MSWEP_daily"
files = []
for r,d,f in os.walk(FOLDER):
    for i in range(len(f)):
        files.append(f"{r}/{f[i]}")
count_MSWEP_on_MSWEP_percentiles("percentiles", "frequencies/MSWEP-on-MSWEP", "intensities/MSWEP-on-MSWEP", files, percentiles)


# DMET on DMET percentiles
# files = ["../OBS/" + f for f in os.listdir("../OBS")]
# count_DMET_on_DMET_percentiles("percentiles", "frequencies/DMET-on-DMET", "intensities/DMET-on-DMET", files, percentiles)