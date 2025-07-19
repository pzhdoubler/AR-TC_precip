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

# takes map data and projects it into the MSWEP domain
def to_mswep():
    pass

# takes map data and projects it into the CWRF domain
def to_cwrf():
    pass
    

def count_percentile_freqs(FREQ_FOLDER, INTENSITY_FOLDER, all_files, setup):
    try:
        # intake setup data
        percentile_grids, percentile_title, percentiles, lon_labels, lat_labels = setup
    except Exception as e:
        print("Error with setup. Qutting...")
        print(e)
        return

    # set data recording
    frequency_count = np.zeros((len(lat_labels), len(lon_labels), len(percentiles)), dtype=np.int64)
    intensity_sum = np.zeros((len(lat_labels), len(lon_labels), len(percentiles)), dtype=np.float64)
    lon = sorted([round((l + 179.95) * 10) for l in [lon_labels[0], lon_labels[-1] + 0.1]])
    lat = sorted([round((-1*l + 89.95) * 10) for l in [lat_labels[0], lat_labels[-1] - 0.1]])   # lat must be flipped

    # set up seasons and file order
    season = {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
    season_labels = ["DJF", "MAM", "JJA", "SON"]
    all_files.sort(key=parse_precip_filename)
    cur_season = season[parse_precip_filename(all_files[0]).month]

    init = True
    for year in range(1979,2021):
        print(f"Counting frequencies for {year} ...")

        year_files = [f for f in all_files if str(year) in f]

        for f, file in enumerate(year_files):
            try:
                dt = parse_precip_filename(file)

                # calc runtime
                start = time.time()

                # save current frequencies if switched to a new season
                if season[dt.month] != cur_season:
                    for p, percentile in enumerate(percentiles):
                        percentile_str = percentile_title + str(percentile).replace(".","-")
                        # frequency path
                        f_name = f"{year}_{season_labels[cur_season]}_freq.csv"
                        f_path = f"./{FREQ_FOLDER}/{percentile_str}"
                        if not os.path.exists(f_path):
                            os.makedirs(f_path)
                        try:
                            df = pd.DataFrame(frequency_count[:,:,p], index=lat_labels, columns=lon_labels)
                            df.index.name="LAT/LON"
                            df.to_csv(f"{f_path}/{f_name}")
                        except Exception as e:
                            print(f"Error saving {f_name}")
                            print(e)

                        # intensity path
                        
                        i_name = f"{year}_{season_labels[cur_season]}_avg-intensity.csv"
                        i_path = f"./{INTENSITY_FOLDER}/{percentile_str}"
                        if not os.path.exists(i_path):
                            os.makedirs(i_path)
                        try:
                            frequency_count[:,:,p] += frequency_count[:,:,p] == 0    # remove 0 valued counts to avoid div by 0
                            df = pd.DataFrame(intensity_sum[:,:,p] / frequency_count[:,:,p], index=lat_labels, columns=lon_labels)
                            df.index.name="LAT/LON"
                            df.to_csv(f"{i_path}/{i_name}")
                        except Exception as e:
                            print(f"Error saving {i_name}")
                            print(e)

                    # reset data
                    frequency_count = np.zeros((len(lat_labels), len(lon_labels), len(percentiles)), dtype=np.int64)
                    intensity_sum = np.zeros((len(lat_labels), len(lon_labels), len(percentiles)), dtype=np.float64)
                    cur_season = season[dt.month]

                # print progress
                if f % (len(year_files) // 100) == 0:
                    print((f/len(year_files))*100, "% complete", end="\r")

                data = netCDF4.Dataset(year_files[f],mode='r')

                precip = data.variables["precipitation"][:][0][lat[0]:lat[1],lon[0]:lon[1]]
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


def setup_freq_count():
    pass

# returns percentile grids, percentile array, lon_labels, lat_labels
def setup_freq_count(PERCENTILE_FOLDER, percentile_title, percentiles):
    try:
        # get percentile files
        files = [f for f in os.listdir(PERCENTILE_FOLDER) if f.startswith(percentile_title)]
        # get expected shape of  percentile files
        df = pd.read_csv(f"./percentiles/{files[0]}", header=0, index_col=0)
        lon_labels = df.columns.values.astype(np.float64)
        lat_labels = df.index.values.astype(np.float64)
        # (lat, lon, percentile, season)
        percentile_grids = np.zeros((len(lat_labels), len(lon_labels), len(percentiles), 4))

        season_labels = ["DJF", "MAM", "JJA", "SON"]
        for p, percentile in enumerate(percentiles):
            p_label = str(percentile).replace(".","-")
            percentile_files = [f for f in files if p_label in f]
            for s, season in enumerate(season_labels):
                season_file = next(f for f in percentile_files if season in f)
                df = pd.read_csv(f"./percentiles/{season_file}", header=0, index_col=0)
                if np.allclose(lon_labels, df.columns.values.astype(np.float64)) and np.allclose(lat_labels, df.index.values.astype(np.float64)):
                    percentile_grids[:,:,p,s] = df.values.astype(np.float64)
                    print(f"Read in {season_file} ...")
                else:
                    print(f"Problem with lon, lat comparison in {season_file}. Quitting ...")
                    return
            
        return percentile_grids, percentile_title, percentiles, lon_labels, lat_labels

    except Exception as e:
        print("Problem in setup. Quitting ...")
        print(e)
        return