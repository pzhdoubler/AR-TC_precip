import math
import datetime

# takes filename of precip file and returns datetime, assumes format ..\MSWEP_daily\{YEAR}\{YEAR}{DAY}.nc
def parse_precip_filename(f):
    return datetime.datetime.strptime(f.split("/")[-1],"%Y%j.nc")

# takes filename of freq or avg-intensity filename and returns the equivalent time value
def parse_output_filename(f):
    # year_season_misc.csv
    tokens = f.split("_")

    if tokens[0].isdigit():
        year = int(tokens[0])
    else:
        year = 0
    if len(tokens) == 1:
        return year

    season_labels = ["DJF", "MAM", "JJA", "SON"]
    season = season_labels.index(tokens[1])
    return year + (season * 0.25) + 2/12

# inverse of parse_output_filename
def time_val_to_label(t):
    year = math.floor(t)
    season_labels = ["DJF", "MAM", "JJA", "SON"]

    season_index = math.floor((t - year) * 4)

    return f"{str(year)}_{season_labels[season_index]}"