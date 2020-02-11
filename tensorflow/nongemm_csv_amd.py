import argparse
import pandas as pd
import numpy as np
import os
import subprocess
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def save_fig(fig, fig_name, dir_name=None):
    if dir_name:
        dir_name = dir_name
    else:
        dir_name = "."

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # double fig dpi to prevent loss of info
    plt.savefig(os.path.join(dir_name, fig_name.replace(
        '/', '--') + ".png"), dpi=2*fig.dpi)


def convert_to_microseconds(time):
    time_stripped = time.strip()

    if "ms" in time_stripped:
        return float(time_stripped[:-2])*1000
    elif "ns" in time_stripped:
        return float(time_stripped[:-2])/1000
    elif "us" in time_stripped:
        return float(time_stripped[:-2])
    elif "s" in time_stripped:
        return float(time_stripped[:-1])*1000000
    else:
        return float(time_stripped)


def remove_percent(percent):
    percent_stripped = percent.strip()
    if "%" in percent_stripped:
        return float(percent_stripped[:-1])
    else:
        return float(percent)


def nvidia_tag(name):
    if "gemm" in name.lower():
        return "GEMM"
    elif "gap" in name:
        return "Gap"
    else:
        return "Other"


def amd_tag(name):
    if "Cijk" in name:
        return "GEMM"
    elif "gap" in name:
        return "Gap"
    else:
        return "Other"


def remove_whitespace(string):
    return string.replace(" ", "")


def read_amd_histogram_file(amd_histogram_file_path):
    amd_histogram = open(amd_histogram_file_path)
    amd_histogram.seek(0)

    new_table = False
    table_count = 0
    amd_tables = []
    for line in amd_histogram:
        line_stripped = line.strip()
        if line_stripped.startswith("Resource="):
            new_table = True
            amd_tables.append([])
            table_count += 1

        if new_table and line_stripped:
            amd_tables[table_count-1].append(line_stripped + "\n")

    amd_csv_file = StringIO("".join(amd_tables[0][2:]))
    amd_csv_file.seek(0)
    amd_df_unprocessed = pd.read_table(
        amd_csv_file, index_col=False, header=None, names=["raw"])
    amd_df_unprocessed = amd_df_unprocessed["raw"].str.split(
        "\s+", n=6, expand=True)
    amd_df_unprocessed.columns = amd_tables[0][1].split()

    amd_df = pd.DataFrame()

    amd_df["Total(%)"] = amd_df_unprocessed["Total(%)"].map(remove_percent)
    amd_df["Time(us)"] = amd_df_unprocessed["Time(us)"].map(float)
    amd_df["Calls"] = amd_df_unprocessed["Calls"].map(float)
    amd_df["Avg(us)"] = amd_df_unprocessed["Avg(us)"].map(float)
    amd_df["Min(us)"] = amd_df_unprocessed["Min(us)"].map(float)
    amd_df["Max(us)"] = amd_df_unprocessed["Max(us)"].map(float)
#     amd_df["Mangled Name"]=amd_df_unprocessed["Name"]
    amd_df["Name"] = amd_df_unprocessed["Name"].map(str.strip)
    amd_df["Tag"] = amd_df_unprocessed["Name"].map(amd_tag)

    return amd_df

def nongemm_csv(amd_path):
    amd_path = os.path.expanduser(amd_path)
    
    amd_df = read_amd_histogram_file(amd_path)
   
    amd_nongemm = amd_df.loc[amd_df["Tag"] == "Other"][['Name', 'Time(us)']]

    amd_nongemm.to_csv("nongemm_amd"+".csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amd_path")

    args = parser.parse_args()
    nongemm_csv(args.amd_path)
