import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import random
import os
import glob
from tqdm import tqdm
import gc
import fiona
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

OSWD = "/home/grace/FloodFiles"
GDB_PATTERN = "FPR*Bldgs_SVI_Pop_Feb2025.gdb"
COUNTY_SHP = "/home/grace/US_COUNTY_SHPFILE/US_county_cont.shp"
SAMPLE_COORD_CSV = "sample_coordinates.csv"

GRID_RES_M = 750
TEXAS_CRS = "EPSG:3083"
RANDOM_SEED = 13545
RADIUS_1MI_M = np.sqrt((1609.344**2) / np.pi)
M2_PER_MI2 = 1609.344**2

EEI_BY_TYPE = {"Residential":5000,"Commercial":6000,"Industrial":6500,"Institutional":7500,"Other":5000}
ECI_BY_TYPE = {"Residential":200,"Commercial":400,"Industrial":600,"Institutional":500,"Other":300}
DEFAULT_TYPE = "Other"
btype_column = "Simp_type"
num_floors_column = "num_floors"

def estimate_floors(btype, footprint_area_m2):
    """
    Estimate number of floors based on building type and footprint area.
    Returns an integer >= 1.
    """

    if btype == "Residential":
        floors = 1.1145 * np.exp(0.000257 * footprint_area_m2)

    elif btype == "Commercial":
        floors = 1.1808 * np.exp(0.00004 * footprint_area_m2)

    elif btype == "Industrial":
        floors = 1

    elif btype == "Institutional":
        floors = 1.1808 * np.exp(0.00004 * footprint_area_m2)

    else:  # Other
        floors = 1

    floors = max(1, floors)      
    floors = min(floors, 70)

    floors = int(round(floors))

    return floors

os.chdir(OSWD)
random.seed(RANDOM_SEED)

all_gdb_files = sorted(glob.glob(os.path.join(OSWD, GDB_PATTERN)))
counties = gpd.read_file(COUNTY_SHP)
texas = counties[counties["STATE_NAME"]=="Texas"].to_crs(TEXAS_CRS)
tx_minx, tx_miny, tx_maxx, tx_maxy = texas.total_bounds

sample_coords_df = pd.read_csv(SAMPLE_COORD_CSV)
sample_points = [Point(xy) for xy in zip(sample_coords_df['x_m'], sample_coords_df['y_m'])]
sample_points_gdf = gpd.GeoDataFrame({
    'geometry': sample_points,
    'x': sample_coords_df['x_m'],
    'y': sample_coords_df['y_m']
}, crs=TEXAS_CRS)

results = []
sample_id = 0

for gdb_file in tqdm(all_gdb_files, desc="Processing GDB files"):
    try:
        layers = fiona.listlayers(gdb_file)
        if not layers:
            continue

        buildings = gpd.read_file(gdb_file, layer=layers[0]).to_crs(TEXAS_CRS)
        minx, miny, maxx, maxy = buildings.total_bounds
        pts_in_gdb = sample_points_gdf.cx[minx:maxx, miny:maxy]

        if len(pts_in_gdb) == 0:
            del buildings
            continue

        for idx, row in pts_in_gdb.iterrows():
            pt = row.geometry
            buffer = pt.buffer(RADIUS_1MI_M)
            bxmin, bymin, bxmax, bymax = buffer.bounds
            subset_bldg = buildings.cx[bxmin:bxmax, bymin:bymax]

            total_area = 0.0
            total_ee = 0.0
            total_ec = 0.0
            count = 0

            for _, bldg in subset_bldg.iterrows():
                geom = bldg.geometry
                if geom is None or geom.is_empty:
                    continue

                inter = geom.intersection(buffer)
                if inter.is_empty:
                    continue

                footprint_area = inter.area
                if footprint_area <= 0:
                    continue

                btype = bldg.get(btype_column, DEFAULT_TYPE)
                if pd.isna(btype) or btype not in EEI_BY_TYPE:
                    btype = DEFAULT_TYPE

                num_floors = bldg.get(num_floors_column)
                if pd.isna(num_floors) or num_floors <= 0:
                    num_floors = estimate_floors(btype, footprint_area)

                total_bldg_area = footprint_area * num_floors

                total_area += total_bldg_area
                total_ee += total_bldg_area * EEI_BY_TYPE[btype]
                total_ec += total_bldg_area * ECI_BY_TYPE[btype]
                count += 1

            results.append({
                "sample_id": sample_id,
                "geometry": pt,
                "building_count": count,
                "building_area_m2": total_area,
                "building_area_mi2": total_area / M2_PER_MI2,
                "coverage_fraction": total_area / M2_PER_MI2,
                "total_ee_mj": total_ee,
                "total_ec_kgco2e": total_ec,
                "avg_eei_mj_m2": total_ee / total_area if total_area > 0 else 0.0,
                "avg_eci_kgco2e_m2": total_ec / total_area if total_area > 0 else 0.0
            })

            sample_id += 1

        del buildings
        gc.collect()

    except Exception as e:
        print(f"ERROR processing {os.path.basename(gdb_file)}: {e}")
        continue

centroids_gdf = gpd.GeoDataFrame(results, crs=TEXAS_CRS)
centroids_gdf.to_file("sample_centroids_with_stats.shp")
centroids_gdf[[
    "sample_id", "building_count", "building_area_m2", 
    "coverage_fraction", "total_ee_mj", "total_ec_kgco2e",
    "avg_eei_mj_m2", "avg_eci_kgco2e_m2"
]].to_csv("sample_centroids_with_stats.csv", index=False)
