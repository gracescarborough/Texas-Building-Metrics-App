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

def estimate_floors(btype, footprint_area_m2, is_urban=False):
    """
    Estimate floors based on building type, footprint, and urban context.
    is_urban should be True for high-density urban cores where tall buildings exist
    """
    
    if btype == "Residential":
        if is_urban:
            if footprint_area_m2 < 600:
                floors = 1.164 + 0.0002*footprint_area_m2
            else:
                floors = -2.196 + 0.0051*footprint_area_m2
        else:
            if footprint_area_m2 < 700:
                floors = 1 + 0.001*footprint_area_m2
            else:
                floors = 3.8679 + 0.0005*footprint_area_m2
                
    elif btype == "Commercial":
        if is_urban:
            if footprint_area_m2 < 2000:
                floors = 1.1762 + 0.0001*footprint_area_m2
            else:
                floors = 3.8608 + 0.0002*footprint_area_m2
        else:
            if footprint_area_m2 < 3000:
                floors = 0.9721 + 0.0006*footprint_area_m2
            else:
                floors = 3.0796 + 0.00003*footprint_area_m2
                
    elif btype == "Industrial":
        floors = 1 
        
    elif btype == "Institutional":
        if is_urban:
            if footprint_area_m2 < 2000:
                floors = 1.1762 + 0.0001*footprint_area_m2
            else:
                floors = 3.8608 + 0.0002*footprint_area_m2
        else:
            if footprint_area_m2 < 3000:
                floors = 0.9721 + 0.0006*footprint_area_m2
            else:
                floors = 3.0796 + 0.00003*footprint_area_m2
    else:
        floors = 1
    
    floors = max(1, min(floors, 70))
    return int(round(floors))

os.chdir(OSWD)
random.seed(RANDOM_SEED)

sample_coords_df = pd.read_csv(SAMPLE_COORD_CSV)
sample_points = [Point(xy) for xy in zip(sample_coords_df['x_m'], sample_coords_df['y_m'])]
sample_points_gdf = gpd.GeoDataFrame({
    'geometry': sample_points,
    'x': sample_coords_df['x_m'],
    'y': sample_coords_df['y_m']
}, crs=TEXAS_CRS)

print(f"Loaded {len(sample_points_gdf)} sample points")

counties = gpd.read_file(COUNTY_SHP)
texas = counties[counties["STATE_NAME"]=="Texas"].to_crs(TEXAS_CRS)
tx_minx, tx_miny, tx_maxx, tx_maxy = texas.total_bounds

all_gdb_files = sorted(glob.glob(os.path.join(OSWD, GDB_PATTERN)))
results = []
sample_id = 0

for gdb_file in tqdm(all_gdb_files, desc="Processing GDB files"):
    try:
        layers = fiona.listlayers(gdb_file)
        if not layers:
            print(f"No layers in {gdb_file}")
            continue

        buildings = gpd.read_file(gdb_file, layer=layers[0]).to_crs(TEXAS_CRS)
        minx, miny, maxx, maxy = buildings.total_bounds
        pts_in_gdb = sample_points_gdf.cx[minx:maxx, miny:maxy]

        print(f"{os.path.basename(gdb_file)}: {len(buildings)} buildings, {len(pts_in_gdb)} points in extent")

        if len(pts_in_gdb) == 0:
            del buildings
            continue

        for idx, row in pts_in_gdb.iterrows():
            pt = row.geometry
            buffer = pt.buffer(RADIUS_1MI_M)
            bxmin, bymin, bxmax, bymax = buffer.bounds
            subset_bldg = buildings.cx[bxmin:bxmax, bymin:bymax]

            if len(subset_bldg) == 0:
                continue

            total_area = 0.0
            total_footprint = 0.0
            total_floors_weighted = 0.0
            total_ee = 0.0
            total_ec = 0.0
            count = 0

            temp_footprint = 0.0 ##############
            for _, bldg in subset_bldg.iterrows():
                geom = bldg.geometry
                if geom is None or geom.is_empty:
                    continue
                inter = geom.intersection(buffer)
                if not inter.is_empty and inter.area > 0:
                    temp_footprint += inter.area
            
            current_footprint_density = temp_footprint / M2_PER_MI2
            is_urban = (current_footprint_density > 0.012) #########

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
                    num_floors = estimate_floors(btype, footprint_area, is_urban)

                total_bldg_area = footprint_area * num_floors
                total_area += total_bldg_area
                total_footprint += footprint_area
                total_floors_weighted += footprint_area * num_floors
                total_ee += total_bldg_area * EEI_BY_TYPE[btype]
                total_ec += total_bldg_area * ECI_BY_TYPE[btype]
                count += 1

            avg_floors = (total_floors_weighted / total_footprint) if total_footprint > 0 else 0.0

            # Right before results.append({...})
            if abs(total_footprint - temp_footprint) > 0.01:
                print(f"WARNING Sample {sample_id}: Footprint mismatch! temp={temp_footprint:.2f}, total={total_footprint:.2f}")

            # Also check for impossible densities
            if total_footprint / M2_PER_MI2 > 1.0:
                print(f"WARNING Sample {sample_id}: Impossible footprint density = {total_footprint / M2_PER_MI2:.4f}")

            results.append({
                "sample_id": sample_id,
                "geometry": pt,
                "building_count": count,
                "total_floor_area_m2": total_area,
                "total_floor_density": total_area / M2_PER_MI2,
                "footprint_area_m2": total_footprint,
                "footprint_density": total_footprint / M2_PER_MI2,
                "avg_floors": avg_floors,
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

if len(results) == 0:
    print("WARNING: No results were generated. Creating empty GeoDataFrame with correct columns.")
    results = [{
        "sample_id": 0,
        "geometry": Point(0, 0),
        "building_count": 0,
        "total_floor_area_m2": 0.0,
        "total_floor_density": 0.0,
        "footprint_area_m2": 0.0,
        "footprint_density": 0.0,
        "avg_floors": 0.0,
        "total_ee_mj": 0.0,
        "total_ec_kgco2e": 0.0,
        "avg_eei_mj_m2": 0.0,
        "avg_eci_kgco2e_m2": 0.0
    }]

centroids_gdf = gpd.GeoDataFrame(results, crs=TEXAS_CRS)
centroids_gdf = centroids_gdf.rename(columns={
    "footprint_density": "fp_dens",
    "total_floor_density": "flr_dens",
    "total_floor_area_m2": "flr_area",
    "footprint_area_m2": "fp_area"
})

cols_to_keep = [
    "sample_id",
    "building_count",
    "flr_area",
    "flr_dens",
    "fp_area",
    "fp_dens",
    "avg_floors",
    "total_ee_mj",
    "total_ec_kgco2e",
    "avg_eei_mj_m2",
    "avg_eci_kgco2e_m2",
    "geometry"
]

centroids_gdf = centroids_gdf[cols_to_keep]

print("WRITING CENTROIDS TO:", os.getcwd())
print("COLUMNS BEING WRITTEN:")
print(centroids_gdf.columns)

centroids_gdf.to_file("sample_centroids_with_stats.shp")
centroids_gdf.to_csv("sample_centroids_with_stats.csv", index=False)

print(f"Saved {len(centroids_gdf)} sample centroids with stats")
