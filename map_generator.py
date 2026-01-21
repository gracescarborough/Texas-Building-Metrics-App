import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
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

                area = inter.area
                if area <= 0:
                    continue

                btype = bldg.get(btype_column, DEFAULT_TYPE)
                if pd.isna(btype) or btype not in EEI_BY_TYPE:
                    btype = DEFAULT_TYPE

                total_area += area
                total_ee += area * EEI_BY_TYPE[btype]
                total_ec += area * ECI_BY_TYPE[btype]
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

            if count > 50 and sample_id < 5:
                print(f"\n=== Sample {sample_id} ===")
                print(f"Coordinates: ({pt.x:.2f}, {pt.y:.2f})")
                print(f"Building count: {count}")
                print(f"Total building area: {total_area:.2f} m²")
                print(f"Coverage fraction: {total_area / M2_PER_MI2:.4f} ({(total_area / M2_PER_MI2)*100:.2f}%)")
                print(f"Buffer area: {M2_PER_MI2:.2f} m²")

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

x_coords = np.arange(tx_minx, tx_maxx+GRID_RES_M, GRID_RES_M)
y_coords = np.arange(tx_miny, tx_maxy+GRID_RES_M, GRID_RES_M)

grid_points = [Point(x,y) for x in x_coords for y in y_coords if texas.contains(Point(x,y)).any()]
grid = gpd.GeoDataFrame(geometry=grid_points, crs=TEXAS_CRS)

valid = centroids_gdf['coverage_fraction'] > 0

centroid_coords = np.vstack([
    centroids_gdf.loc[valid].geometry.x.values,
    centroids_gdf.loc[valid].geometry.y.values
]).T

centroid_values = centroids_gdf.loc[valid, 'coverage_fraction'].values

grid_coords = np.vstack([
    grid.geometry.x.values,
    grid.geometry.y.values
]).T

tree = cKDTree(centroid_coords)

k = min(16, len(centroid_coords))
distances, indices = tree.query(grid_coords, k=k)

p = 2
weights = 1 / (distances**p + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

grid['density'] = (centroid_values[indices] * weights).sum(axis=1)

bins = [0, 0.0001, 0.0008, 0.005, 0.015, 1]
labels = ["Unoccupied","Rural","Slightly Rural","Suburban","Urban"]
grid['class'] = pd.cut(grid['density'], bins=bins, labels=labels, include_lowest=True)
class_to_id = dict(zip(labels, range(1,6)))
grid['class_id'] = grid['class'].map(class_to_id)
centroids_gdf['class'] = pd.cut(centroids_gdf['coverage_fraction'], bins=bins, labels=labels, include_lowest=True)
centroids_gdf['class_id'] = centroids_gdf['class'].map(class_to_id).astype("Int64")

grid.to_file("tx_grid_classified.shp")
centroids_gdf.to_file("sample_centroids_with_density.shp")

fig, ax = plt.subplots(figsize=(12,12))
colors = {1:"#FFEB97",2:"#FFC166",3:"#FF9C3C",4:"#F84E00",5:"#E50000"}
texas.boundary.plot(ax=ax, linewidth=0.5, color="black")
for cls, col in colors.items():
    grid[grid["class_id"]==cls].plot(ax=ax, color=col, alpha=0.8)
centroids_gdf.plot(ax=ax, color="black", markersize=2, alpha=0.3)
ax.set_title("Texas Building Density", fontsize=16)
ax.set_axis_off()
plt.tight_layout(); plt.savefig("texas_density_map.png", dpi=300); plt.show()