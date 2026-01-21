import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
import random
import os
import glob
from tqdm import tqdm
import fiona
import gc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
OSWD = "/home/grace/FloodFiles"
GDB_PATTERN = "FPR*Bldgs_SVI_Pop_Feb2025.gdb"
COUNTY_SHP = "/home/grace/US_COUNTY_SHPFILE/US_county_cont.shp"
N_SAMPLES = 10000
TEXAS_CRS = "EPSG:3083"
RANDOM_SEED = 13545
BUFFER_AREA_SQKM = 2.589988
BUFFER_RADIUS_M = np.sqrt(BUFFER_AREA_SQKM * 1e6 / np.pi)
MIN_POINT_DISTANCE = 2 * BUFFER_RADIUS_M
os.chdir(OSWD)
random.seed(RANDOM_SEED)

counties = gpd.read_file(COUNTY_SHP)
texas = counties[counties["STATE_NAME"] == "Texas"].to_crs(TEXAS_CRS)
tx_minx, tx_miny, tx_maxx, tx_maxy = texas.total_bounds

all_gdb_files = sorted(glob.glob(os.path.join(OSWD, GDB_PATTERN)))
flood_map_bounds = []

for gdb_file in tqdm(all_gdb_files, desc="Reading GDB boundaries"):
    try:
        layers = fiona.listlayers(gdb_file)
        if not layers:
            continue
        with fiona.open(gdb_file, layer=layers[0]) as src:
            bounds = src.bounds
            src_crs = src.crs
        bounds_geom = box(bounds[0], bounds[1], bounds[2], bounds[3])
        if src_crs is None:
            bounds_gdf = gpd.GeoDataFrame([{'geometry': bounds_geom}], crs=TEXAS_CRS)
        else:
            bounds_gdf = gpd.GeoDataFrame([{'geometry': bounds_geom}], crs=src_crs)
            bounds_gdf = bounds_gdf.to_crs(TEXAS_CRS)
        flood_map_bounds.append({
            'gdb_file': gdb_file,
            'geometry': bounds_gdf.geometry.iloc[0]
        })
        del bounds_gdf
        gc.collect()
    except Exception as e:
        print(f"Error with {os.path.basename(gdb_file)}: {e}")
        continue

if len(flood_map_bounds) == 0:
    raise RuntimeError("No flood map boundaries loaded.")

flood_maps_gdf = gpd.GeoDataFrame(flood_map_bounds, geometry='geometry', crs=TEXAS_CRS)

sample_points = []
sample_coords = []
tree = None

attempts = 0
max_attempts = N_SAMPLES * 5000

pbar = tqdm(total=N_SAMPLES, desc="Sampling points (1.13 mi spacing)")

while len(sample_points) < N_SAMPLES and attempts < max_attempts:
    attempts += 1

    x = random.uniform(tx_minx, tx_maxx)
    y = random.uniform(tx_miny, tx_maxy)
    pt = Point(x, y)

    if not texas.contains(pt).any():
        continue

    if not flood_maps_gdf.contains(pt).any():
        continue

    if tree is not None:
        dist, _ = tree.query([x, y], k=1)
        if dist < MIN_POINT_DISTANCE:
            continue

    sample_points.append(pt)
    sample_coords.append([x, y])
    tree = cKDTree(sample_coords)

    pbar.update(1)

pbar.close()

print(f"Generated {len(sample_points)} points in {attempts} attempts")

if len(sample_points) < N_SAMPLES:
    raise RuntimeError(
        f"Only generated {len(sample_points)} points. "
        "Texas floodplains cannot support 10,000 "
        "independent 1-sq-mile buffers."
    )

points_gdf = gpd.GeoDataFrame({
    'geometry': sample_points,
    'x_m': [p.x for p in sample_points],
    'y_m': [p.y for p in sample_points]
}, crs=TEXAS_CRS)

points_gdf_wgs84 = points_gdf.to_crs("EPSG:4326")

output_df = pd.DataFrame({
    'x_m': points_gdf['x_m'].values,
    'y_m': points_gdf['y_m'].values,
    'lon': points_gdf_wgs84.geometry.x.values,
    'lat': points_gdf_wgs84.geometry.y.values
})

output_df.to_csv("sample_coordinates.csv", index=False)
print("Saved sample coordinates with lat/lon to 'sample_coordinates.csv'")

plt.figure(figsize=(10,10))
plt.scatter(output_df['x_m'], output_df['y_m'], s=5, color='blue', alpha=0.5)
plt.title(f"{len(sample_points)} Sample Points in Texas Flood Zones")
plt.xlabel("X (meters, EPSG:3083)")
plt.ylabel("Y (meters, EPSG:3083)")
plt.axis('equal')
plt.tight_layout()
plt.savefig("sample_points.png", dpi=300, bbox_inches='tight')
plt.show()
