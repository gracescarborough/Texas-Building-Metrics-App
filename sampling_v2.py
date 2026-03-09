import geopandas as gpd
import pandas as pd
import numpy as np
import os
import glob
import fiona
import gc
from tqdm import tqdm
from shapely.geometry import Point

for f in glob.glob("/home/grace/FloodFiles/BuildingParquets/*.parquet"):
    os.remove(f)
print("Deleted all parquets")

OSWD = "/home/grace/FloodFiles"
GDB_PATTERN = "FPR*Bldgs_SVI_Pop_Feb2025.gdb"
HEIGHTS_DIR = "/home/grace/FloodFiles"
OUTPUT_DIR = "/home/grace/FloodFiles/BuildingParquets"
MERGED_OUTPUT = "/home/grace/FloodFiles/all_buildings.parquet"

TEXAS_CRS = "EPSG:3083"
btype_column = "Simp_type"
num_floors_column = "num_floors"
DEFAULT_TYPE = "Other"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(OSWD)

height_data = {}
for btype, fname in [
    ("Residential",   "TX_residential_heights2.csv"),
    ("Commercial",    "TX_commercial_heights2.csv"),
    ("Institutional", "TX_public_heights2.csv"),
]:
    path = os.path.join(HEIGHTS_DIR, fname)
    df = pd.read_csv(path)[["footprint_m2", "Est_floors", "is_urban"]].dropna()
    df["Est_floors"] = df["Est_floors"].astype(int).clip(lower=1, upper=70)
    height_data[btype] = df

print("Loaded height distributions:")
for k, v in height_data.items():
    print(f"  {k}: {len(v)} buildings")

rng = np.random.default_rng(seed=13545)

def sample_floors(btype, footprint_area_m2, is_urban=False, n_bins=5):
    """Sample floor count from empirical distribution, used only when GDB num_floors is missing."""
    if btype not in height_data:
        return 1
    df = height_data[btype]
    pool = df[df["is_urban"] == is_urban]
    if len(pool) < 10:
        pool = df
    for n in [n_bins, 3, 1]:
        bins = pd.qcut(pool["footprint_m2"], q=n, duplicates="drop", retbins=True)[1]
        bin_idx = np.clip(np.searchsorted(bins, footprint_area_m2, side="right") - 1, 0, len(bins) - 2)
        lo, hi = bins[bin_idx], bins[bin_idx + 1]
        neighbors = pool[(pool["footprint_m2"] >= lo) & (pool["footprint_m2"] <= hi)]
        if len(neighbors) >= 5:
            break
    if len(neighbors) == 0:
        neighbors = pool
    return max(1, min(int(rng.choice(neighbors["Est_floors"].values)), 70))


all_gdb_files = sorted(glob.glob(os.path.join(OSWD, GDB_PATTERN)))
print(f"Found {len(all_gdb_files)} GDB files")

for gdb_file in tqdm(all_gdb_files, desc="Processing GDB files"):
    basename = os.path.basename(gdb_file).replace(".gdb", "")
    out_path = os.path.join(OUTPUT_DIR, f"{basename}.parquet")

    if os.path.exists(out_path):
        print(f"  Skipping {basename} (already processed)")
        continue

    try:
        layers = fiona.listlayers(gdb_file)
        if not layers:
            print(f"  No layers in {gdb_file}")
            continue

        buildings = gpd.read_file(gdb_file, layer=layers[0]).to_crs(TEXAS_CRS)
        print(f"  Raw btypes: {buildings[btype_column].unique()}")
        print(f"  {basename}: {len(buildings)} buildings")

        buildings["footprint_m2"] = buildings.geometry.area
        buildings["centroid_geom"] = buildings.geometry.centroid

        if btype_column in buildings.columns:
            type_mapping = {
                "Residential":      "Residential",
                "Commercial":       "Commercial",
                "Industrial":       "Industrial",
                "Public":           "Institutional",
                "Agricultural":     "Agricultural",
                "Vacant or Unknown": "Other",
            }
            buildings["btype"] = buildings[btype_column].map(type_mapping).fillna(DEFAULT_TYPE)
        else:
            buildings["btype"] = DEFAULT_TYPE

        if num_floors_column in buildings.columns:
            has_floors = buildings[num_floors_column].notna() & (buildings[num_floors_column] > 0)
            buildings["Est_floors"] = np.where(
                has_floors,
                buildings[num_floors_column].clip(lower=1, upper=70),
                buildings.apply(
                    lambda row: sample_floors(row["btype"], row["footprint_m2"], is_urban=False),
                    axis=1
                )
            )
        else:
            buildings["Est_floors"] = buildings.apply(
                lambda row: sample_floors(row["btype"], row["footprint_m2"], is_urban=False),
                axis=1
            )

        buildings["Est_floors"] = buildings["Est_floors"].astype(int)

        lean = gpd.GeoDataFrame(
            {
                "footprint_m2": buildings["footprint_m2"].values,
                "Est_floors":   buildings["Est_floors"].values,
                "btype":        buildings["btype"].values,
            },
            geometry=buildings["centroid_geom"].values,
            crs=TEXAS_CRS
        )

        lean.to_parquet(out_path)
        print(f"  Saved {len(lean)} buildings → {out_path}")

        del buildings, lean
        gc.collect()

    except Exception as e:
        print(f"  ERROR processing {basename}: {e}")
        continue

print("\nMerging all GDB parquets...")
parquet_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.parquet")))

chunks = []
for p in tqdm(parquet_files, desc="Loading parquets"):
    chunks.append(gpd.read_parquet(p))

all_buildings = pd.concat(chunks, ignore_index=True)
all_buildings = gpd.GeoDataFrame(all_buildings, crs=TEXAS_CRS)

all_buildings.to_parquet(MERGED_OUTPUT)
print(f"Merged {len(all_buildings)} buildings → {MERGED_OUTPUT}")