import fiona
import random
import os
import gc
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape

# -----------------------------
# SETTINGS
# -----------------------------
FLOOD_DIR = "/home/grace/FloodFiles"
GRID_PATH = "/home/grace/FloodFiles/tx_grid_classified.shp"
TARGET_N = 500
TOP_K = 10
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Urban threshold: 11.5% footprint density
URBAN_THRESHOLD = 0.115

BUILDING_CATEGORIES = {
    "Residential": "residential",
    "Commercial": "commercial",
}

samples = {"residential": [], "commercial": []}
seen = {"residential": 0, "commercial": 0}
tallest = {"residential": [], "commercial": []}

# -----------------------------
# HELPERS
# -----------------------------
def classify_building(simp_type):
    return BUILDING_CATEGORIES.get(simp_type)

def reservoir_update(sample_list, feature, seen_count):
    seen_count += 1
    if len(sample_list) < TARGET_N:
        sample_list.append(feature)
    else:
        j = random.randint(0, seen_count - 1)
        if j < TARGET_N:
            sample_list[j] = feature
    return sample_list, seen_count

def update_top_k(top_list, feature, k):
    floors = feature["properties"].get("Est_floors")
    if floors is None:
        return top_list

    top_list.append(feature)
    top_list.sort(
        key=lambda f: f["properties"].get("Est_floors") or 0,
        reverse=True
    )
    return top_list[:k]

# -----------------------------
# MAIN LOOP - SAMPLE BUILDINGS
# -----------------------------
gdbs = [os.path.join(FLOOD_DIR, f) for f in os.listdir(FLOOD_DIR) if f.endswith(".gdb")]
print(f"Found {len(gdbs)} geodatabases\n")

for gdb_path in gdbs:
    try:
        layers = fiona.listlayers(gdb_path)
    except Exception as e:
        print(f"⚠️ Cannot list layers in {gdb_path}: {e}")
        continue

    for layer in layers:
        if "bldg" not in layer.lower():
            continue

        print(f"📂 Processing {os.path.basename(gdb_path)} :: {layer}")

        try:
            with fiona.open(gdb_path, layer=layer) as src:
                total = len(src)
                print(f"   Sampling from {total} features...")
                
                for i, feat in enumerate(src):
                    props = feat["properties"]
                    btype = classify_building(props.get("Simp_type"))
                    if not btype:
                        continue

                    # Create new properties dict
                    new_props = {
                        "Bldg_ID": props.get("Bldg_ID"),
                        "Simp_type": props.get("Simp_type"),
                        "Est_floors": props.get("Est_floors"),
                        "Shape_Area": props.get("Shape_Area"),
                        "footprint_m2": props.get("Shape_Area"),
                        "height_proxy": props.get("Est_floors"),
                        "source": f"{os.path.basename(gdb_path)}::{layer}",
                    }
                    
                    # Create feature dict
                    new_feat = {
                        "geometry": feat["geometry"],
                        "properties": new_props
                    }

                    # Random sample
                    samples[btype], seen[btype] = reservoir_update(
                        samples[btype], new_feat, seen[btype]
                    )

                    # Tallest buildings
                    tallest[btype] = update_top_k(
                        tallest[btype], new_feat, TOP_K
                    )
                    
                    # Progress indicator
                    if (i + 1) % 10000 == 0:
                        print(f"   Processed {i + 1}/{total} buildings...")

        except Exception as e:
            print(f"⚠️ Failed {gdb_path}::{layer}: {e}")

        gc.collect()

# -----------------------------
# LOAD DENSITY GRID & ADD DENSITY TO SAMPLES
# -----------------------------
print("\n" + "="*50)
print("ADDING DENSITY DATA TO SAMPLED BUILDINGS")
print("="*50)

print("Loading density grid...")
density_grid = gpd.read_file(GRID_PATH)
print(f"   Loaded {len(density_grid)} grid cells")
print(f"   Grid CRS: {density_grid.crs}")

# -----------------------------
# EXPORT
# -----------------------------
for btype in ["residential", "commercial"]:
    if not samples[btype]:
        print(f"⚠️ No {btype} buildings sampled")
        continue

    print(f"\nProcessing {btype} buildings...")
    
    # Merge tallest into sample
    samples[btype].extend(tallest[btype])

    # De-duplicate by Bldg_ID
    unique = {}
    for f in samples[btype]:
        bid = f["properties"].get("Bldg_ID")
        unique[bid] = f
    features = list(unique.values())

    print(f"   {len(features)} unique buildings after deduplication")

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(features)

    # Set CRS (TWDB data should be Web Mercator)
    if gdf.crs is None:
        gdf.set_crs(epsg=3857, inplace=True)
    
    print(f"   Buildings CRS: {gdf.crs}")
    
    # Convert to same CRS as density grid
    gdf = gdf.to_crs(density_grid.crs)
    
    # Spatial join to get density
    print(f"   Performing spatial join with density grid...")
    gdf_with_density = gpd.sjoin(
        gdf, 
        density_grid[['geometry', 'fp_dens']], 
        how='left', 
        predicate='within'
    )
    
    # Add density and urban flag
    gdf_with_density['local_fp_density'] = gdf_with_density['fp_dens'].fillna(0)
    gdf_with_density['is_urban'] = gdf_with_density['local_fp_density'] >= URBAN_THRESHOLD
    
    # Drop the extra columns from spatial join
    gdf_with_density = gdf_with_density.drop(columns=['fp_dens', 'index_right'], errors='ignore')

    # Convert to WGS84 for final output
    gdf_with_density = gdf_with_density.to_crs(epsg=4326)

    # Centroid lat/lon
    gdf_with_density["lon"] = gdf_with_density.geometry.centroid.x
    gdf_with_density["lat"] = gdf_with_density.geometry.centroid.y

    gdf_with_density.drop(columns="geometry", inplace=True)

    out_csv = f"TX_{btype}_heights.csv"
    gdf_with_density.to_csv(out_csv, index=False)

    # Print urban/rural stats
    n_urban = gdf_with_density["is_urban"].sum()
    n_total = len(gdf_with_density)
    print(f"✅ Exported {n_total} {btype} buildings → {out_csv}")
    print(f"   Urban: {n_urban} ({100*n_urban/n_total:.1f}%), Rural: {n_total-n_urban} ({100*(n_total-n_urban)/n_total:.1f}%)")

print("\n📁 Done.")