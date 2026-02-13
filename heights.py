import fiona
import random
import os
import gc
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, shape

FLOOD_DIR = "/home/grace/FloodFiles"
GRID_PATH = "/home/grace/FloodFiles/tx_grid_classified.shp"
TARGET_N = 6000
TOP_K = 10
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
MIN_FLOORS = 1
MAX_FLOORS = None

URBAN_THRESHOLD = 0.12

BUILDING_CATEGORIES = {
    "Residential": "residential",
    "Commercial": "commercial",
    "Industrial": "industrial",
    "Agricultural": "agricultural",
    "Public": "public"
}

samples = {"residential": [], "commercial": [], "industrial": [], "agricultural": [], "public": []}
seen = {"residential": 0, "commercial": 0, "industrial": 0, "agricultural": 0, "public": 0}
tallest = {"residential": [], "commercial": [], "industrial": [], "agricultural": [], "public": []}

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

                    floors = props.get("Est_floors")
                    if floors is None:
                        continue
                    try:
                        floors = float(floors)
                    except (ValueError, TypeError):
                        continue
                    if MIN_FLOORS is not None and floors < MIN_FLOORS:
                        continue
                    if MAX_FLOORS is not None and floors > MAX_FLOORS:
                        continue

                    new_props = {
                        "Bldg_ID": props.get("Bldg_ID"),
                        "Simp_type": props.get("Simp_type"),
                        "Est_floors": props.get("Est_floors"),
                        "Shape_Area": props.get("Shape_Area"),
                        "footprint_m2": props.get("Shape_Area"),
                        "height_proxy": props.get("Est_floors"),
                        "source": f"{os.path.basename(gdb_path)}::{layer}",
                    }
                    
                    new_feat = {
                        "geometry": feat["geometry"],
                        "properties": new_props
                    }

                    samples[btype], seen[btype] = reservoir_update(
                        samples[btype], new_feat, seen[btype]
                    )

                    tallest[btype] = update_top_k(
                        tallest[btype], new_feat, TOP_K
                    )
                    
                    if (i + 1) % 10000 == 0:
                        print(f"   Processed {i + 1}/{total} buildings...")

        except Exception as e:
            print(f"⚠️ Failed {gdb_path}::{layer}: {e}")

        gc.collect()

print("\n" + "="*50)
print("ADDING DENSITY DATA TO SAMPLED BUILDINGS")
print("="*50)

print("Loading density grid...")
density_grid = gpd.read_file(GRID_PATH)
print(f"   Loaded {len(density_grid)} grid cells")
print(f"   Grid CRS: {density_grid.crs}")

for btype in ["residential", "commercial", "industrial", "agricultural", "public"]:
    if not samples[btype]:
        print(f"⚠️ No {btype} buildings sampled")
        continue

    print(f"\nProcessing {btype} buildings...")
    
    samples[btype].extend(tallest[btype])

    unique = {}
    for f in samples[btype]:
        bid = f["properties"].get("Bldg_ID")
        unique[bid] = f
    features = list(unique.values())

    print(f"   {len(features)} unique buildings after deduplication")

    gdf = gpd.GeoDataFrame.from_features(features)

    if gdf.crs is None:
        gdf.set_crs(epsg=3857, inplace=True)
    
    print(f"   Buildings CRS: {gdf.crs}")
    
    gdf = gdf.to_crs(density_grid.crs)
    
    print(f"   Performing spatial join with density grid...")
    gdf_with_density = gpd.sjoin(
        gdf, 
        density_grid[['geometry', 'fp_dens']], 
        how='left', 
        predicate='within'
    )
    
    gdf_with_density['local_fp_density'] = gdf_with_density['fp_dens'].fillna(0)
    gdf_with_density['is_urban'] = gdf_with_density['local_fp_density'] >= URBAN_THRESHOLD
    
    gdf_with_density = gdf_with_density.drop(columns=['fp_dens', 'index_right'], errors='ignore')

    gdf_with_density = gdf_with_density.to_crs(epsg=4326)

    gdf_with_density["lon"] = gdf_with_density.geometry.centroid.x
    gdf_with_density["lat"] = gdf_with_density.geometry.centroid.y

    gdf_with_density.drop(columns="geometry", inplace=True)

    out_csv = f"TX_{btype}_heights2.csv"
    gdf_with_density.to_csv(out_csv, index=False)

    n_urban = gdf_with_density["is_urban"].sum()
    n_total = len(gdf_with_density)
    print(f"✅ Exported {n_total} {btype} buildings → {out_csv}")
    print(f"   Urban: {n_urban} ({100*n_urban/n_total:.1f}%), Rural: {n_total-n_urban} ({100*(n_total-n_urban)/n_total:.1f}%)")

print("\n📁 Done.")