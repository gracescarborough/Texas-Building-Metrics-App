import fiona
import random
import os
import gc
import geopandas as gpd

# -----------------------------
# SETTINGS
# -----------------------------
FLOOD_DIR = "/home/grace/FloodFiles"
TARGET_N = 500
TOP_K = 10
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

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
# MAIN LOOP
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
                for feat in src:
                    props = feat["properties"]
                    btype = classify_building(props.get("Simp_type"))
                    if not btype:
                        continue

                    # Attach attributes
                    props["footprint_m2"] = props.get("Shape_Area")
                    props["height_proxy"] = props.get("Est_floors")
                    props["source"] = f"{os.path.basename(gdb_path)}::{layer}"

                    # Random sample
                    samples[btype], seen[btype] = reservoir_update(
                        samples[btype], feat, seen[btype]
                    )

                    # Tallest buildings
                    tallest[btype] = update_top_k(
                        tallest[btype], feat, TOP_K
                    )

        except Exception as e:
            print(f"⚠️ Failed {gdb_path}::{layer}: {e}")

        gc.collect()

# -----------------------------
# EXPORT
# -----------------------------
for btype in ["residential", "commercial"]:
    if not samples[btype]:
        print(f"⚠️ No {btype} buildings sampled")
        continue

    # Merge tallest into sample
    samples[btype].extend(tallest[btype])

    # De-duplicate by Bldg_ID
    unique = {}
    for f in samples[btype]:
        bid = f["properties"].get("Bldg_ID")
        unique[bid] = f
    features = list(unique.values())

    gdf = gpd.GeoDataFrame.from_features(features)

    # TWDB data are Web Mercator
    if gdf.crs is None:
        gdf.set_crs(epsg=3857, inplace=True)

    gdf = gdf.to_crs(epsg=4326)

    # Centroid lat/lon
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y

    gdf.drop(columns="geometry", inplace=True)

    out_csv = f"TX_TWDB_{btype}_sample_plus_top10.csv"
    gdf.to_csv(out_csv, index=False)

    print(f"✅ Exported {len(gdf)} {btype} buildings → {out_csv}")

print("📁 Done.")