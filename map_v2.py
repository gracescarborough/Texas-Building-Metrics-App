import geopandas as gpd
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

TEXAS_CRS    = "EPSG:3083"
GRID_RES_M   = 750
ALL_BUILDINGS_PATH = "/home/grace/FloodFiles/all_buildings.parquet"
OUTPUT_SHP   = "/home/grace/FloodFiles/sample_centroids_with_stats.shp"
OUTPUT_CSV   = "/home/grace/FloodFiles/sample_centroids_with_stats.csv"

def get_eei(building_type: str, stories: int) -> float:
    """Calculate Embodied Energy Intensity based on building type and number of stories."""
    match building_type:
        case "Industrial" | "Agricultural":
            return 4909
        case "Commercial" | "Public":
            return 5163 if stories <= 6 else 10500
        case "Residential":
            if stories == 1:
                return 6700
            elif stories == 2:
                return 6300
            else:
                return 6500
        case _:
            return 5500

def get_eci(building_type: str, stories: int) -> float:
    """Calculate Embodied Carbon Intensity based on building type and number of stories."""
    match building_type:
        case "Industrial" | "Agricultural":
            return 509
        case "Commercial" | "Public":
            return 400 if stories <= 6 else 0.94 * stories + 282
        case "Residential":
            return 284.7 + 1.32 * stories
        case _:
            return 350

M2_PER_MI2 = 1_609.344 ** 2
CELL_AREA_M2 = GRID_RES_M ** 2

print("Loading buildings...")
buildings = gpd.read_parquet(ALL_BUILDINGS_PATH)
print(f"  {len(buildings)} buildings loaded")
print(buildings["btype"].value_counts())

buildings["grid_x"] = np.floor(buildings.geometry.x / GRID_RES_M).astype(int)
buildings["grid_y"] = np.floor(buildings.geometry.y / GRID_RES_M).astype(int)
buildings["cell_id"] = buildings["grid_x"].astype(str) + "_" + buildings["grid_y"].astype(str)

buildings["floor_area_m2"] = buildings["footprint_m2"] * buildings["Est_floors"]

floors = buildings["Est_floors"].astype(int)
btype  = buildings["btype"]

buildings["EEI"] = np.select(
    [
        (btype == "Industrial") | (btype == "Agricultural"),
        ((btype == "Commercial") | (btype == "Institutional")) & (floors <= 6),
        ((btype == "Commercial") | (btype == "Institutional")) & (floors > 6),
        (btype == "Residential") & (floors == 1),
        (btype == "Residential") & (floors == 2),
        (btype == "Residential") & (floors >= 3),
    ],
    [
        4909,
        5163,
        10500,
        6700,
        6300,
        6500,
    ],
    default=5500
)

buildings["ECI"] = np.select(
    [
        (btype == "Industrial") | (btype == "Agricultural"),
        ((btype == "Commercial") | (btype == "Institutional")) & (floors <= 6),
        ((btype == "Commercial") | (btype == "Institutional")) & (floors > 6),
        btype == "Residential",
    ],
    [
        509,
        400,
        0.94 * floors + 282,
        284.7 + 1.32 * floors,
    ],
    default=350
)
buildings["ee_mj"]    = buildings["floor_area_m2"] * buildings["EEI"]
buildings["ec_kgco2"] = buildings["floor_area_m2"] * buildings["ECI"]

buildings["fp_x_floors"] = buildings["footprint_m2"] * buildings["Est_floors"]

print("Aggregating to grid cells...")
grp = buildings.groupby("cell_id")

agg = grp.agg(
    building_count   = ("footprint_m2",  "count"),
    total_footprint  = ("footprint_m2",  "sum"),
    total_floor_area = ("floor_area_m2", "sum"),
    fp_x_floors_sum  = ("fp_x_floors",   "sum"),
    total_ee_mj      = ("ee_mj",         "sum"),
    total_ec_kgco2e  = ("ec_kgco2",      "sum"),
    mean_x           = ("grid_x",        "first"),
    mean_y           = ("grid_y",        "first"),
).reset_index()

agg["cx"] = (agg["mean_x"] + 0.5) * GRID_RES_M
agg["cy"] = (agg["mean_y"] + 0.5) * GRID_RES_M

agg["fp_dens"]  = agg["total_footprint"]  / M2_PER_MI2
agg["flr_dens"] = agg["total_floor_area"] / M2_PER_MI2

agg["avg_floors"] = (agg["fp_x_floors_sum"] / agg["total_footprint"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)

agg["avg_eei_mj_m2"]      = (agg["total_ee_mj"]    / agg["total_floor_area"]).fillna(0)
agg["avg_eci_kgco2e_m2"]  = (agg["total_ec_kgco2e"] / agg["total_floor_area"]).fillna(0)

from shapely.geometry import Point
geometry = [Point(x, y) for x, y in zip(agg["cx"], agg["cy"])]

centroids_gdf = gpd.GeoDataFrame(
    {
        "sample_id":         range(len(agg)),
        "building_count":    agg["building_count"].values,
        "flr_area":          agg["total_floor_area"].values,
        "flr_dens":          agg["flr_dens"].values,
        "fp_area":           agg["total_footprint"].values,
        "fp_dens":           agg["fp_dens"].values,
        "avg_floors":        agg["avg_floors"].values,
        "total_ee_mj":       agg["total_ee_mj"].values,
        "total_ec_kgco2e":   agg["total_ec_kgco2e"].values,
        "avg_eei_mj_m2":     agg["avg_eei_mj_m2"].values,
        "avg_eci_kgco2e_m2": agg["avg_eci_kgco2e_m2"].values,
    },
    geometry=geometry,
    crs=TEXAS_CRS
)

print(f"  {len(centroids_gdf)} grid cells with buildings")
print("Density stats:")
print(centroids_gdf["flr_dens"].describe())

centroids_gdf.to_file(OUTPUT_SHP)
centroids_gdf.drop(columns="geometry").assign(
    x_m=agg["cx"].values,
    y_m=agg["cy"].values
).to_csv(OUTPUT_CSV, index=False)

print(f"Saved → {OUTPUT_SHP}")
print(f"Saved → {OUTPUT_CSV}")