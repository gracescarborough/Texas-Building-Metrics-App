import geopandas as gpd
import fiona
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

flood_paths = glob.glob("/home/grace/FloodFiles/FPR*Bldgs_SVI_Pop_Feb2025.gdb")
centroids_path = "/home/grace/FloodFiles/sample_centroids_with_stats.shp"

try:
    centroids_gdf = gpd.read_file(centroids_path)
except Exception as e:
    raise RuntimeError(f"Could not load centroids_gdf: {e}")

def load_first_polygon_layer(gdb_path):
    """Load the first polygon layer inside a GDB, safely and with minimal memory."""
    try:
        layers = fiona.listlayers(gdb_path)
    except Exception as e:
        print(f"  ❌ Could not list layers: {e}")
        return None

    for layer in layers:
        try:
            gdf = gpd.read_file(gdb_path, layer=layer)
            if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                return gdf
        except Exception:
            continue
    return None

results = []
for path in flood_paths:
    gdf = load_first_polygon_layer(path)
    if gdf is None:
        print(f"{path} → ❌ No polygon layer found.")
        continue
    gdf = gdf.to_crs("EPSG:5070")
    gdf["area_sq_mi"] = gdf.area / 2_589_988.110336
    total_area = float(gdf["area_sq_mi"].sum())
    results.append((path, total_area))

for p, a in results:
    print(f"{p.split('/')[-1]} → {a:.2f} sq mi")

persons_per_unit = 2.6
footprint_per_unit = {
    "single_family": 150.0,
    "lowrise_multifamily": 60.0,
    "highrise_multifamily": 30.0
}
EEI_new = {"single_family": 5000, "lowrise_multifamily": 6000.0, "highrise_multifamily": 7000.0}
ECI_new = {"single_family": 200.0, "lowrise_multifamily": 350.0, "highrise_multifamily": 500.0}

baseline_EE_MJ = 7.885e+12
baseline_EC_kg = 4.112e+11

centroids_gdf = gpd.read_file("/home/grace/FloodFiles/sample_centroids_with_stats.shp")
print("Columns in centroids_gdf:", centroids_gdf.columns.tolist())
df = centroids_gdf.copy()
df["flr_dens"] = df["flr_dens"].fillna(0.0)

pop_scenarios = {
    "low": 3_057_263 ,
    "average": 3_657_339,
    "high": 4_615_276 
}

def calculate_scenarios(df, new_pop):
    new_units = new_pop / persons_per_unit
    results = {}

    # Sprawl scenario
    thr_low = df["flr_dens"].quantile(0.40)
    candidates1 = df[df["flr_dens"] <= thr_low].copy()
    weights1 = (1.0 - candidates1["flr_dens"]).clip(lower=0.0)
    if weights1.sum() == 0:
        weights1 = pd.Series(1, index=candidates1.index)
    weights1 = weights1 / weights1.sum()
    candidates1["units_assigned"] = (weights1 * new_units).round().astype(int)
    candidates1["added_footprint_m2"] = candidates1["units_assigned"] * footprint_per_unit["single_family"]
    candidates1["added_EE_MJ"] = candidates1["added_footprint_m2"] * EEI_new["single_family"]
    candidates1["added_EC_kg"] = candidates1["added_footprint_m2"] * ECI_new["single_family"]
    scenario1_total_EE = baseline_EE_MJ + candidates1["added_EE_MJ"].sum()
    scenario1_total_EC = baseline_EC_kg + candidates1["added_EC_kg"].sum()
    results["sprawl"] = (scenario1_total_EE, scenario1_total_EC, candidates1)

    # Dense scenario
    thr_high = df["flr_dens"].quantile(0.60)
    candidates2 = df[df["flr_dens"] >= thr_high].copy()
    weights2 = candidates2["flr_dens"].clip(lower=1e-6)
    if weights2.sum() == 0:
        weights2 = pd.Series(1, index=candidates2.index)
    weights2 = weights2 / weights2.sum()
    candidates2["units_assigned"] = (weights2 * new_units).round().astype(int)
    candidates2["added_footprint_m2"] = candidates2["units_assigned"] * footprint_per_unit["highrise_multifamily"]
    candidates2["added_EE_MJ"] = candidates2["added_footprint_m2"] * EEI_new["highrise_multifamily"]
    candidates2["added_EC_kg"] = candidates2["added_footprint_m2"] * ECI_new["highrise_multifamily"]
    scenario2_total_EE = baseline_EE_MJ + candidates2["added_EE_MJ"].sum()
    scenario2_total_EC = baseline_EC_kg + candidates2["added_EC_kg"].sum()
    results["dense"] = (scenario2_total_EE, scenario2_total_EC, candidates2)

    # half dense half sprawling
    half_units = new_units / 2

    candidates1_mixed = candidates1.copy()
    candidates1_mixed["units_assigned"] = (weights1 * half_units).round().astype(int)
    candidates1_mixed["added_footprint_m2"] = candidates1_mixed["units_assigned"] * footprint_per_unit["single_family"]
    candidates1_mixed["added_EE_MJ"] = candidates1_mixed["added_footprint_m2"] * EEI_new["single_family"]
    candidates1_mixed["added_EC_kg"] = candidates1_mixed["added_footprint_m2"] * ECI_new["single_family"]

    candidates2_mixed = candidates2.copy()
    candidates2_mixed["units_assigned"] = (weights2 * half_units).round().astype(int)
    candidates2_mixed["added_footprint_m2"] = candidates2_mixed["units_assigned"] * footprint_per_unit["highrise_multifamily"]
    candidates2_mixed["added_EE_MJ"] = candidates2_mixed["added_footprint_m2"] * EEI_new["highrise_multifamily"]
    candidates2_mixed["added_EC_kg"] = candidates2_mixed["added_footprint_m2"] * ECI_new["highrise_multifamily"]

    mixed_total_EE = baseline_EE_MJ + candidates1_mixed["added_EE_MJ"].sum() + candidates2_mixed["added_EE_MJ"].sum()
    mixed_total_EC = baseline_EC_kg + candidates1_mixed["added_EC_kg"].sum() + candidates2_mixed["added_EC_kg"].sum()
    candidates_mixed = pd.concat([candidates1_mixed, candidates2_mixed])
    results["mixed"] = (mixed_total_EE, mixed_total_EC, candidates_mixed)

    # Baseline
    results["current"] = (baseline_EE_MJ, baseline_EC_kg, None)

    return results

all_results = {}
for pop_label, pop in pop_scenarios.items():
    all_results[pop_label] = calculate_scenarios(df, pop)

fmt = lambda x: f"{x:,.2f}"
for pop_label, res in all_results.items():
    print(f"Population scenario: {pop_label}")
    for scen, vals in res.items():
        ee, ec = vals[0], vals[1]
        print(f"  {scen}: EE = {fmt(ee)} MJ, EC = {fmt(ec)} kg CO2e")
    print()

pop_labels = list(pop_scenarios.keys())
dev_scenarios = ["current", "sprawl", "mixed", "dense"]
eei_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)
eci_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)

for pop_label, res in all_results.items():
    for scen in dev_scenarios:
        eei_matrix.loc[pop_label, scen] = res[scen][0]
        eci_matrix.loc[pop_label, scen] = res[scen][1]

eei_matrix = eei_matrix.applymap(lambda x: round(x, 2))
eci_matrix = eci_matrix.applymap(lambda x: round(x, 2))
eei_matrix.to_csv("FloodFiles/EEI_matrix.csv")
eci_matrix.to_csv("FloodFiles/ECI_matrix.csv")

print("✅ EEI and ECI matrices saved as 'EEI_matrix.csv' and 'ECI_matrix.csv'")

def per_class_summary(candidates, label, pop_label):
    if candidates is None:
        return pd.DataFrame()
    s = candidates.groupby(pd.cut(
        candidates["flr_dens"], 
        bins=[-1,0.0001,0.001,0.005,0.02,1], 
        labels=[1,2,3,4,5]
    )).agg(
        units_assigned=("units_assigned","sum"),
        added_footprint_m2=("added_footprint_m2","sum"),
        added_EE_MJ=("added_EE_MJ","sum"),
        added_EC_kg=("added_EC_kg","sum")
    ).fillna(0)
    s["scenario"] = label
    s["pop_scenario"] = pop_label
    return s.reset_index()

per_class_list = []

for pop_label, res in all_results.items():
    for scen, vals in res.items():
        candidates = vals[2]
        per_class_list.append(per_class_summary(candidates, scen, pop_label))

per_class_df = pd.concat(per_class_list)
per_class_df.to_csv("per_class_breakdown.csv", index=False)