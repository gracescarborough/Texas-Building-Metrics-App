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

persons_per_unit = 2.5 #from census bureau
footprint_per_unit = {"single_family": 220.0, "lowrise_multifamily": 141.0, "highrise_multifamily": 90.0} #from census bureau (low-rise: bought, high-rise: rent)
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
        
EEI_new = {
    "single_family":        get_eei("Residential", 1),
    "lowrise_multifamily":  get_eei("Residential", 4),
    "highrise_multifamily": get_eei("Residential", 10),
}

ECI_new = {
    "single_family":        get_eci("Residential", 1),
    "lowrise_multifamily":  get_eci("Residential", 4),
    "highrise_multifamily": get_eci("Residential", 10),
}

baseline_EE_MJ = 2.47e+13
baseline_EC_kg = 1.31e+12

centroids_gdf = gpd.read_file("/home/grace/FloodFiles/sample_centroids_with_stats.shp")
print("Columns in centroids_gdf:", centroids_gdf.columns.tolist())
df = centroids_gdf.copy()
df["flr_dens"] = df["flr_dens"].fillna(0.0)

if "avg_floors" in df.columns:
    df["avg_floors"] = df["avg_floors"].fillna(1.0)
else:
    df["avg_floors"] = (df["flr_dens"] / df["fp_dens"]).replace([np.inf, -np.inf], 1.0).fillna(1.0)

population_by_year = {
    2030: {
        "low migration": 1_522_039,
        "average migration": 1_836_676,
        "high migration": 2_347_617,
    },
    2035: {
        "low migration": 3_057_263,
        "average migration": 3_657_339,
        "high migration": 4_615_276,
    },
    2040: {
        "low migration": 4_455_870,
        "average migration": 5_366_439,
        "high migration": 6_800_749,
    },
    2045: {
        "low migration": 5_684_477,
        "average migration": 6_926_251,
        "high migration": 8_861_672,
    },
    2050: {
        "low migration": 6_753_846,
        "average migration": 8_344_347,
        "high migration": 10_802_255,
    }
}

sample_df = pd.read_csv("/home/grace/TX_residential_heights2.csv")

floor_counts = sample_df["Est_floors"].dropna()
floor_value_counts = floor_counts.value_counts(normalize=True).sort_index()
floor_levels = floor_value_counts.index.values
floor_probs = floor_value_counts.values

def calculate_scenarios(df, new_pop, year):
    new_units = new_pop / persons_per_unit
    sf_footprint = 220 + 2 * (year - 2024)
    results = {}

    # Sprawl scenario
    thr_low = df["flr_dens"].quantile(0.40)
    candidates1 = df[df["flr_dens"] <= thr_low].copy()
    weights1 = (1.0 - candidates1["flr_dens"]).clip(lower=0.0)
    if weights1.sum() == 0:
        weights1 = pd.Series(1, index=candidates1.index)
    weights1 = weights1 / weights1.sum()
    candidates1["units_assigned"] = (weights1 * new_units).round().astype(int)

    candidates1["added_floor_area_m2"] = candidates1["units_assigned"] * sf_footprint
    candidates1["added_footprint_m2"] = candidates1["added_floor_area_m2"] / candidates1["avg_floors"]
    candidates1["added_EE_MJ"] = candidates1["added_floor_area_m2"] * EEI_new["single_family"]
    candidates1["added_EC_kg"] = candidates1["added_floor_area_m2"] * ECI_new["single_family"]
    
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
    candidates2["added_floor_area_m2"] = candidates2["units_assigned"] * footprint_per_unit["highrise_multifamily"]
    candidates2["added_footprint_m2"] = candidates2["added_floor_area_m2"] / candidates2["avg_floors"]
    candidates2["added_EE_MJ"] = candidates2["added_floor_area_m2"] * EEI_new["highrise_multifamily"]
    candidates2["added_EC_kg"] = candidates2["added_floor_area_m2"] * ECI_new["highrise_multifamily"]
    
    scenario2_total_EE = baseline_EE_MJ + candidates2["added_EE_MJ"].sum()
    scenario2_total_EC = baseline_EC_kg + candidates2["added_EC_kg"].sum()
    results["dense"] = (scenario2_total_EE, scenario2_total_EC, candidates2)

    # Current housing distribution scenario
    candidates3 = df.copy()
    weights3 = pd.Series(1.0 / len(candidates3), index=candidates3.index)
    candidates3["units_assigned"] = (weights3 * new_units).round().astype(int)

    rng = np.random.default_rng(seed=42)
    sampled_floors = rng.choice(floor_levels, size=int(candidates3["units_assigned"].sum()), p=floor_probs)
    avg_sampled_floors = sampled_floors.mean()

    sf_mask = sampled_floors <= 2
    lr_mask = (sampled_floors >= 3) & (sampled_floors <= 6)
    hr_mask = sampled_floors >= 7

    sf_frac = sf_mask.mean()
    lr_frac = lr_mask.mean()
    hr_frac = hr_mask.mean()

    sf_units = new_units * sf_frac
    lr_units = new_units * lr_frac
    hr_units = new_units * hr_frac

    avg_sf_floors = sampled_floors[sf_mask].mean() if sf_mask.any() else 1.0
    avg_lr_floors = sampled_floors[lr_mask].mean() if lr_mask.any() else 4.0
    avg_hr_floors = sampled_floors[hr_mask].mean() if hr_mask.any() else 10.0

    sf_floor_area = sf_units * sf_footprint        * avg_sf_floors
    lr_floor_area = lr_units * footprint_per_unit["lowrise_multifamily"]  * avg_lr_floors
    hr_floor_area = hr_units * footprint_per_unit["highrise_multifamily"] * avg_hr_floors

    sf_ee = sum(get_eei("Residential", int(f)) for f in sampled_floors[sf_mask])
    sf_ec = sum(get_eci("Residential", int(f)) for f in sampled_floors[sf_mask])
    lr_ee = sum(get_eei("Residential", int(f)) for f in sampled_floors[lr_mask])
    lr_ec = sum(get_eci("Residential", int(f)) for f in sampled_floors[lr_mask])
    hr_ee = sum(get_eei("Residential", int(f)) for f in sampled_floors[hr_mask])
    hr_ec = sum(get_eci("Residential", int(f)) for f in sampled_floors[hr_mask])

    total_added_EE = (
        sf_floor_area * (sf_ee / sf_mask.sum()) +
        lr_floor_area * (lr_ee / lr_mask.sum()) +
        hr_floor_area * (hr_ec / hr_mask.sum())
    )
    total_added_EC = (
        sf_floor_area * (sf_ec / sf_mask.sum()) +
        lr_floor_area * (lr_ec / lr_mask.sum()) +
        hr_floor_area * (hr_ec / hr_mask.sum())
    )

    n = len(candidates3)
    candidates3["added_floor_area_m2"] = (sf_floor_area + lr_floor_area + hr_floor_area) / n
    candidates3["added_EE_MJ"]         = total_added_EE / n
    candidates3["added_EC_kg"]         = total_added_EC / n

    scenario3_total_EE = baseline_EE_MJ + total_added_EE
    scenario3_total_EC = baseline_EC_kg + total_added_EC
    results["current_dist"] = (scenario3_total_EE, scenario3_total_EC, candidates3)

    # Baseline
    results["current"] = (baseline_EE_MJ, baseline_EC_kg, None)

    return results

dev_scenarios = ["sprawl", "dense", "current_dist"]

for year, pop_scenarios in population_by_year.items():

    print(f"\n====== {year} ======")

    all_results = {}
    for pop_label, pop in pop_scenarios.items():
        all_results[pop_label] = calculate_scenarios(df, pop, year)

    pop_labels = list(pop_scenarios.keys())

    eei_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)
    eci_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)
    eei_pct_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)
    eci_pct_matrix = pd.DataFrame(index=pop_labels, columns=dev_scenarios)

    for pop_label, res in all_results.items():
        for scen in dev_scenarios:
            ee_val = res[scen][0]
            ec_val = res[scen][1]

            eei_matrix.loc[pop_label, scen] = ee_val
            eci_matrix.loc[pop_label, scen] = ec_val

            eei_pct_matrix.loc[pop_label, scen] = (ee_val - baseline_EE_MJ) / baseline_EE_MJ * 100
            eci_pct_matrix.loc[pop_label, scen] = (ec_val - baseline_EC_kg) / baseline_EC_kg * 100

    eei_matrix = eei_matrix.astype(float)
    eci_matrix = eci_matrix.astype(float)

    eei_pct_matrix = eei_pct_matrix.astype(float).round(2)
    eci_pct_matrix = eci_pct_matrix.astype(float).round(2)

    rename_map = {"current_dist": "Current housing distribution", "sprawl": "Sprawling development", "dense": "Dense development"}

    eei_matrix.rename(columns=rename_map, inplace=True)
    eci_matrix.rename(columns=rename_map, inplace=True)
    eei_pct_matrix.rename(columns=rename_map, inplace=True)
    eci_pct_matrix.rename(columns=rename_map, inplace=True)

    eei_path = f"FloodFiles/Matrices/EEI_matrix_{year}.csv"
    eci_path = f"FloodFiles/Matrices/ECI_matrix_{year}.csv"

    eei_matrix.to_csv(eei_path)
    eci_matrix.to_csv(eci_path)

    eei_pct_path = f"FloodFiles/Matrices/EEI_percent_{year}.csv"
    eci_pct_path = f"FloodFiles/Matrices/ECI_percent_{year}.csv"

    eei_pct_matrix.to_csv(eei_pct_path)
    eci_pct_matrix.to_csv(eci_pct_path)

    print(f"✅ Saved {eei_pct_path}")
    print(f"✅ Saved {eci_pct_path}")
    print(f"✅ Saved {eei_path}")
    print(f"✅ Saved {eci_path}")
