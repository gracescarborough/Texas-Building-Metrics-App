import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

grid = gpd.read_file("/home/grace/FloodFiles/polygons_for_eei_eci.shp")
centroids = gpd.read_file("/home/grace/FloodFiles/sample_centroids_with_stats.shp")

eei_col = None
eci_col = None
for col in centroids.columns:
    if "eei" in col.lower():
        eei_col = col
    if "eci" in col.lower():
        eci_col = col

if eei_col is None or eci_col is None:
    raise ValueError(f"Could not find EEI/ECI columns in centroids. Found columns: {centroids.columns}")

print(f"Using EEI column: {eei_col}, ECI column: {eci_col}")

valid = (centroids[eei_col] > 0) | (centroids[eci_col] > 0)
centroids_valid = centroids[valid]

if len(centroids_valid) == 0:
    raise ValueError("No centroids have EEI/ECI data!")

centroid_coords = np.vstack([
    centroids_valid.geometry.x.values,
    centroids_valid.geometry.y.values
]).T

grid_centroids = grid.geometry.centroid

grid_coords = np.vstack([
    grid_centroids.x.values,
    grid_centroids.y.values
]).T

tree = cKDTree(centroid_coords)

k = min(16, len(centroids_valid))
distances, indices = tree.query(grid_coords, k=k)

p = 2
weights = 1 / (distances**p + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

eei_values = centroids_valid[eei_col].values
eci_values = centroids_valid[eci_col].values

grid['eei_interp'] = (eei_values[indices] * weights).sum(axis=1)
grid['eci_interp'] = (eci_values[indices] * weights).sum(axis=1)

print(f"EEI range: {grid['eei_interp'].min():.2f} - {grid['eei_interp'].max():.2f} MJ/m²")
print(f"ECI range: {grid['eci_interp'].min():.2f} - {grid['eci_interp'].max():.2f} kgCO2e/m²")

print("Grid updated with EEI/ECI interpolation.")

grid_m = grid.to_crs("EPSG:3083")

grid_m["area_m2"] = grid_m.geometry.area

# --- ADD TOTAL EEI / ECI COLUMNS ---

required = ["eei_interp", "eci_interp", "flr_dens", "area_m2"]
missing = [c for c in required if c not in grid_m.columns]
if missing:
    raise ValueError(f"Missing required columns for totals: {missing}")

grid_m["total_eei_MJ"] = (
    grid_m["eei_interp"] *
    grid_m["flr_dens"] *
    grid_m["area_m2"]
)

grid_m["total_eci_kg"] = (
    grid_m["eci_interp"] *
    grid_m["flr_dens"] *
    grid_m["area_m2"]
)

grid_final = grid_m.to_crs("EPSG:4326")
grid_final.to_file("/home/grace/FloodFiles/tx_grid_classified.shp")
print("Grid updated with total EEI/ECI columns.")

total_eei_MJ = grid_m["total_eei_MJ"].sum()
total_eci_kg = grid_m["total_eci_kg"].sum()

print("\n===== Texas Totals =====")
print(f"Total embodied energy: {total_eei_MJ:,.2e} MJ")
print(f"Total embodied carbon: {total_eci_kg:,.2e} kg CO2e")
print(f"Total embodied energy: {total_eei_MJ/1e12:,.2f} TJ")
print(f"Total embodied carbon: {total_eci_kg/1e9:,.2f} Mt CO2e")
print("Mean cell area (m²):", grid_m["area_m2"].mean())
print(grid.geom_type.value_counts())
print("CRS:", grid.crs)
print(grid.geometry.area.describe())

centroids_valid = centroids_valid.to_crs(grid.crs)

eei_vmin = grid["eei_interp"].quantile(0.02)
eei_vmax = grid["eei_interp"].quantile(0.98)
eci_vmin = grid["eci_interp"].quantile(0.02)
eci_vmax = grid["eci_interp"].quantile(0.98)

#ECI
fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
ax.set_facecolor("white") 
grid.plot(
    ax=ax,
    column="eci_interp",
    cmap="Blues",
    vmin=eci_vmin,
    vmax=eci_vmax,
    linewidth=0,
    legend=True,
    legend_kwds={
        "label": "ECI (kgCO₂e/m²)",
        "shrink": 0.7
    }
)

cbar = fig.axes[-1]
for tick in cbar.get_yticklabels():
    tick.set_visible(True)
    tick.set_color("black")
    tick.set_fontsize(16)

cbar.tick_params(colors="black")
cbar.set_ylabel("kgCO₂e/m²", fontsize=18, color="black")
cbar.yaxis.get_offset_text().set_color("black")
for spine in cbar.spines.values():
    spine.set_edgecolor("black")

ax.set_title("Texas Embodied Carbon Intensity (ECI)", fontsize=22, color="black")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("texas_eci_map.png", dpi=300, bbox_inches="tight")
plt.show()

#EEI
fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
ax.set_facecolor("white") 

plot = grid.plot(
    ax=ax,
    column="eei_interp",
    cmap="YlOrRd",
    vmin=eei_vmin,
    vmax=eei_vmax,
    linewidth=0,
    legend=True,
    legend_kwds={
        "label": "MJ/m²",
        "shrink": 0.7
    }
)

cbar = fig.axes[-1]
for tick in cbar.get_yticklabels():
    tick.set_visible(True)
    tick.set_color("black")
    tick.set_fontsize(16)

cbar.tick_params(colors="black")
cbar.set_ylabel("MJ/m²", fontsize=18, color="black")
cbar.yaxis.get_offset_text().set_color("black")
for spine in cbar.spines.values():
    spine.set_edgecolor("black")

ax.set_title("Texas Embodied Energy Intensity (EEI)", fontsize=22, color="black")
ax.set_axis_off()

plt.tight_layout()
plt.savefig("texas_eei_map.png", dpi=300, bbox_inches="tight")
plt.show()

