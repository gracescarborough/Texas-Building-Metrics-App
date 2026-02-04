import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box

TEXAS_CRS = "EPSG:3083"
GRID_PATH = "/home/grace/FloodFiles/tx_grid_classified.shp"
CENTROIDS_PATH = "/home/grace/FloodFiles/sample_centroids_with_stats.shp"

grid = gpd.read_file(GRID_PATH).to_crs(TEXAS_CRS)
centroids = gpd.read_file(CENTROIDS_PATH).to_crs(TEXAS_CRS)

centroids = centroids[centroids["fp_dens"] > 0]

centroid_coords = np.vstack([
    centroids.geometry.x.values,
    centroids.geometry.y.values
]).T

grid_centroids = grid.geometry.centroid

grid_coords = np.vstack([
    grid_centroids.x.values,
    grid_centroids.y.values
]).T

tree = cKDTree(centroid_coords)

k = min(16, len(centroids))
distances, indices = tree.query(grid_coords, k=k)

p = 2
weights = 1 / (distances**p + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

print("Interpolating metrics onto grid...")

cols_to_interp = ["fp_dens", "flr_dens"]

for col in cols_to_interp:
    values = centroids[col].values
    grid[col] = (values[indices] * weights).sum(axis=1)
    grid[col] = grid[col].replace([np.inf, -np.inf], np.nan).fillna(0)

bins = [0, 0.008, 0.04, 0.08, 0.12, 0.2, 2, np.inf]
labels = [
    "Very Low",
    "Low",
    "Moderate",
    "Medium",
    "High",
    "Very High",
    "Extreme"
]

assert len(labels) == len(bins) - 1

grid["class"] = pd.cut(
    grid["flr_dens"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

grid["class_id"] = grid["class"].map(dict(zip(labels, range(1, len(labels)+1))))

print("Saving updated grid as Shapefile...")
grid.to_file(GRID_PATH, driver="ESRI Shapefile")
print("Density stats:")
print(grid["flr_dens"].describe())
print("Max density:", grid["flr_dens"].max())

grid_centroids = grid.geometry.centroid
grid_centroids = grid_centroids[~grid_centroids.is_empty]

coords = np.vstack([grid_centroids.x.values, grid_centroids.y.values]).T

tree2 = cKDTree(coords)
dists, _ = tree2.query(coords, k=2)
cell_size = np.median(dists[:, 1])
half = cell_size / 2.0

polys = [box(x - half, y - half, x + half, y + half)
         for x, y in zip(grid_centroids.x.values, grid_centroids.y.values)]

grid_poly = gpd.GeoDataFrame(
    grid.loc[grid_centroids.index][["flr_dens", "fp_dens"]],
    geometry=polys,
    crs=grid.crs
)

print("Invalid geometries:", grid_poly.is_valid.value_counts())

POLY_PATH = "/home/grace/FloodFiles/polygons_for_eei_eci.shp"
grid_poly.to_file(POLY_PATH)
print(f"Polygon grid saved to: {POLY_PATH}")
print("Mean polygon area (m²):", grid_poly.geometry.area.mean())

colors = {
    "Very Low": "#fff7bc",
    "Low": "#fee391",
    "Moderate": "#fec44f",
    "Medium": "#fe9929",
    "High": "#ec7014",
    "Very High": "#cc4c02",
    "Extreme": "#800026"
}

fig, ax = plt.subplots(figsize=(12, 12))
for cls in labels:
    subset = grid[grid["class"] == cls]
    if len(subset) > 0:
        subset.plot(ax=ax, color=colors[cls], linewidth=0)
legend_patches = [
    mpatches.Patch(color=colors[l], label=l)
    for l in labels
]
ax.set_axis_off()
plt.tight_layout()
plt.savefig("texas_density_map.png", dpi=300)
plt.show()