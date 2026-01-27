import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

TEXAS_CRS = "EPSG:3083"
GRID_PATH = "/home/grace/FloodFiles/tx_grid_classified.gpkg"
CENTROIDS_PATH = "/home/grace/FloodFiles/sample_centroids_with_stats.gpkg"

print("Loading grid and centroids...")
grid = gpd.read_file(GRID_PATH).to_crs(TEXAS_CRS)
centroids = gpd.read_file(CENTROIDS_PATH).to_crs(TEXAS_CRS)

centroids = centroids[centroids["coverage_f"] > 0]

print(f"Grid points: {len(grid)}")
print(f"Centroids: {len(centroids)}")

centroid_coords = np.vstack([
    centroids.geometry.x.values,
    centroids.geometry.y.values
]).T

grid_coords = np.vstack([
    grid.geometry.x.values,
    grid.geometry.y.values
]).T

tree = cKDTree(centroid_coords)

k = min(16, len(centroids))
distances, indices = tree.query(grid_coords, k=k)

p = 2
weights = 1 / (distances**p + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

density_values = centroids["coverage_f"].values

print("Interpolating floor-area density onto grid...")
grid["density"] = (density_values[indices] * weights).sum(axis=1)

# Clean density
grid["density"] = grid["density"].replace([np.inf, -np.inf], np.nan)
grid["density"] = grid["density"].fillna(0)

# Classify density
bins = [0, 0.02, 0.1, 0.5, 1, 4, 9, np.inf]
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
    grid["density"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Map classes to numeric IDs (adjust as needed)
grid["class_id"] = grid["class"].map(dict(zip(labels, range(1, len(labels)+1))))

print("Saving updated grid as Shapefile...")
grid.to_file("tx_grid_classified.shp", driver="ESRI Shapefile")

print("Density stats:")
print(grid["density"].describe())
print("Max density:", grid["density"].max())

