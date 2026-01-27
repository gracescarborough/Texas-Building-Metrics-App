import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

# -----------------------------
# Load data
# -----------------------------
grid = gpd.read_file("/home/grace/FloodFiles/tx_grid_classified.shp")
centroids = gpd.read_file("/home/grace/FloodFiles/sample_centroids_with_stats.shp")

# -----------------------------
# Detect EEI/ECI columns (handle shapefile truncation)
# -----------------------------
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

# -----------------------------
# Filter valid centroids
# -----------------------------
valid = (centroids[eei_col] > 0) | (centroids[eci_col] > 0)
centroids_valid = centroids[valid]

if len(centroids_valid) == 0:
    raise ValueError("No centroids have EEI/ECI data!")

# -----------------------------
# KD-tree setup
# -----------------------------
centroid_coords = np.vstack([
    centroids_valid.geometry.x.values,
    centroids_valid.geometry.y.values
]).T

grid_coords = np.vstack([
    grid.geometry.x.values,
    grid.geometry.y.values
]).T

tree = cKDTree(centroid_coords)

# -----------------------------
# Interpolation
# -----------------------------
k = min(16, len(centroids_valid))
distances, indices = tree.query(grid_coords, k=k)

p = 2
weights = 1 / (distances**p + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)

# EEI / ECI interpolation
eei_values = centroids_valid[eei_col].values
eci_values = centroids_valid[eci_col].values

grid['eei_interp'] = (eei_values[indices] * weights).sum(axis=1)
grid['eci_interp'] = (eci_values[indices] * weights).sum(axis=1)

print(f"EEI range: {grid['eei_interp'].min():.2f} - {grid['eei_interp'].max():.2f} MJ/m²")
print(f"ECI range: {grid['eci_interp'].min():.2f} - {grid['eci_interp'].max():.2f} kgCO2e/m²")

grid.to_file("/home/grace/FloodFiles/tx_grid_classified.shp")
print("Done! Grid updated with EEI/ECI interpolation.")

