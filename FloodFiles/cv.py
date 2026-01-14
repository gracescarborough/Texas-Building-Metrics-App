import geopandas as gpd
gdf = gpd.read_file("tx_grid_classified_embodied.shp")
gdf.to_file("tx_grid_classified_embodied.gpkg", driver="GPKG")
