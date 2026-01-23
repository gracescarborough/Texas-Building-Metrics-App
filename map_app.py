import geopandas as gpd
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from shapely.geometry import Point
import h3
import os

st.set_page_config(layout="wide", page_title="Texas Building Metrics")

GRID_PATH = "tx_grid_classified.shp"
CENTROIDS_PATH = "FloodFiles/sample_centroids_with_density.shp"

@st.cache_data(show_spinner=False)
def load_and_process_data(_filehash=None):

    """Load data and downsample if needed"""
    grid = gpd.read_file(
        "FloodFiles/tx_grid_classified.gpkg"
    )

    centroids = gpd.read_file(
        "FloodFiles/sample_centroids_with_stats.gpkg"
    )
    
    st.write("Grid rows:", len(grid))
    st.write("Centroid rows:", len(centroids))
    st.write("Grid CRS:", grid.crs)
    st.write("Centroid CRS:", centroids.crs)

    if len(grid) > 50000:
        high_density = grid[grid['density'] > 0.05]
        low_density = grid[grid['density'] <= 0.05]
        
        n_sample = min(50000 - len(high_density), len(low_density))
        low_density_sampled = low_density.sample(n=n_sample, random_state=42)
        
        grid_sampled = pd.concat([high_density, low_density_sampled])
    else:
        grid_sampled = grid
    
    grid_sampled['lon'] = grid_sampled.geometry.x
    grid_sampled['lat'] = grid_sampled.geometry.y
    
    return grid_sampled, centroids

try:
    with st.spinner("Loading data..."):
        filehash = os.path.getmtime(GRID_PATH)
        grid, centroids = load_and_process_data(filehash)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

st.sidebar.title("Map Controls")

metric = st.sidebar.selectbox(
    "Select Metric to Display",
    options=["Building Density", "Embodied Energy Intensity (EEI)", "Embodied Carbon Intensity (ECI)"],
    index=0
)

hex_size_option = st.sidebar.slider("Hexagon Size", 1, 3, 2)
hex_size_map = {1: 7, 2: 6, 3: 5}
h3_resolution = hex_size_map[hex_size_option]

enable_3d = st.sidebar.checkbox("Enable 3D View", value=False)
elevation_scale = 500 if enable_3d else 0

metric_config = {
    "Building Density": {
        "column": "density",
        "color_scheme": "YlOrRd",
        "title": "Building Coverage Fraction"
    },
    "Embodied Energy Intensity (EEI)": {
        "column": "eei_interp",
        "color_scheme": "YlOrRd",
        "title": "EEI (MJ/m²)"
    },
    "Embodied Carbon Intensity (ECI)": {
        "column": "eci_interp",
        "color_scheme": "YlGnBu",
        "title": "ECI (kgCO2e/m²)"
    }
}

selected_config = metric_config[metric]
column_name = selected_config["column"]

if column_name not in grid.columns:
    st.error(f"Column '{column_name}' not found. Available: {list(grid.columns)}")
    st.stop()

@st.cache_data
def aggregate_to_h3(_grid_df, column_name, resolution):
    """Aggregate grid data into H3 hexagons"""
    grid_df = _grid_df
    grid_clean = grid_df[['lon', 'lat', column_name]].dropna().copy()
    
    grid_clean['h3'] = grid_clean.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lon'], resolution),
        axis=1
    )
    
    hex_agg = grid_clean.groupby('h3').agg({
        column_name: 'max',
        'lon': 'count'
    }).reset_index()
    
    hex_agg.columns = ['h3', 'avg_value', 'point_count']
    
    def get_boundary_polygon(h3_cell):
        boundary = h3.cell_to_boundary(h3_cell)
        return [[lng, lat] for lat, lng in boundary]
    
    hex_agg['boundary'] = hex_agg['h3'].apply(get_boundary_polygon)
    
    hex_agg['center'] = hex_agg['h3'].apply(
        lambda h: h3.cell_to_latlng(h)
    )
    hex_agg['lat'] = hex_agg['center'].apply(lambda x: x[0])
    hex_agg['lon'] = hex_agg['center'].apply(lambda x: x[1])
    
    return hex_agg

with st.spinner("Aggregating hexagons..."):
    hex_data = aggregate_to_h3(grid, column_name, h3_resolution)
vmin, vmax = hex_data['avg_value'].min(), hex_data['avg_value'].max()
vmean = hex_data['avg_value'].mean()

hex_data['normalized'] = (hex_data['avg_value'] - vmin) / (vmax - vmin) if vmax > vmin else 0
hex_data['normalized'] = hex_data['normalized'].clip(0, 1)

if enable_3d:
    hex_data['elevation'] = hex_data['normalized'] * 100000
else:
    hex_data['elevation'] = 0

if column_name == 'density':
    hex_data['display_value'] = (hex_data['avg_value'] * 100).round(2).astype(str) + '%'
elif column_name == 'eei_interp':
    hex_data['display_value'] = hex_data['avg_value'].round(2).astype(str) + ' MJ/m²'
else:
    hex_data['display_value'] = hex_data['avg_value'].round(2).astype(str) + ' kgCO2e/m²'

def get_color(normalized, color_scheme):
    """Get RGB color based on normalized value"""
    if color_scheme == "YlOrRd":
        r = int(255)
        g = int(255 * (1 - normalized * 0.7))
        b = int(150 * (1 - normalized))
    else:
        r = int(247 + (34 - 247) * normalized)
        g = int(252 + (94 - 252) * normalized) 
        b = int(240 + (168 - 240) * normalized)
    return [r, g, b, 255]

hex_data['color'] = hex_data['normalized'].apply(
    lambda x: get_color(x, selected_config['color_scheme'])
)

polygon_layer = pdk.Layer(
    "PolygonLayer",
    hex_data,
    get_polygon='boundary',
    get_fill_color='color',
    get_elevation='elevation',
    elevation_scale=1,
    filled=True,
    extruded=enable_3d,
    wireframe=False,
    pickable=True,
    auto_highlight=True,
    opacity=1.0
)

center_lat = centroids.geometry.y.mean()
center_lon = centroids.geometry.x.mean()

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=6,
    pitch=45 if enable_3d else 0,
    bearing=0
)

if column_name == "density":
    value_label = "Building density"
else:
    value_label = "Avg " + metric

tooltip = {
    "html": "<b>Hexagon Data</b><br/>"
            "Grid points: {point_count}<br/>"
            f"{value_label}: {{display_value}}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white",
        "fontSize": "12px",
        "padding": "10px"
    }
}

st.title("Texas Building Metrics Explorer")
st.markdown(
    "<p style='text-align: right; color: #999; font-size: 11px; margin-top: -15px;'>By Grace Scarborough | © 2026</p>",
    unsafe_allow_html=True
)
st.markdown(f"**Currently displaying:** {metric}")

r = pdk.Deck(
    layers=[polygon_layer],
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/light-v10",
    tooltip=tooltip
)

st.pydeck_chart(r, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Statistics")
st.sidebar.metric("Min Value", f"{vmin:.4f}")
st.sidebar.metric("Max Value", f"{vmax:.4f}")
