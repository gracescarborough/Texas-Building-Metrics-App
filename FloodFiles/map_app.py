import geopandas as gpd
import streamlit as st
import pandas as pd
import numpy as np
import h3
import plotly.express as px
import os

st.set_page_config(layout="wide", page_title="Texas Building Metrics")

GRID_PATH = "tx_grid_classified_embodied.shp"
CENTROIDS_PATH = "sample_centroids_with_stats_and_class_embodied.shp"

@st.cache_data
def load_and_process_data():
    """Load data and downsample if needed"""
    grid = gpd.read_file(GRID_PATH).to_crs("EPSG:4326")
    centroids = gpd.read_file(CENTROIDS_PATH).to_crs("EPSG:4326")
    
    if len(grid) > 50000:
        grid_sampled = grid.sample(n=50000, random_state=42)
    else:
        grid_sampled = grid
    
    grid_sampled['lon'] = grid_sampled.geometry.x
    grid_sampled['lat'] = grid_sampled.geometry.y
    
    return grid_sampled, centroids

try:
    with st.spinner("Loading data..."):
        grid, centroids = load_and_process_data()
    st.success(f"✅ Loaded {len(grid)} grid points")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

st.title("Texas Building Metrics Explorer")
st.sidebar.title("Map Controls")

metric = st.sidebar.selectbox(
    "Select Metric to Display",
    options=["Building Density", "Embodied Energy Intensity (EEI)", "Embodied Carbon Intensity (ECI)"],
    index=0
)

hex_size_option = st.sidebar.slider("Hexagon Size", 1, 3, 2)
hex_size_map = {1: 8, 2: 7, 3: 6}
h3_resolution = hex_size_map[hex_size_option]

metric_config = {
    "Building Density": {
        "column": "density",
        "color_scheme": "YlOrRd",
        "title": "Building Coverage %"
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
    st.error(f"Column '{column_name}' not found")
    st.stop()

@st.cache_data
def aggregate_to_h3(_grid_df, column_name, resolution):
    """Aggregate grid data into H3 hexagons"""
    grid_clean = _grid_df[['lon', 'lat', column_name]].dropna().copy()
    
    grid_clean['h3'] = grid_clean.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lon'], resolution),
        axis=1
    )
    
    hex_agg = grid_clean.groupby('h3').agg({
        column_name: 'mean',
        'lon': 'count'
    }).reset_index()
    
    hex_agg.columns = ['h3', 'avg_value', 'point_count']
    
    # Get hexagon centers
    hex_agg['center'] = hex_agg['h3'].apply(lambda h: h3.cell_to_latlng(h))
    hex_agg['lat'] = hex_agg['center'].apply(lambda x: x[0])
    hex_agg['lon'] = hex_agg['center'].apply(lambda x: x[1])
    
    return hex_agg

with st.spinner("Aggregating hexagons..."):
    hex_data = aggregate_to_h3(grid, column_name, h3_resolution)

st.success(f"✅ Created {len(hex_data)} hexagons")

vmin, vmax = hex_data['avg_value'].min(), hex_data['avg_value'].max()
vmean = hex_data['avg_value'].mean()

# Format display
if column_name == 'density':
    hex_data['display'] = (hex_data['avg_value'] * 100).round(2)
    hex_data['formatted'] = hex_data['display'].astype(str) + '%'
else:
    hex_data['display'] = hex_data['avg_value'].round(2)
    hex_data['formatted'] = hex_data['display'].astype(str)

st.markdown(f"**Currently displaying:** {metric}")

# Create simple scatter mapbox
fig = px.scatter_mapbox(
    hex_data,
    lat='lat',
    lon='lon',
    color='display',
    size='point_count',
    hover_name='formatted',
    hover_data={
        'point_count': True,
        'display': False,
        'lat': False,
        'lon': False
    },
    color_continuous_scale=selected_config['color_scheme'],
    size_max=20,
    zoom=6,
    height=700,
    labels={'display': selected_config['title'], 'point_count': 'Grid Points'}
)

fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": centroids.geometry.y.mean(), "lon": centroids.geometry.x.mean()},
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Statistics")
st.sidebar.metric("Min Value", f"{vmin:.4f}")
st.sidebar.metric("Max Value", f"{vmax:.4f}")
st.sidebar.metric("Mean Value", f"{vmean:.4f}")
st.sidebar.metric("Total Hexagons", f"{len(hex_data):,}")