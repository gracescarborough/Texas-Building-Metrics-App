import geopandas as gpd
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pathlib import Path
import h3
import os

st.set_page_config(layout="wide", page_title="Texas Building Metrics")
page = st.sidebar.radio(
    "Select Page",
    ["Main Map", "Future Building Development", "About"]
)

if page == "Main Map":
    GRID_PATH = "FloodFiles/tx_grid_classified.shp"
    CENTROIDS_PATH = "FloodFiles/sample_centroids_with_stats.shp"

    @st.cache_data(show_spinner=False)
    def load_and_process_data(_filehash=None):
        """Load data and downsample if needed"""
        grid = gpd.read_file(GRID_PATH).to_crs("EPSG:4326")
        centroids = gpd.read_file(CENTROIDS_PATH).to_crs("EPSG:4326")
    
        MAX_POINTS = 50000
        if len(grid) > MAX_POINTS:
            high_density = grid[grid['flr_dens'] > 0.05]
            low_density = grid[grid['flr_dens'] <= 0.05]
            if len(high_density) >= MAX_POINTS:
                grid_sampled = high_density.sample(n=MAX_POINTS, random_state=42)
            else:
                remaining = MAX_POINTS - len(high_density)
                n_sample = min(remaining, len(low_density))
                low_density_sampled = low_density.sample(n=n_sample, random_state=42)
                grid_sampled = pd.concat([high_density, low_density_sampled])
        else:
            grid_sampled = grid

        grid_centroids = grid_sampled.geometry.centroid
        grid_sampled['lon'] = grid_centroids.x
        grid_sampled['lat'] = grid_centroids.y

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

    metric_config = {
        "Building Density": {
            "column": "flr_dens",
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
        grid_df = _grid_df.copy()
        cols_needed = ['lon', 'lat', column_name]
        if 'fp_dens' in grid_df.columns:
            cols_needed.append('fp_dens')
        grid_clean = grid_df[cols_needed].copy()
        grid_clean = grid_clean.dropna(subset=['lon', 'lat', column_name])

        grid_clean['h3'] = grid_clean.apply(
            lambda row: h3.latlng_to_cell(row['lat'], row['lon'], resolution),
            axis=1
        )
        
        agg_dict = {column_name: 'mean', 'lon': 'count'}
        if 'fp_dens' in grid_clean.columns:
            agg_dict['fp_dens'] = 'mean'
        hex_agg = grid_clean.groupby('h3').agg(agg_dict).reset_index()
        hex_agg.columns = ['h3', 'avg_value', 'point_count'] + (['fp_dens'] if 'fp_dens' in hex_agg.columns else [])

        def get_boundary_polygon(h3_cell):
            boundary = h3.cell_to_boundary(h3_cell)
            return [[lng, lat] for lat, lng in boundary]
        
        hex_agg['boundary'] = hex_agg['h3'].apply(get_boundary_polygon)
        hex_agg['center'] = hex_agg['h3'].apply(lambda h: h3.cell_to_latlng(h))
        hex_agg['lat'] = hex_agg['center'].apply(lambda x: x[0])
        hex_agg['lon'] = hex_agg['center'].apply(lambda x: x[1])
        return hex_agg

    with st.spinner("Aggregating hexagons..."):
        hex_data = aggregate_to_h3(grid, column_name, h3_resolution)

    hex_data['display_value'] = (hex_data['avg_value'] * 100).round(2).astype(str) + '%' \
        if column_name == 'flr_dens' else hex_data['avg_value'].round(2).astype(str)
    if 'fp_dens' in hex_data.columns:
        hex_data['footprint_display'] = (hex_data['fp_dens'] * 100).round(2).astype(str) + '%'
    else:
        hex_data['footprint_display'] = 'N/A'

    vmin = hex_data['avg_value'].quantile(0.05)
    vmax = hex_data['avg_value'].quantile(0.99)
    hex_data['normalized'] = (hex_data['avg_value'] - vmin) / (vmax - vmin)
    hex_data['normalized'] = hex_data['normalized'].clip(0, 1)
    hex_data['elevation'] = hex_data['normalized'] * 100000 if enable_3d else 0

    def get_color(normalized, color_scheme):
        if color_scheme == "YlOrRd":
            r = int(255)
            g = int(255 * (1 - normalized * 0.7))
            b = int(150 * (1 - normalized))
        else:
            r = int(247 + (34 - 247) * normalized)
            g = int(252 + (94 - 252) * normalized) 
            b = int(240 + (168 - 240) * normalized)
        return [r, g, b, 255]
    
    hex_data['color'] = hex_data['normalized'].apply(lambda x: get_color(x, selected_config['color_scheme']))

    center_lat = centroids.geometry.y.mean()
    center_lon = centroids.geometry.x.mean()
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=6,
        pitch=45 if enable_3d else 0,
        bearing=0
    )

    tooltip = {
        "html": "<b>Building Density</b><br/>"
                "Total Floor Density: {display_value}<br/>"
                "Footprint Density: {footprint_display}",
        "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "12px", "padding": "10px"}
    }

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

    r = pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/light-v10",
        tooltip=tooltip
    )

    st.title("Texas Building Metrics Explorer")
    st.markdown("<p style='text-align: right; color: #999; font-size: 11px; margin-top: -15px;'>By Grace Scarborough | © 2026</p>", unsafe_allow_html=True)
    st.markdown(f"**Currently displaying:** {metric}")

    st.pydeck_chart(r, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Statistics")

    max_val = hex_data['avg_value'].max()

    if metric == "Building Density":
        st.sidebar.metric(
            "Max Total Floor Density",
            f"{max_val * 100:.2f}%"
        )
        if 'fp_dens' in hex_data.columns:
            st.sidebar.metric(
                "Max Footprint Density",
                f"{hex_data['fp_dens'].max() * 100:.2f}%"
            )

    elif metric == "Embodied Energy Intensity (EEI)":
        st.sidebar.metric(
            "Max EEI (MJ/m²)",
            f"{max_val:.1f}"
        )

    elif metric == "Embodied Carbon Intensity (ECI)":
        st.sidebar.metric(
            "Max ECI (kgCO₂e/m²)",
            f"{max_val:.1f}"
        )

elif page == "About":
    st.title("About This App")
    st.markdown("""
    ### Texas Building Metrics Explorer
    This app visualizes building metrics across Texas using hexagon aggregation. 
    You can explore metrics like:
    - **Building Density**
    - **Embodied Energy Intensity (EEI)**
    - **Embodied Carbon Intensity (ECI)**

    **How to Use:**
    1. Select a metric from the sidebar.
    2. Adjust the hexagon size and toggle 3D view.
    3. Hover over hexagons to see values.
    
    This app is created by Grace Scarborough, © 2026.
    """)

elif page == "Future Building Development":
    st.title("Future Building Development Metrics")

    years = [2030, 2035, 2040, 2045, 2050]

    for year in years:
        st.header(f"{year} Projections")

        eei_path = f"FloodFiles/EEI_matrix_{year}.csv"
        eci_path = f"FloodFiles/ECI_matrix_{year}.csv"

        try:
            eei_df = pd.read_csv(eei_path, index_col=0).apply(pd.to_numeric)
            eci_df = pd.read_csv(eci_path, index_col=0).apply(pd.to_numeric)
        except Exception as e:
            st.warning(f"Could not load matrices for {year}: {e}")
            continue

        st.subheader("EEI Matrix (MJ/m²)")
        st.dataframe(eei_df.style.format("{:.2e}"))

        st.subheader("ECI Matrix (kgCO2e/m²)")
        st.dataframe(eci_df.style.format("{:.2e}"))

        st.divider()