import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import box
import os

# Page config
st.set_page_config(
    page_title="CropSIF - Drought Detection",
    page_icon="🌾",
    layout="wide"
)

# Constants
IOWA_BOUNDS = [-94.5, 41.5, -93.0, 42.5]
JULY_DOYS = [177, 185, 193, 201, 209]
DOY_LABELS = {177: "Jun 25", 185: "Jul 3", 193: "Jul 11", 201: "Jul 19", 209: "Jul 27"}

@st.cache_data
def load_sif_data(year, doy):
    """Load Iowa SIF data"""
    filepath = f"cropsif-demo/data/GOSIF_{year}{doy}.tif"
    if not os.path.exists(filepath):
        return None
    
    with rasterio.open(filepath) as src:
        data = src.read(1) * 0.0001  # Scale factor
        valid = (data > 0) & (data < 6)
        data = np.where(valid, data, np.nan)
        return data

@st.cache_data
def get_timeseries(year):
    """Get July timeseries for a year"""
    values = []
    for doy in JULY_DOYS:
        data = load_sif_data(year, doy)
        if data is not None:
            values.append(np.nanmean(data))
        else:
            values.append(None)
    return values

# Header
st.title("🌾 CropSIF: Satellite Drought Detection")
st.markdown("**Real-time crop stress monitoring using Solar-Induced Fluorescence**")

# Sidebar
st.sidebar.header("Controls")
selected_doy = st.sidebar.select_slider(
    "Select Date",
    options=JULY_DOYS,
    value=201,
    format_func=lambda x: DOY_LABELS[x]
)

# Load data
sif_2012 = load_sif_data(2012, selected_doy)
sif_2023 = load_sif_data(2023, selected_doy)

# Top metrics
col1, col2, col3, col4 = st.columns(4)

if sif_2012 is not None and sif_2023 is not None:
    mean_2012 = np.nanmean(sif_2012)
    mean_2023 = np.nanmean(sif_2023)
    pct_diff = ((mean_2012 - mean_2023) / mean_2023) * 100
    
    col1.metric("2023 SIF (Normal)", f"{mean_2023:.3f}", "Baseline")
    col2.metric("2012 SIF (Drought)", f"{mean_2012:.3f}", f"{pct_diff:+.1f}%")
    
    if pct_diff < -25:
        stress_level = "🔴 SEVERE"
    elif pct_diff < -15:
        stress_level = "🟠 MODERATE"
    elif pct_diff < -5:
        stress_level = "🟡 MILD"
    else:
        stress_level = "🟢 NORMAL"
    
    col3.metric("Stress Level", stress_level)
    col4.metric("Date", DOY_LABELS[selected_doy])

# Main content
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📍 Spatial Analysis")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    if sif_2023 is not None:
        im1 = axes[0].imshow(sif_2023, cmap='YlGn', vmin=0.3, vmax=0.8,
                             extent=[IOWA_BOUNDS[0], IOWA_BOUNDS[2], 
                                     IOWA_BOUNDS[1], IOWA_BOUNDS[3]])
        axes[0].set_title('2023 (Normal)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    if sif_2012 is not None:
        im2 = axes[1].imshow(sif_2012, cmap='YlGn', vmin=0.3, vmax=0.8,
                             extent=[IOWA_BOUNDS[0], IOWA_BOUNDS[2], 
                                     IOWA_BOUNDS[1], IOWA_BOUNDS[3]])
        axes[1].set_title('2012 (Drought)')
        axes[1].set_xlabel('Longitude')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Anomaly map
    if sif_2012 is not None and sif_2023 is not None:
        st.subheader("🎯 Stress Anomaly Map")
        pct_anomaly = ((sif_2012 - sif_2023) / sif_2023) * 100
        
        fig2, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pct_anomaly, cmap='RdYlGn', vmin=-50, vmax=50,
                       extent=[IOWA_BOUNDS[0], IOWA_BOUNDS[2], 
                               IOWA_BOUNDS[1], IOWA_BOUNDS[3]])
        ax.set_title('2012 Drought Anomaly (% vs 2023)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='SIF Anomaly (%)')
        st.pyplot(fig2)
        plt.close()

with right_col:
    st.subheader("📈 July Time Series")
    
    ts_2012 = get_timeseries(2012)
    ts_2023 = get_timeseries(2023)
    
    fig3, ax = plt.subplots(figsize=(10, 5))
    
    # Convert None to np.nan for plotting
    ts_2012_clean = [v if v is not None else np.nan for v in ts_2012]
    ts_2023_clean = [v if v is not None else np.nan for v in ts_2023]
    
    ax.plot(JULY_DOYS, ts_2023_clean, 'g-o', linewidth=2, markersize=10, label='2023 (Normal)')
    ax.plot(JULY_DOYS, ts_2012_clean, 'r-s', linewidth=2, markersize=10, label='2012 (Drought)')
    
    # Only fill if we have valid data
    if all(v is not None for v in ts_2012) and all(v is not None for v in ts_2023):
        ax.fill_between(JULY_DOYS, ts_2012_clean, ts_2023_clean, alpha=0.3, color='red')
    
    ax.axvline(x=selected_doy, color='blue', linestyle='--', alpha=0.5, label='Selected Date')
    
    ax.set_xlabel('Day of Year', fontsize=12)
    ax.set_ylabel('SIF (W/m²/sr/μm)', fontsize=12)
    ax.set_title('Iowa Corn Belt: Photosynthetic Activity', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(JULY_DOYS)
    ax.set_xticklabels([DOY_LABELS[d] for d in JULY_DOYS])
    ax.set_ylim(0.35, 0.65)
    
    st.pyplot(fig3)
    plt.close()
    
    # Stats table
    st.subheader("📊 Period Statistics")
    stats_data = []
    for i, doy in enumerate(JULY_DOYS):
        if ts_2012[i] and ts_2023[i]:
            diff = ((ts_2012[i] - ts_2023[i]) / ts_2023[i]) * 100
            stats_data.append({
                "Date": DOY_LABELS[doy],
                "2023 SIF": f"{ts_2023[i]:.4f}",
                "2012 SIF": f"{ts_2012[i]:.4f}",
                "Anomaly": f"{diff:+.1f}%"
            })
    st.table(stats_data)

st.markdown("---")
st.markdown("""
**Data Source:** GOSIF v2 (University of New Hampshire) | **Resolution:** 0.05° (~5km)  
**Study Area:** Iowa Corn Belt | **Contact:** github.com/tlaw6500
""")
