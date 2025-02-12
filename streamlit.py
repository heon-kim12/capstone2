import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# feature importances from the prediction_analysis.ipynb
feature_importances = {
    "zhvi_growth": 4.34,
    "zhvi_slope": 1.35,
    "zhvi_mean": 0.87,
    "pending_2023-12-31": 0.62,
    "pending_2024-03-31": 0.53,
    "pending_2022-09-30": 0.49,
    "RegionID": 0.46,
    "pending_2019-03-31": 0.45,
    "pending_2022-06-30": 0.45,
    "pending_2019-08-31": 0.42
}
# Convert to DataFrame for plotting
fi_df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)

# A small dictionary to simulate lat/lon for known metro regions.
# In practice, you would use geocoding or an existing GeoDataFrame.
region_coordinates = {
    "New York, NY": {"lat": 40.7128, "lon": -74.0060},
    "Los Angeles, CA": {"lat": 34.0522, "lon": -118.2437},
    "Chicago, IL": {"lat": 41.8781, "lon": -87.6298},
    "Dallas, TX": {"lat": 32.7767, "lon": -96.7970},
    "United States": {"lat": 39.8283, "lon": -98.5795}
}

# Load the Prediction-Ready Dataset
@st.cache_data
def load_dataset(path="prediction_ready_dataset.csv"):
    df = pd.read_csv(path)
    return df

df = load_dataset()
st.title("Real Estate Investment Risk Dashboard")

st.header("High-Risk Regions")
# Filter for high-risk regions (risk == 1)
high_risk = df[df["risk"] == 1].copy()
st.write(f"Total high-risk regions: {len(high_risk)}")

# Display a table of high-risk regions
risk_table = high_risk[["RegionName", "StateName", "risk"]]
st.dataframe(risk_table)

# Select a High-Risk Region for Detailed Explanation
selected_region = st.selectbox("Select a region for explanation:", high_risk["RegionName"].unique())

if selected_region:
    st.subheader(f"Explanation for {selected_region}")
    # Filter for the selected region
    region_data = high_risk[high_risk["RegionName"] == selected_region]
    st.write(region_data[["RegionName", "StateName", "risk"]])
    
    # Plot a bar chart of the top features (using the logistic regression coefficients)
    st.write("Top contributing features (based on model coefficients):")
    fig, ax = plt.subplots()
    ax.bar(fi_df["Feature"], fi_df["Importance"], color="skyblue")
    ax.set_ylabel("Absolute Coefficient")
    ax.set_title("Feature Importance (Model Coefficients)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Optionally, you might show the actual time series summary for the selected region
    st.write("Time series summary features for the region:")
    ts_cols = [col for col in region_data.columns if any(prefix in col for prefix in ["zhvi_", "zori_", "pending_"])]
    st.dataframe(region_data[ts_cols].T)  # Transpose for easier viewing

# Map High-Risk Regions Geographically
st.header("Geographical Map of High-Risk Regions")
# For mapping, we need a DataFrame with 'lat' and 'lon'
def add_coordinates(row):
    region = row["RegionName"]
    # Use our simulated dictionary; if not found, use NaN
    coords = region_coordinates.get(region, {"lat": np.nan, "lon": np.nan})
    return pd.Series([coords["lat"], coords["lon"]], index=["lat", "lon"])

high_risk_coords = high_risk.apply(add_coordinates, axis=1)
map_df = pd.concat([high_risk[["RegionName", "StateName"]].reset_index(drop=True), high_risk_coords.reset_index(drop=True)], axis=1)
# Remove rows with missing coordinates
map_df = map_df.dropna(subset=["lat", "lon"])
st.map(map_df)

st.write("The map shows the approximate locations of high-risk regions. In practice, you would use geocoded coordinates for precise mapping.")

st.write("### Summary")
st.write("""
Our dashboard identifies high-risk regions using a predictive model that flags regions with negative home value growth as risky. 
Key features such as the growth rate and trend of the Zillow Home Value Index play a dominant role in these predictions.
The dashboard allows you to view detailed explanations for each high-risk region and explore their geographic distribution.
This insight empowers investors, homebuyers, and portfolio managers to make more informed decisions in a volatile real estate market.
""")
