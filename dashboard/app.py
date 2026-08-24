import os
import time
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="SmartRoads Live GIS Telemetry", layout="wide")
st.title("\U0001F697 SmartRoads: Real-Time Road Defect GIS Telemetry")

csv_file = "logs/detections.csv"
SEVERITY_COLORS = {"LOW": "green", "MEDIUM": "orange", "CRITICAL": "red"}

col1, col2 = st.columns([3, 2])

if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
    df = pd.read_csv(csv_file)
    total_detections = len(df)
    
    with col1:
        st.subheader(f"\U0001F4CD Live GIS Defect Map ({total_detections} Verified)")
        center_lat = float(df["Latitude"].iloc[-1]) if not df.empty else 26.9124
        center_lon = float(df["Longitude"].iloc[-1]) if not df.empty else 75.7873
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=18)
        
        for idx, row in df.iterrows():
            color = SEVERITY_COLORS.get(str(row["Severity"]), "blue")
            folium.Marker(
                location=[float(row["Latitude"]), float(row["Longitude"])],
                popup=f"Defect #{idx+1} | {row['Severity']} | Fused: {row.get('Fused_Score_Pct', 'N/A')}",
                tooltip=f"Defect #{idx+1} ({row['Severity']})",
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(m)
            
        st_folium(m, width=750, height=480, key=f"live_map_{total_detections}")

    with col2:
        st.subheader("\U0001F4CA Real-Time Defect Log")
        display_cols = [c for c in ["Timestamp", "Severity", "Cam_70Pct", "IMU_30Pct", "Fused_Score_Pct"] if c in df.columns]
        st.dataframe(df[display_cols].iloc[::-1], use_container_width=True, height=240)
        
        if not df.empty and os.path.exists(str(df["Snapshot_Path"].iloc[-1])):
            st.image(
                df["Snapshot_Path"].iloc[-1],
                caption=f"Latest Sighting: Defect #{total_detections} ({df['Severity'].iloc[-1]})",
                use_container_width=True
            )
else:
    st.info("\U0001F697 Vehicle is navigating the track... Waiting for defect detections.")

time.sleep(1)
st.rerun()
