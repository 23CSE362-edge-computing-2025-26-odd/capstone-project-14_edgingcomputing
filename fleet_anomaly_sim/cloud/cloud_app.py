import os, json, time, threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from collections import deque

# --- Environment Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MAX_REPORTS = int(os.getenv("MAX_REPORTS", "500"))

# --- In-memory Data Stores ---
reports_stream = deque(maxlen=MAX_REPORTS)
fleet_snapshot = {}
latest_reports = {} # Stores the most recent report from each edge device

# --- MQTT Client Setup ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to MQTT."""
    print("[cloud] Connected to MQTT Broker.")
    client.subscribe("cloud/reports") # From Edge nodes
    client.subscribe("cloud/fleet_snapshot") # From Fog

def on_message(client, userdata, msg):
    """Callback for when a message is received from MQTT."""
    global fleet_snapshot
    if msg.topic == "cloud/reports":
        data = json.loads(msg.payload.decode())
        reports_stream.appendleft(data)
        latest_reports[data['edge_id']] = data # Keep track of the latest report
    elif msg.topic == "cloud/fleet_snapshot":
        fleet_snapshot = json.loads(msg.payload.decode())

def mqtt_thread_func():
    """Function to run the MQTT client loop in a separate thread."""
    client = mqtt.Client(client_id="cloud-dashboard-fixed", clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()

threading.Thread(target=mqtt_thread_func, daemon=True).start()

# --- Streamlit Dashboard UI ---
st.set_page_config(page_title="Fleet Anomaly Dashboard", layout="wide")
st.title("Fleet Anomaly Dashboard (Edge→Fog→Cloud)")

placeholder = st.empty()

while True:
    with placeholder.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Fleet Snapshot")
            ts = fleet_snapshot.get('ts', 'N/A')
            size = fleet_snapshot.get('fleet_size', 0)
            st.write(f"**Timestamp:** {ts} | **Fleet Size:** {size}")

            edges_data = fleet_snapshot.get("edges", [])
            if edges_data:
                df_status = pd.DataFrame(edges_data)
                
                # Add latest anomaly score to the status dataframe
                df_status['anom_score'] = df_status['edge_id'].map(lambda id: latest_reports.get(id, {}).get('score', 0))
                
                st.write("Node Status & Anomaly Score")
                st.dataframe(df_status[['edge_id', 'status', 'anom_score']], use_container_width=True)

                st.write("Anomaly Scores (Alive Nodes)")
                df_display = df_status[df_status['status'] == 'alive']
                if not df_display.empty:
                    # This will now work correctly as 'anom_score' exists
                    st.bar_chart(df_display.set_index("edge_id")['anom_score'])
                else:
                    st.info("No nodes are currently reporting as 'alive'.")
            else:
                st.info("Waiting for fleet snapshot from the fog layer...")

        with col2:
            st.subheader("Live Report Stream")
            if reports_stream:
                df_reports = pd.DataFrame(list(reports_stream))
                st.dataframe(df_reports, use_container_width=True, height=500)
            else:
                st.info("Waiting for reports from edge nodes...")
        
    time.sleep(1)

