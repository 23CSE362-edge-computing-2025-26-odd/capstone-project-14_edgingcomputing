import os
import json
import time
import threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from collections import deque
import numpy as np

# --- Environment Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MAX_REPORTS = int(os.getenv("MAX_REPORTS", "500"))

# --- In-memory Data Stores ---
reports_stream = deque(maxlen=MAX_REPORTS)
fleet_snapshot = {}
# Stores the most recent PROCESSED report from each edge device
latest_reports = {}

def run_fleet_clustering(data):

    data = np.array(data)
    n_machines = len(data)
    if n_machines == 0:
        return []

    # --- normalization (min-max) ---
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min == 0:
        data_norm = np.zeros_like(data)
    else:
        data_norm = (data - data_min) / (data_max - data_min)

    # --- compute mean distance per point ---
    distances = np.abs(data_norm[:, None] - data_norm)
    mean_distances = np.mean(distances, axis=1)

    # --- anomaly scoring ---
    # Machines far from others get higher mean distance
    scores = (mean_distances - np.min(mean_distances)) / (
        np.max(mean_distances) - np.min(mean_distances) + 1e-9
    )

    # Threshold
    anomaly_threshold = 0.66
    anomalous_machines = np.where(scores > anomaly_threshold)[0].tolist()

    return anomalous_machines

# --- MQTT Client Setup ---
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to MQTT."""
    print("[cloud] Connected to MQTT Broker.")
    client.subscribe("cloud/reports")  # From Edge nodes
    client.subscribe("cloud/fleet_snapshot")  # From Fog

def on_message(client, userdata, msg):
    """
    Callback for when a message is received from MQTT.
    This function is updated to handle the new, richer report format from edge nodes.
    """
    global fleet_snapshot, reports_stream, latest_reports

    if msg.topic == "cloud/reports":
        # Decode the incoming message payload
        data = json.loads(msg.payload.decode())

        processed_report = {
            "edge_id": data.get("edge_id", "N/A"),
            "ts": data.get("ts", "N/A"),
            "n_machines": data.get("n_machines", 0),
            "metrics": data.get("metrics", []),
            "anomalous_machines": data.get("anomalous_machines", []),
        }

        # Append the processed report to our live stream data
        reports_stream.appendleft(processed_report)
        # Update the latest report for this specific edge_id for the status view
        if 'edge_id' in data:
            latest_reports[data['edge_id']] = processed_report

    elif msg.topic == "cloud/fleet_snapshot":
        # Load the fleet-wide snapshot from the fog layer
        data = json.loads(msg.payload.decode())
        fleet_snapshot = {
            "ts": data.get("ts", "N/A"),
            "fleet_size": data.get("fleet_size", 0),
            "edges": data.get("edges", []),
        }


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
            st.write(f"**Timestamp:** {ts} | **Edge nodes:** {size}")

            edges_data = fleet_snapshot.get("edges", [])
            for i in range(len(edges_data)):
                edges_data[i]['n_machines'] = latest_reports.get(edges_data[i]['edge_id'], {}).get('n_machines', 0)
                if edges_data[i]['status'] != 'alive':
                    processed_report = latest_reports[edges_data[i]['edge_id']]
                    processed_report['anomalous_machines'] = run_fleet_clustering(processed_report.get('metrics', []))
                    latest_reports[edges_data[i]['edge_id']] = processed_report
            if edges_data:
                # Base dataframe with node status (alive/dead)
                df_status = pd.DataFrame(edges_data)

                st.write("Node Status:")
                st.dataframe(df_status, use_container_width=True)

                # --- NEW SECTION: Display all anomalous machines ---
                st.write("Anomalous Machines per Edge Node:")
                anomalous_list = latest_reports
                df_anom = pd.DataFrame.from_dict(anomalous_list).T
                df_anom.drop(columns=['ts', 'metrics','n_machines'], inplace=True, errors='ignore')
                st.dataframe(df_anom, use_container_width=True)

            else:
                st.info("Waiting for fleet snapshot from the fog layer...")

        with col2:
            st.subheader("Live Report Stream")
            if reports_stream:
                # The DataFrame will now show the flattened report, including all metrics,
                # n_machines, etc., making the stream much more informative.
                df_reports = pd.DataFrame(list(reports_stream))
                cols_to_drop=['score','n_machines','anomalous_machines']
                df_reports = df_reports.drop(columns=cols_to_drop, errors='ignore')
                st.dataframe(df_reports, use_container_width=True, height=500)
            else:
                st.info("Waiting for reports from edge nodes...")

    time.sleep(1)

