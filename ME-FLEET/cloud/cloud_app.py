# cloud_app.py (dashboard + visualization + CI model evolution + per-edge clustering)
import os
import json
import time
import threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# --- Environment Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MAX_REPORTS = int(os.getenv("MAX_REPORTS", "500"))

# --- In-memory Data Stores ---
reports_stream = deque(maxlen=MAX_REPORTS)
fleet_snapshot = {}
latest_reports = {}
global_model = {}  # layer_name -> numpy array
global_model_meta = {}
global_model_history = []  # Store evolution of layer norms per round

# --- clustering helper (keeps original simple clustering) ---
def run_fleet_clustering(data):
    data = np.array(data)
    n_machines = len(data)
    if n_machines == 0:
        return []

    # normalization (min-max)
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min == 0:
        data_norm = np.zeros_like(data)
    else:
        data_norm = (data - data_min) / (data_max - data_min)

    # compute mean distance per point
    distances = np.abs(data_norm[:, None] - data_norm)
    mean_distances = np.mean(distances, axis=1)

    scores = (mean_distances - np.min(mean_distances)) / (np.max(mean_distances) - np.min(mean_distances) + 1e-9)
    anomaly_threshold = 0.66
    anomalous_machines = np.where(scores > anomaly_threshold)[0].tolist()
    return anomalous_machines

# --- MQTT Client Setup ---
def on_connect(client, userdata, flags, rc, properties=None):
    print("[cloud] Connected to MQTT Broker.")
    client.subscribe("cloud/reports")
    client.subscribe("cloud/fleet_snapshot")
    client.subscribe("fog/global_model")

def on_message(client, userdata, msg):
    global fleet_snapshot, reports_stream, latest_reports, global_model, global_model_meta, global_model_history
    try:
        data = json.loads(msg.payload.decode())
    except:
        data = {}

    if msg.topic == "cloud/reports":
        processed_report = {
            "edge_id": data.get("edge_id", "N/A"),
            "ts": data.get("ts", "N/A"),
            "n_machines": data.get("n_machines", 0),
            "metrics": data.get("metrics", []),
            "anomalous_machines": data.get("anomalous_machines", []),
        }
        reports_stream.appendleft(processed_report)
        if 'edge_id' in data:
            latest_reports[data['edge_id']] = processed_report

    elif msg.topic == "cloud/fleet_snapshot":
        fleet_snapshot = {
            "ts": data.get("ts", "N/A"),
            "fleet_size": data.get("fleet_size", 0),
            "edges": data.get("edges", []),
        }

    elif msg.topic == "fog/global_model":
        global_model_meta = {
            "ts": data.get("ts"),
            "round": data.get("round"),
            "contributors": data.get("contributors", [])
        }
        layers = data.get("layers", {})
        global_model = {k: np.array(v, dtype=np.float32) for k, v in layers.items()}

        # Compute and store layer norms for CI evolution plot
        round_id = global_model_meta.get("round", len(global_model_history))
        for k, v in global_model.items():
            norm_val = float(np.linalg.norm(v))
            global_model_history.append({
                "round": round_id,
                "layer": k,
                "norm": norm_val
            })

def mqtt_thread_func():
    client = mqtt.Client(client_id="cloud-dashboard-fixed", clean_session=True)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()

threading.Thread(target=mqtt_thread_func, daemon=True).start()

# --- Streamlit Dashboard UI ---
st.set_page_config(page_title="Fleet Anomaly Dashboard (ME-FEEL)", layout="wide")
st.title("Fleet Anomaly Dashboard (Edge→Fog→Cloud) — ME-FEEL CI Visualization")

placeholder = st.empty()

while True:
    with placeholder.container():
        col1, col2 = st.columns([1, 1])

        # --- LEFT COLUMN ---
        with col1:
            st.subheader("Fleet Snapshot")
            ts = fleet_snapshot.get('ts', 'N/A')
            size = fleet_snapshot.get('fleet_size', 0)
            st.write(f"**Timestamp:** {ts} | **Edge nodes:** {size}")

            edges_data = fleet_snapshot.get("edges", [])
            for i in range(len(edges_data)):
                edges_data[i]['n_machines'] = latest_reports.get(edges_data[i]['edge_id'], {}).get('n_machines', 0)
                if edges_data[i].get('status') != 'alive':
                    processed_report = latest_reports.get(edges_data[i]['edge_id'], {})
                    if processed_report:
                        processed_report['anomalous_machines'] = run_fleet_clustering(processed_report.get('metrics', []))
                        latest_reports[edges_data[i]['edge_id']] = processed_report

            if edges_data:
                df_status = pd.DataFrame(edges_data)
                st.write("Node Status:")
                st.dataframe(df_status, use_container_width=True)

                st.write("Anomalous Machines per Edge Node:")
                anomalous_list = latest_reports
                df_anom = pd.DataFrame.from_dict(anomalous_list).T
                df_anom.drop(columns=['ts', 'metrics','n_machines'], inplace=True, errors='ignore')
                st.dataframe(df_anom, use_container_width=True)
            else:
                st.info("Waiting for fleet snapshot from fog...")

            # --- Global Model Table ---
            st.subheader("Global Model (Fog Aggregated)")
            if global_model:
                st.write(f"Round: {global_model_meta.get('round','N/A')} | Contributors: {len(global_model_meta.get('contributors',[]))}")
                rows = [{"layer": k, "shape": str(v.shape)} for k, v in global_model.items()]
                st.table(pd.DataFrame(rows))
            else:
                st.info("No aggregated model published yet by fog.")

        # --- RIGHT COLUMN ---
        with col2:
            st.subheader("Live Report Stream & Simulation")
            if reports_stream:
                df_reports = pd.DataFrame(list(reports_stream))
                df_reports = df_reports.drop(columns=['score','n_machines','anomalous_machines'], errors='ignore')
                st.dataframe(df_reports, use_container_width=True, height=300)
            else:
                st.info("Waiting for edge reports...")

            st.markdown("### Simulation view (per-edge machine anomaly highlights)")
            sim_cols = st.columns(2)
            i = 0
            for edge_id, rep in latest_reports.items():
                with sim_cols[i % 2]:
                    st.markdown(f"**{edge_id}** — {rep.get('ts','')}")
                    metrics = rep.get("metrics", [])
                    anomalies = rep.get("anomalous_machines", [])
                    if isinstance(metrics, list) and len(metrics) > 0:
                        arr = np.array(metrics)
                        nshow = min(50, len(arr))

                        df_sim = pd.DataFrame({"index": np.arange(nshow), "value": arr[:nshow]})
                        st.line_chart(df_sim.set_index("index")["value"])
                        if anomalies:
                            st.write(f"Anomalous machine indices: {anomalies}")
                        else:
                            st.write("No anomalies detected.")
                    else:
                        st.write("No metrics available.")
                i += 1

            # --- Per-Edge Scatter Plot (Faulty vs Non-Faulty Machines) ---
            st.markdown("### Per-Edge Machine Clusters (Faulty vs Non-Faulty)")

            if latest_reports:
                for edge_id, rep in latest_reports.items():
                    metrics = rep.get("metrics", [])
                    anomalies = set(rep.get("anomalous_machines", []))
                    if isinstance(metrics, list) and len(metrics) > 0:
                        values = np.array(metrics)
                        indices = np.arange(len(values))

                        fig, ax = plt.subplots()
                        for idx, val in zip(indices, values):
                            color = "red" if idx in anomalies else "green"
                            ax.scatter(idx, val, c=color, s=100, alpha=0.7)

                        ax.set_title(f"{edge_id} — Faulty vs Non-Faulty Machines")
                        ax.set_xlabel("Machine Index")
                        ax.set_ylabel("Metric Value")
                        st.pyplot(fig)
                    else:
                        st.info(f"No metrics to display for {edge_id}")
            else:
                st.info("Waiting for edge data...")

            # --- Global Model Evolution Scatter Plot ---
            st.markdown("### CI Model Evolution (Global Aggregation Dynamics)")
            if global_model_history:
                df_hist = pd.DataFrame(global_model_history)
                fig2, ax2 = plt.subplots()
                for layer_name in df_hist["layer"].unique():
                    subset = df_hist[df_hist["layer"] == layer_name]
                    ax2.scatter(subset["round"], subset["norm"], label=layer_name, s=70)
                ax2.set_xlabel("Federated Learning Round")
                ax2.set_ylabel("Layer Weight Norm")
                ax2.set_title("Global Model Evolution — CI Aggregation over Rounds")
                ax2.legend(fontsize=8)
                st.pyplot(fig2)
            else:
                st.info("Waiting for global model updates from fog to show CI evolution...")

    time.sleep(1)
