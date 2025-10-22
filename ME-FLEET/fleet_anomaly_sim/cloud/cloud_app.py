import os, json, time, threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from collections import deque, defaultdict

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# -------------------- STATE --------------------
reports = deque(maxlen=500)
fleet_snapshot = {"edges": [], "threshold": None, "fleet_size": 0, "ts": None}
fl_progress = deque(maxlen=100)  # global model updates
edge_contrib = []  # list of dicts: {"edge_id": X, "round": R, "exit_depth": D}

# -------------------- MQTT HANDLERS --------------------
def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("cloud/reports")
    client.subscribe("cloud/fleet_snapshot")
    client.subscribe("fog/global_model")
    client.subscribe("edge/+/model_update")

def on_message(client, userdata, msg):
    global fleet_snapshot, edge_contrib
    if msg.topic == "cloud/reports":
        data = json.loads(msg.payload.decode())
        reports.appendleft(data)

    elif msg.topic == "cloud/fleet_snapshot":
        fleet_snapshot = json.loads(msg.payload.decode())

    elif msg.topic == "fog/global_model":
        data = json.loads(msg.payload.decode())
        round_id = data.get("round")
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        fl_progress.appendleft({
            "round": round_id,
            "ts": ts,
            "num_layers": len(data.get("weights_b64", ""))
        })

    elif "model_update" in msg.topic:
        data = json.loads(msg.payload.decode())
        edge_id = data.get("edge_id")
        round_id = data.get("round")
        exit_depth = data.get("exit_depth", 1)
        if edge_id and round_id is not None:
            edge_contrib.append({
                "edge_id": edge_id,
                "round": round_id,
                "exit_depth": exit_depth
            })

client = mqtt.Client(client_id="cloud-dashboard", clean_session=True)
client.on_connect = on_connect
client.on_message = on_message

def mqtt_thread():
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Fleet Anomaly Dashboard", layout="wide")
st.title("Fleet Anomaly Dashboard (Edge→Fog→Cloud)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fleet Snapshot")
    st.write(
        f"Fleet size: {fleet_snapshot.get('fleet_size',0)} | "
        f"Z-threshold: {fleet_snapshot.get('threshold')} | "
        f"ts: {fleet_snapshot.get('ts')}"
    )
    df_edges = pd.DataFrame(fleet_snapshot.get("edges", []))
    if not df_edges.empty:
        st.bar_chart(df_edges.set_index("edge_id"))
    else:
        st.info("Waiting for fleet data...")

with col2:
    st.subheader("Recent Reports")
    if reports:
        df = pd.DataFrame(list(reports))
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("Waiting for reports...")

# -------------------- Federated Learning Progress --------------------
st.subheader("Federated Learning Progress (ME-FEEL)")
if fl_progress:
    df_fl = pd.DataFrame(list(fl_progress))
    st.line_chart(df_fl.set_index("round")["num_layers"])
    st.dataframe(df_fl, use_container_width=True, height=200)
else:
    st.info("Waiting for global model updates from fog...")

# -------------------- Per-Edge Contributions --------------------
st.subheader("Per-Edge Training Contributions (by Exit Depth)")
if edge_contrib:
    df_contrib = pd.DataFrame(edge_contrib)
    # Count contributions grouped by edge and exit depth
    df_grouped = df_contrib.groupby(["edge_id","exit_depth"]).size().reset_index(name="updates")

    st.dataframe(df_grouped, use_container_width=True, height=200)

    # Pivot for stacked bar chart
    df_pivot = df_grouped.pivot(index="edge_id", columns="exit_depth", values="updates").fillna(0)
    st.bar_chart(df_pivot)
else:
    st.info("Waiting for edge model updates...")

st.caption("Use docker-compose to run MQTT, Edge nodes, Fog, and this Cloud dashboard.")
