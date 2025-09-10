
import os, json, time, threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from collections import deque

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

reports = deque(maxlen=500)
fleet_snapshot = {"edges": [], "threshold": None, "fleet_size": 0, "ts": None}

def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("cloud/reports")
    client.subscribe("cloud/fleet_snapshot")

def on_message(client, userdata, msg):
    global fleet_snapshot
    if msg.topic == "cloud/reports":
        data = json.loads(msg.payload.decode())
        reports.appendleft(data)
    elif msg.topic == "cloud/fleet_snapshot":
        fleet_snapshot = json.loads(msg.payload.decode())

client = mqtt.Client(client_id="cloud-dashboard", clean_session=True)
client.on_connect = on_connect
client.on_message = on_message

def mqtt_thread():
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()

threading.Thread(target=mqtt_thread, daemon=True).start()

st.set_page_config(page_title="Fleet Anomaly Dashboard", layout="wide")
st.title("Fleet Anomaly Dashboard (Edge→Fog→Cloud)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Fleet Snapshot")
    st.write(f"Fleet size: {fleet_snapshot.get('fleet_size',0)} | Z-threshold: {fleet_snapshot.get('threshold')} | ts: {fleet_snapshot.get('ts')}")
    df_edges = pd.DataFrame(fleet_snapshot.get("edges", []))
    if not df_edges.empty:
        st.bar_chart(df_edges.set_index("edge_id"))

with col2:
    st.subheader("Recent Reports")
    if reports:
        df = pd.DataFrame(list(reports))
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("Waiting for reports...")

st.caption("Use docker-compose to run MQTT, Edge nodes, Fog, and this Cloud dashboard.")
