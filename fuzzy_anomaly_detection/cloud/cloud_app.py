import os
import json
import time
import threading
import streamlit as st
import pandas as pd
import paho.mqtt.client as mqtt
from queue import Queue
from fuzzy_engine import get_anomaly_score

# --- Environment Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# --- Using st.session_state to persist data across reruns ---
if 'edge_data' not in st.session_state:
    st.session_state.edge_data = {}

@st.cache_resource
def init_mqtt_client_with_queue():
    """
    Initializes MQTT client and a queue to safely pass messages.
    This function is cached and will only run ONCE.
    """
    q = Queue()
    print("[cloud] Initializing MQTT Client and Message Queue...")
    client = mqtt.Client(client_id="cloud-dashboard-final", clean_session=True)

    def on_connect(client, userdata, flags, rc, properties=None):
        print("[cloud] Connected to MQTT Broker.")
        client.subscribe("edge/+/data")

    def on_message(client, userdata, msg):
        q.put(msg.payload.decode())

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    return q

# --- Initialize the queue using the cached function ---
msg_queue = init_mqtt_client_with_queue()

# --- Streamlit Dashboard UI ---
st.set_page_config(page_title="Fuzzy Anomaly Dashboard", layout="wide")
st.title("Centralized Fuzzy Anomaly Detection Dashboard")

placeholder = st.empty()

while True:
    # --- Process all messages from the queue in the main thread ---
    while not msg_queue.empty():
        try:
            payload_str = msg_queue.get()
            payload = json.loads(payload_str)
            edge_id = payload.get("edge_id")
            metrics = payload.get("metrics", [])
            
            anomaly_score = get_anomaly_score(metrics)
            
            st.session_state.edge_data[edge_id] = {
                "ts": payload.get("ts"),
                "n_machines": len(metrics),
                "anomaly_score": anomaly_score,
                "metrics": metrics
            }
        except Exception as e:
            print(f"[cloud] Error processing message from queue: {e}")
    
    with placeholder.container():
        st.subheader("Live Fleet Health Status")
        
        if not st.session_state.edge_data:
            st.info("Waiting for data from edge nodes...")
        else:
            rows = []
            for edge_id, data in st.session_state.edge_data.items():
                score = data.get('anomaly_score', 0.0)
                health_status = "Critical" if score > 0.6 else "Warning" if score > 0.3 else "Healthy"
                rows.append({
                    "Edge ID": edge_id,
                    "Timestamp": data.get('ts', 'N/A'),
                    "Machines": data.get('n_machines', 0),
                    "Fuzzy Anomaly Score": f"{score:.3f}",
                    "Health Status": health_status
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, width='stretch')
            
            st.subheader("Machine Data Visualization")
            
            if len(st.session_state.edge_data) > 0:
                chart_cols = st.columns(len(st.session_state.edge_data))
                
                for i, (edge_id, data) in enumerate(st.session_state.edge_data.items()):
                    with chart_cols[i]:
                        st.markdown(f"**{edge_id}**")
                        df_metrics = pd.DataFrame({
                            "Machine Index": range(len(data['metrics'])),
                            "Vibration": data['metrics']
                        })
                        st.line_chart(df_metrics.set_index("Machine Index"))

    time.sleep(1)