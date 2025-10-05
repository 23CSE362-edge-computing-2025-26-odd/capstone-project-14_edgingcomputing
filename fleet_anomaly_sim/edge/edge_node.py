import os, json, time, random, math, collections
from datetime import datetime
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path

EDGE_ID = os.getenv("EDGE_ID", "edge")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "30"))  # for local anomaly

data_path = Path(__file__).resolve().parent / "Data" / f"simulated_vibration_{EDGE_ID}.csv"

TOPIC_HEARTBEAT = f"edge/{EDGE_ID}/heartbeat"
TOPIC_METRICS_SELF = f"edge/{EDGE_ID}/metrics"
TOPIC_METRICS_FLEET = "edge/+/metrics"  # subscribe to every edge's metrics
TOPIC_ACTUATORS = f"fog/actuators/{EDGE_ID}"
TOPIC_REPORTS = "cloud/reports"

# MQTT Client init
client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)

# Latest fleet state (last known vibration per edge_id)
fleet_latest_vibration = {}  # { edge_id: float }
sensor_buffer = []

# Fleet clustering (Hendrickx et al., 2020-ish)
def run_fleet_clustering(data):

    n_machines = len(data)
    method = "single"
    thrcc = 0.2

   # normalization (min-max)
    data_min, data_max = np.min(data), np.max(data)
    if data_max - data_min == 0:
        data_normalized = np.zeros_like(data)
    else:
        data_normalized = (data - data_min) / (data_max - data_min)

    # pairwise distance matrix
    def calculate_distances(values):
        n = len(values)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(values[i] - values[j])
                distances[i, j] = distances[j, i] = dist
        return distances

    distance_matrix = calculate_distances(data_normalized)

    # convert to condensed form for scipy
    condensed_dist = squareform(distance_matrix)

    # hierarchical clustering
    Z = linkage(condensed_dist, method=method)

    # cut the dendrogram
    max_distance = np.max(Z[:, 2])
    cut_distance = thrcc * max_distance
    clusters = fcluster(Z, t=cut_distance, criterion='distance')

    # anomaly_score(machine) = 1 - (cluster_size / total_machines)
    anomaly_scores = np.zeros(n_machines)
    for i in range(n_machines):
        cluster_id = clusters[i]
        cluster_size = np.sum(clusters == cluster_id)
        anomaly_scores[i] = 1 - (cluster_size / n_machines)

    # thresholding
    anomaly_threshold = 2/3
    anomalous_machines = np.where(anomaly_scores > anomaly_threshold)[0].tolist()
    healthy_machines = np.where(anomaly_scores <= anomaly_threshold)[0].tolist()
    anomaly_scores = anomaly_scores.tolist()
    
    result = {
        "anomaly_scores": anomaly_scores,
        "healthy_machines": healthy_machines,
        "anomalous_machines": anomalous_machines
    }
    return result


# Reporting & MQTT publishing
def publish_heartbeat():
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat() + "Z", "status": "alive"}
    client.publish(TOPIC_HEARTBEAT, json.dumps(msg), qos=0, retain=False)


def publish_metrics(current_reading):
    """
    Publish the current metrics for this edge so other edges (and the fog) can see the 'fleet'.
    Topic: edge/{EDGE_ID}/metrics
    """
    payload = {
        "edge_id": EDGE_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "metrics": current_reading,
    }
    client.publish(TOPIC_METRICS_SELF, json.dumps(payload), qos=0, retain=False)


def publish_report(last_fleet_result, current_reading, n_machines):
    """
    Combines local anomaly and latest fleet clustering snapshot (if available),
    and publishes a single report to the cloud.
    """

    report = {
        "edge_id": EDGE_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "anomalous_machines": last_fleet_result.get("anomalous_machines", []),
        "metrics": current_reading,
        "n_machines": n_machines
    }
    client.publish(TOPIC_REPORTS, json.dumps(report), qos=0)


# MQTT callbacks / subscriptions
def on_connect(client, userdata, flags, rc):
    print(f"[{EDGE_ID}] Connected to MQTT with result code {rc}")
    # Subscribe to actuators for this edge
    client.subscribe(TOPIC_ACTUATORS)
    # Subscribe to all edges' metrics to build fleet map
    client.subscribe(TOPIC_METRICS_FLEET)


def on_message(client, userdata, msg):
    """Callback for receiving actuator commands and fleet metrics."""
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        payload = {"raw": msg.payload.decode()}

    # Actuator commands from fog
    if topic.startswith(f"fog/actuators/{EDGE_ID}"):
        print(f"[{EDGE_ID}] ACTUATOR CMD RECEIVED: {payload}")
        return

    # Fleet metrics collection: topic pattern edge/{id}/metrics
    if topic.startswith("edge/") and topic.endswith("/metrics"):
        edge_id = payload.get("edge_id")
        metr = payload.get("metrics", {})
        if edge_id and ("vibration" in metr):
            fleet_latest_vibration[edge_id] = float(metr["vibration"])


def main():
    print(f"Starting Edge Node '{EDGE_ID}'")
    print(f"Publishing to MQTT at {MQTT_HOST}:{MQTT_PORT}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    last_fleet_run = 0.0
    last_fleet_result = None

    df = pd.read_csv(data_path)
    reading = df["vibration"].to_list()
    data = df["vibration"].values

    try:
        while True:
            # 1) Generate / collect current reading
            sensor_buffer.append(reading)

            # 2) Publish metrics
            publish_metrics(reading)

            # 3) Run fleet clustering (periodic)
            now = time.time()
            if (now - last_fleet_run) >= 5:
                last_fleet_run = now
                # Ensure we have our own latest vibration included
                last_fleet_result = run_fleet_clustering(data)
                # 4) Compute local anomaly + combine with fleet snapshot → publish
                publish_report(last_fleet_result,reading,len(reading))
                # 5) Heartbeat
                publish_heartbeat()

            time.sleep(1.0)
    except KeyboardInterrupt:
        print(f"[{EDGE_ID}] Shutting down...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
