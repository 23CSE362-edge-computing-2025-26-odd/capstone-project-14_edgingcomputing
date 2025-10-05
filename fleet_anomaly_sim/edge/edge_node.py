#!/usr/bin/env python3
import os, json, time, random, math, collections
from datetime import datetime

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

# Optional SciPy for hierarchical clustering (required for fleet model)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =========================
# Configuration / Defaults
# =========================
EDGE_ID = os.getenv("EDGE_ID", "edge-1")

# MQTT
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Simulation / Local analytics
ANOMALY_RATE = float(os.getenv("ANOMALY_RATE", "0.01"))  # probability to inject anomaly into a metric
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))        # sliding window for local stats
Z_THRESH = float(os.getenv("Z_THRESH", "2.5"))           # z-score threshold per metric

# Fleet (unsupervised clustering) analytics
FLEET_ANALYSIS_PERIOD_SEC = float(os.getenv("FLEET_PERIOD", "5"))  # how often to run fleet clustering
FLEET_LINKAGE_METHOD = os.getenv("FLEET_METHOD", "single")         # single/complete/average/ward etc.
FLEET_THRCC = float(os.getenv("FLEET_THRCC", "0.2"))               # cut ratio (thrcc * max_distance)
FLEET_ANOMALY_THRESHOLD = float(os.getenv("FLEET_ANOM_THRESHOLD", str(2.0/3.0)))  # same as your code

# Topics
TOPIC_HEARTBEAT = f"edge/{EDGE_ID}/heartbeat"
TOPIC_METRICS_SELF = f"edge/{EDGE_ID}/metrics"
TOPIC_METRICS_FLEET = "edge/+/metrics"  # subscribe to every edge's metrics
TOPIC_ACTUATORS = f"fog/actuators/{EDGE_ID}"
TOPIC_REPORTS = "cloud/reports"

# ================
# MQTT Client init
# ================
client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)

# -------------
# Edge buffers
# -------------
sensor_buffer = collections.deque(maxlen=WINDOW_SIZE)
metrics = ["temperature", "humidity", "pressure", "electrical", "vibration"]

# Latest fleet state (last known vibration per edge_id)
fleet_latest_vibration = {}  # { edge_id: float }


# ====================
# Simulation of sensors
# ====================
def simulate_sensors(t: float) -> dict:
    """Generates simulated sensor data with occasional anomalies."""
    # Base normal signals
    temp = 40 + 5*math.sin(t/30.0) + random.gauss(0, 0.2)
    humidity = 55 + 10*math.sin(t/45.0) + random.gauss(0, 0.5)
    pressure = 1.2 + 0.05*math.sin(t/25.0) + random.gauss(0, 0.01)
    electrical = 10 + 0.8*math.sin(t/5.0) + random.gauss(0, 0.05)
    vibration = 0.6 + 0.15*math.sin(t/7.0) + random.gauss(0, 0.01)

    # Optional random anomalies
    if random.random() < ANOMALY_RATE:
        kind = random.choice(metrics)
        if kind == "temperature":
            temp += random.uniform(8, 15)
        elif kind == "humidity":
            humidity += random.uniform(20, 35)
        elif kind == "pressure":
            pressure += random.uniform(0.3, 0.7)
        elif kind == "electrical":
            electrical += random.uniform(3, 6)
        elif kind == "vibration":
            vibration += random.uniform(0.5, 1.0)
        print(f"[{EDGE_ID}] Injected '{kind}' anomaly.")

    return {
        "temperature": round(temp, 3),
        "humidity": round(humidity, 3),
        "pressure": round(pressure, 3),
        "electrical": round(electrical, 3),
        "vibration": round(vibration, 3),
    }


# ===============================================
# Local analysis (z-score) and Cloud report publish
# ===============================================
def compute_local_anomaly(df: pd.DataFrame, current_reading: dict) -> dict:
    """
    Computes per-metric z-score anomaly against the local sliding window.
    Returns a summary dict.
    """
    anom_count = 0
    per_metric = {}

    for m in metrics:
        mu = df[m].mean()
        sigma = df[m].std(ddof=0) or 1e-6
        z = (current_reading[m] - mu) / sigma
        is_anom = abs(z) > Z_THRESH
        if is_anom:
            anom_count += 1
        per_metric[m] = {
            "mean": float(mu),
            "std": float(sigma),
            "z": float(z),
            "is_anomaly": bool(is_anom),
        }

    score = anom_count / len(metrics)  # fraction of metrics flagged
    decision = "ALERT" if score > 0 else "OK"
    return {
        "decision": decision,
        "score": float(round(score, 3)),
        "per_metric": per_metric,
    }


# ============================================
# Fleet clustering (Hendrickx et al., 2020-ish)
# ============================================
def run_fleet_clustering(fleet_vibration: dict) -> dict:
    """
    Runs hierarchical clustering on the fleet's latest vibration readings.
    - Min-max normalize
    - Pairwise absolute distance
    - SciPy linkage + fcluster with cut_distance = FLEET_THRCC * max_distance
    - Anomaly score: 1 - (cluster_size / N)
    """
    # Need at least 2 edges to form clusters
    if len(fleet_vibration) < 2:
        return {
            "enabled": False,
            "reason": "insufficient_fleet",
            "fleet_size": len(fleet_vibration),
        }

    # Prepare arrays
    edge_ids = sorted(fleet_vibration.keys())
    values = np.array([fleet_vibration[eid] for eid in edge_ids], dtype=float)

    # Min-max normalization
    vmin, vmax = float(values.min()), float(values.max())
    if vmax - vmin == 0:
        norm = np.zeros_like(values)
    else:
        norm = (values - vmin) / (vmax - vmin)

    # Pairwise |Δ| distances
    n = len(norm)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            d = abs(norm[i] - norm[j])
            D[i, j] = D[j, i] = d

    # Convert to condensed vector
    condensed = squareform(D, checks=False)

    # Linkage & cluster cut
    Z = linkage(condensed, method=FLEET_LINKAGE_METHOD)
    max_dist = float(np.max(Z[:, 2])) if Z.size > 0 else 0.0
    cut_distance = FLEET_THRCC * max_dist if max_dist > 0 else 0.0
    clusters = fcluster(Z, t=cut_distance, criterion="distance") if max_dist > 0 else np.ones(n, dtype=int)

    # Anomaly scores by cluster sizes
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}

    scores = {}
    for i, eid in enumerate(edge_ids):
        cid = int(clusters[i])
        size = cluster_sizes[cid]
        scores[eid] = float(1.0 - (size / n))

    # Determine anomalies per threshold
    anomalous = [eid for eid in edge_ids if scores[eid] > FLEET_ANOMALY_THRESHOLD]
    healthy = [eid for eid in edge_ids if scores[eid] <= FLEET_ANOMALY_THRESHOLD]

    return {
        "enabled": True,
        "fleet_size": n,
        "edge_ids": edge_ids,
        "clusters": {eid: int(clusters[i]) for i, eid in enumerate(edge_ids)},
        "cluster_sizes": cluster_sizes,
        "scores": scores,
        "anomalous_edges": anomalous,
        "healthy_edges": healthy,
        "params": {
            "linkage_method": FLEET_LINKAGE_METHOD,
            "thrcc": FLEET_THRCC,
            "cut_distance": float(cut_distance),
            "max_distance": float(max_dist),
            "threshold": float(FLEET_ANOMALY_THRESHOLD),
        },
    }


# ===========================
# Reporting & MQTT publishing
# ===========================
def publish_heartbeat():
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat() + "Z", "status": "alive"}
    client.publish(TOPIC_HEARTBEAT, json.dumps(msg), qos=0, retain=False)


def publish_metrics(current_reading: dict):
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


def compute_and_publish_report(last_fleet_result: dict, current_reading: dict):
    """
    Combines local anomaly and latest fleet clustering snapshot (if available),
    and publishes a single report to the cloud.
    """
    if len(sensor_buffer) < WINDOW_SIZE:
        # Defer local analysis until we have enough history
        return

    df = pd.DataFrame(list(sensor_buffer))
    local = compute_local_anomaly(df, current_reading)

    # Determine this edge's fleet score/decision if clustering is enabled
    fleet_summary = None
    fleet_decision = "NA"
    fleet_score_self = None
    if last_fleet_result and last_fleet_result.get("enabled"):
        scores = last_fleet_result.get("scores", {})
        fleet_score_self = scores.get(EDGE_ID)
        if fleet_score_self is not None:
            fleet_decision = "ALERT" if fleet_score_self > FLEET_ANOMALY_THRESHOLD else "OK"

        # Trim heavy internals before sending
        fleet_summary = {
            "enabled": True,
            "fleet_size": last_fleet_result.get("fleet_size"),
            "params": last_fleet_result.get("params"),
            "anomalous_edges": last_fleet_result.get("anomalous_edges"),
            "healthy_edges": last_fleet_result.get("healthy_edges"),
            "clusters": last_fleet_result.get("clusters"),
            "score_self": fleet_score_self,
        }
    else:
        fleet_summary = {"enabled": False}

    # Overall decision: raise ALERT if either local or fleet triggers
    overall = "ALERT" if (local["decision"] == "ALERT" or fleet_decision == "ALERT") else "OK"

    report = {
        "edge_id": EDGE_ID,
        "ts": datetime.utcnow().isoformat() + "Z",
        "overall_decision": overall,
        "local": local,
        "fleet": fleet_summary,
        "metrics": current_reading,
    }
    client.publish(TOPIC_REPORTS, json.dumps(report), qos=0)


# ============================
# MQTT callbacks / subscriptions
# ============================
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


# ==========
# Main loop
# ==========
def main():
    print(f"Starting Edge Node '{EDGE_ID}'")
    print(f"Publishing to MQTT at {MQTT_HOST}:{MQTT_PORT}")

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    last_fleet_run = 0.0
    last_fleet_result = None

    try:
        while True:
            # 1) Generate / collect current reading
            reading = simulate_sensors(time.time())
            sensor_buffer.append(reading)

            # 2) Publish metrics so others can build the fleet view
            publish_metrics(reading)

            # 3) Maybe run fleet clustering (periodic)
            now = time.time()
            if (now - last_fleet_run) >= FLEET_ANALYSIS_PERIOD_SEC:
                last_fleet_run = now
                # Ensure we have our own latest vibration included
                fleet_latest_vibration[EDGE_ID] = float(reading["vibration"])
                last_fleet_result = run_fleet_clustering(fleet_latest_vibration)

            # 4) Compute local anomaly + combine with fleet snapshot → publish
            compute_and_publish_report(last_fleet_result, reading)

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

