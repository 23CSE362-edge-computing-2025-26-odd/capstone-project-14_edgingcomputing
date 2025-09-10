
import os, json, time, threading, collections, statistics
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))
Z_THRESH = float(os.getenv("Z_THRESH", "2.5"))

metrics = ["temperature","humidity","pressure","electrical","vibration"]

buffers = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW_SIZE))

client = mqtt.Client(client_id="fog-collector", clean_session=True)

def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("edge/+/sensors")
    client.subscribe("edge/+/heartbeat")

def on_message(client, userdata, msg):
    topic = msg.topic
    if topic.endswith("/sensors"):
        data = json.loads(msg.payload.decode())
        edge_id = data["edge_id"]
        buffers[edge_id].append(data)
    elif topic.endswith("/heartbeat"):
        pass

client.on_connect = on_connect
client.on_message = on_message

def compute_anomalies():
    while True:
        time.sleep(2.0)
        # Build latest dataframe across edges (last sample)
        latest = []
        for edge, buf in buffers.items():
            if len(buf) == 0: 
                continue
            latest.append({ "edge_id": edge, **buf[-1] })
        if not latest:
            continue
        df = pd.DataFrame(latest)
        df = df[["edge_id"] + metrics]
        # fleet z-score per metric
        anomalies = []
        for metric in metrics:
            mu = df[metric].mean()
            sigma = df[metric].std(ddof=0) or 1e-6
            df[f"{metric}_z"] = (df[metric] - mu) / sigma
        df["anom_count"] = (df[[f"{m}_z" for m in metrics]].abs() > Z_THRESH).sum(axis=1)
        df["anom_score"] = df["anom_count"] / len(metrics)
        # Decide and act
        for _, row in df.iterrows():
            decision = "OK"
            if row["anom_score"] >= 0.2:  # at least one metric z>threshold
                decision = "ALERT"
                # command actuator for that edge
                cmd = {"edge_id": row["edge_id"], "ts": datetime.utcnow().isoformat()+"Z", "alarm": True, "reason": "anomaly_detected", "score": row["anom_score"]}
                client.publish(f"fog/actuators/{row['edge_id']}", json.dumps(cmd), qos=0)
            report = {
                "edge_id": row["edge_id"],
                "ts": datetime.utcnow().isoformat()+"Z",
                "decision": decision,
                "score": round(float(row["anom_score"]),3),
                "metrics": {m: float(row[m]) for m in metrics}
            }
            client.publish("cloud/reports", json.dumps(report), qos=0)
        # publish fleet snapshot
        snapshot = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "fleet_size": len(df),
            "threshold": Z_THRESH,
            "edges": df[["edge_id","anom_score"]].to_dict(orient="records")
        }
        client.publish("cloud/fleet_snapshot", json.dumps(snapshot), qos=0)

if __name__ == "__main__":
    print(f"[fog] connecting to MQTT {MQTT_HOST}:{MQTT_PORT}")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    t = threading.Thread(target=compute_anomalies, daemon=True)
    t.start()
    while True:
        time.sleep(1)
