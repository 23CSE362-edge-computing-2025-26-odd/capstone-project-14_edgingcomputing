import os, json, time, threading, collections
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
import base64
import msgpack_numpy as m

# -------------------- CONFIG --------------------
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))
Z_THRESH = float(os.getenv("Z_THRESH", "2.5"))
ROUND_DURATION = int(os.getenv("ROUND_DURATION", "20"))  # seconds for one FL round

metrics = ["temperature","humidity","pressure","electrical","vibration"]

# Buffers for anomaly detection
buffers = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW_SIZE))

# Buffers for federated learning
model_updates = {}  # round -> { layer: [arrays] }
current_round = 0
last_round_time = time.time()

client = mqtt.Client(client_id="fog-collector", clean_session=True)

# -------------------- MQTT HANDLERS --------------------
def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("edge/+/sensors")
    client.subscribe("edge/+/heartbeat")
    client.subscribe("edge/+/model_update")

def on_message(client, userdata, msg):
    topic = msg.topic
    if topic.endswith("/sensors"):
        data = json.loads(msg.payload.decode())
        edge_id = data["edge_id"]
        buffers[edge_id].append(data)

    elif topic.endswith("/heartbeat"):
        # heartbeat already tracked in cloud snapshot
        pass

    elif topic.endswith("/model_update"):
        handle_model_update(msg.payload.decode())

client.on_connect = on_connect
client.on_message = on_message

# -------------------- FEDERATED LEARNING --------------------
def handle_model_update(payload):
    global model_updates, current_round
    data = json.loads(payload)
    round_idx = data["round"]
    weights = m.unpackb(base64.b64decode(data["weights_b64"]))

    for k,v in weights.items():
        model_updates.setdefault(round_idx, {}).setdefault(k, []).append(np.array(v))

def aggregate_and_publish():
    global model_updates, current_round, last_round_time
    while True:
        time.sleep(5.0)
        if time.time() - last_round_time >= ROUND_DURATION:
            if current_round in model_updates:
                collected = model_updates[current_round]
                averaged = {}
                for layer, arrs in collected.items():
                    averaged[layer] = np.mean(arrs, axis=0)
                packed = m.packb(averaged)
                b64 = base64.b64encode(packed).decode("ascii")

                payload = {
                    "round": current_round,
                    "weights_b64": b64
                }
                client.publish("fog/global_model", json.dumps(payload), qos=0)
                print(f"[fog] Published global model for round {current_round}")

            current_round += 1
            last_round_time = time.time()

# -------------------- ANOMALY DETECTION --------------------
def compute_anomalies():
    while True:
        time.sleep(2.0)
        latest = []
        for edge, buf in buffers.items():
            if len(buf) == 0:
                continue
            latest.append({"edge_id": edge, **buf[-1]})
        if not latest:
            continue

        df = pd.DataFrame(latest)
        df = df[["edge_id"] + metrics]

        # fleet z-score
        for metric in metrics:
            mu = df[metric].mean()
            sigma = df[metric].std(ddof=0) or 1e-6
            df[f"{metric}_z"] = (df[metric] - mu) / sigma

        df["anom_count"] = (df[[f"{m}_z" for m in metrics]].abs() > Z_THRESH).sum(axis=1)
        df["anom_score"] = df["anom_count"] / len(metrics)

        # Reports & actuator commands
        for _, row in df.iterrows():
            decision = "OK"
            if row["anom_score"] >= 0.2:
                decision = "ALERT"
                cmd = {
                    "edge_id": row["edge_id"],
                    "ts": datetime.utcnow().isoformat()+"Z",
                    "alarm": True,
                    "reason": "anomaly_detected",
                    "score": row["anom_score"]
                }
                client.publish(f"fog/actuators/{row['edge_id']}", json.dumps(cmd), qos=0)

            report = {
                "edge_id": row["edge_id"],
                "ts": datetime.utcnow().isoformat()+"Z",
                "decision": decision,
                "score": round(float(row["anom_score"]),3),
                "metrics": {m: float(row[m]) for m in metrics}
            }
            client.publish("cloud/reports", json.dumps(report), qos=0)

        # Fleet snapshot
        snapshot = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "fleet_size": len(df),
            "threshold": Z_THRESH,
            "edges": df[["edge_id","anom_score"]].to_dict(orient="records")
        }
        client.publish("cloud/fleet_snapshot", json.dumps(snapshot), qos=0)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print(f"[fog] connecting to MQTT {MQTT_HOST}:{MQTT_PORT}")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    # anomaly detection thread
    t1 = threading.Thread(target=compute_anomalies, daemon=True)
    t1.start()

    # federated learning aggregator thread
    t2 = threading.Thread(target=aggregate_and_publish, daemon=True)
    t2.start()

    while True:
        time.sleep(1)
