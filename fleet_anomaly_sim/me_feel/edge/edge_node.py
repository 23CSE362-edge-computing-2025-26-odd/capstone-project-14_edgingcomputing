# edge_node.py  (ME-FEEL edge implementation)
import os, json, time, random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

import torch
import torch.nn as nn
import torch.optim as optim

from model import MultiExitResNet  # your uploaded model.py

# --- config from env ---
EDGE_ID = os.getenv("EDGE_ID", "edge1")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DATA_PATH = Path(__file__).resolve().parent / "Data" / f"simulated_vibration_{EDGE_ID}.csv"
PUBLISH_INTERVAL = float(os.getenv("PUBLISH_INTERVAL", "1.0"))
EXIT_CAPABILITY = int(os.getenv("EXIT_CAPABILITY", "2"))  # 1..num_exits, simulate device heterogeneity
NUM_EXITS = int(os.getenv("NUM_EXITS", "3"))
LOCAL_TRAIN_STEPS = int(os.getenv("LOCAL_TRAIN_STEPS", "3"))
ROUND_SLEEP = float(os.getenv("ROUND_SLEEP", "5.0"))

# MQTT topics
TOPIC_HEARTBEAT = f"edge/{EDGE_ID}/heartbeat"
TOPIC_METRICS_SELF = f"edge/{EDGE_ID}/metrics"
TOPIC_REPORTS = "cloud/reports"
TOPIC_MODEL_UPDATE = f"edge/{EDGE_ID}/model_update"
TOPIC_METRICS_FLEET = "edge/+/metrics"

client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)

# --- model setup ---
device = torch.device("cpu")
model = MultiExitResNet(input_dim=1 if True else 5, hidden_dim=32, num_exits=NUM_EXITS, num_classes=3)
model.to(device)

# helpers for serialization
def tensor_to_list(t):
    return t.detach().cpu().numpy().tolist()

def state_dict_to_json(state_dict, layer_names):
    # only include layers in layer_names
    out = {}
    for k, v in state_dict.items():
        if any(k.startswith(name) for name in layer_names):
            out[k] = tensor_to_list(v)
    return out

def json_to_state_dict(jdict):
    sd = {}
    for k, v in jdict.items():
        sd[k] = torch.tensor(np.array(v), dtype=torch.float32)
    return sd

# small local unsupervised "train" that only updates params for chosen exit
def local_update_on_exit(model, data_window, exit_idx, steps=3, lr=1e-3):
    # We'll create a simple self-supervised target:
    # target = normalized mean of window (scalar); model outputs logits -> we minimize MSE
    model.train()
    params_to_optimize = []
    # choose parameters belonging to the exit layers and shared layers
    for name, p in model.named_parameters():
        # simple rule: include fc1 and exitX layers for exit X
        if name.startswith("fc1") or name.startswith(f"exit{exit_idx}") or name.startswith("fc2") and exit_idx>=2 or name.startswith("fc3") and exit_idx>=3:
            params_to_optimize.append(p)
    if not params_to_optimize:
        return {}
    optimizer = optim.Adam(params_to_optimize, lr=lr)
    criterion = nn.MSELoss()

    # prepare input: average over many samples -> shape (B, input_dim). edge data_window is list of floats.
    x = torch.tensor(np.array(data_window).reshape(-1,1), dtype=torch.float32)
    # pseudo-target: normalized mean vector repeated
    mean_val = float(np.mean(data_window))
    target = torch.full((x.shape[0], 3), fill_value=mean_val, dtype=torch.float32)  # 3 classes dimension (not used meaningfully)

    for _ in range(steps):
        optimizer.zero_grad()
        outs = model(x, exit_idx=None)  # all exits
        # pick the output corresponding to the chosen exit index
        out = outs[exit_idx-1]  # index guard ensured upstream
        # expand/regress to target shape
        # Use softmax logits -> convert to "prob" and try to match target mean
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    # collect updated layer weights to send to fog
    updated_layers = {}
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in ["fc1","exit1","fc2","exit2","fc3","exit3"]):
            # include only layers up to the chosen exit
            # mapping exit -> max layer prefix that belongs to it
            if exit_idx == 1 and (name.startswith("fc1") or name.startswith("exit1")):
                updated_layers[name] = tensor_to_list(p)
            elif exit_idx == 2 and (name.startswith("fc1") or name.startswith("exit1") or name.startswith("fc2") or name.startswith("exit2")):
                updated_layers[name] = tensor_to_list(p)
            elif exit_idx >= 3:
                updated_layers[name] = tensor_to_list(p)
    return updated_layers

# publish helpers
def publish_metrics(current_reading):
    payload = {
        "edge_id": EDGE_ID,
        "ts": datetime.utcnow().isoformat()+"Z",
        "metrics": {"vibration": float(np.mean(current_reading))}  # simplified metric
    }
    client.publish(TOPIC_METRICS_SELF, json.dumps(payload), qos=0, retain=False)

def publish_heartbeat():
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat() + "Z", "status": "alive"}
    client.publish(TOPIC_HEARTBEAT, json.dumps(msg), qos=0, retain=False)

def publish_report(anomalous_machines, current_reading, n_machines):
    report = {
        "edge_id": EDGE_ID,
        "ts": datetime.utcnow().isoformat()+"Z",
        "anomalous_machines": anomalous_machines,
        "metrics": current_reading,
        "n_machines": n_machines
    }
    client.publish(TOPIC_REPORTS, json.dumps(report), qos=0)

def publish_model_update(round_idx, exit_depth, updated_layers):
    payload = {
        "edge_id": EDGE_ID,
        "round": round_idx,
        "exit_depth": exit_depth,
        "layers": updated_layers  # dict of layer_name -> list
    }
    client.publish(TOPIC_MODEL_UPDATE, json.dumps(payload), qos=0)

# mqtt callbacks (keep minimal)
def on_connect(client, userdata, flags, rc):
    print(f"[{EDGE_ID}] Connected to MQTT Broker with rc={rc}")
    client.subscribe("fog/actuators/"+EDGE_ID)

def on_message(client, userdata, msg):
    # handle actuators if needed
    try:
        payload = json.loads(msg.payload.decode())
    except:
        payload = {"raw": msg.payload.decode()}
    print(f"[{EDGE_ID}] Received MQTT {msg.topic}: {payload}")

client.on_connect = on_connect
client.on_message = on_message

def main():
    print(f"[{EDGE_ID}] Starting ME-FEEL Edge (capability exit={EXIT_CAPABILITY})")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    # load CSV simulated data
    if not DATA_PATH.exists():
        print(f"[{EDGE_ID}] Data file not found: {DATA_PATH}")
        data = np.random.normal(0,1,size=100).tolist()
    else:
        df = pd.read_csv(DATA_PATH)
        if "vibration" in df.columns:
            data = df["vibration"].values.tolist()
        else:
            data = df.iloc[:,0].values.tolist()

    n_machines = len(data)

    round_idx = 0
    try:
        while True:
            # 1) publish metrics
            current_reading = data  # entire set for this edge (sim)
            publish_metrics(current_reading)

            # 2) pick exit based on capability (simulate dynamic selection)
            exit_idx = min(EXIT_CAPABILITY, NUM_EXITS)
            # 3) do a tiny local unsupervised "train" on this exit
            updated_layers = local_update_on_exit(model, current_reading, exit_idx, steps=LOCAL_TRAIN_STEPS)

            # 4) publish model_update (only if updated_layers non-empty)
            if updated_layers:
                publish_model_update(round_idx, exit_idx, updated_layers)

            # 5) compute a quick fleet-local anomaly list using simple zscore heuristic (edge-local)
            # for simulation: treat values far from median in this edge as anomalous machines
            arr = np.array(current_reading)
            med = np.median(arr); mad = np.median(np.abs(arr - med)) + 1e-9
            z = np.abs((arr - med) / (mad))
            anomalous_idx = np.where(z > 3.0)[0].tolist()

            # 6) publish combined report to cloud
            publish_report(anomalous_idx, current_reading, n_machines)

            # 7) heartbeat
            publish_heartbeat()

            round_idx += 1
            time.sleep(ROUND_SLEEP)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
