# fog_node.py  (ME-FEEL fog aggregator - layer-wise averaging)
import os, json, time, threading
from datetime import datetime
from collections import defaultdict

import paho.mqtt.client as mqtt
import numpy as np

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
STATUS_PUBLISH_INTERVAL = int(os.getenv("STATUS_PUBLISH_INTERVAL", "5"))

client = mqtt.Client(client_id="fog-controller", clean_session=True)

# Track heartbeats & statuses (existing behavior)
last_heartbeats = {}
edge_statuses = {}

# ME-FEEL state
# collected_updates[round][layer_name] = [array_from_edge1, array_from_edge2, ...]
collected_updates = defaultdict(lambda: defaultdict(list))
# track contributors per round
contributors = defaultdict(set)
global_model = {}  # layer_name -> numpy array
current_round = 0
AGGREGATION_WAIT = int(os.getenv("AGGREGATION_WAIT", "6"))  # seconds to wait for updates each round

def on_connect(client, userdata, flags, rc):
    print("[fog] Connected to MQTT Broker.")
    client.subscribe("edge/+/heartbeat")
    client.subscribe("edge/+/model_update")

def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
    except:
        payload = None

    if topic.endswith("/heartbeat"):
        parts = topic.split('/')
        edge_id = parts[1]
        last_heartbeats[edge_id] = time.time()
        if edge_statuses.get(edge_id) != "alive":
            edge_statuses[edge_id] = "alive"
            print(f"[fog] Edge '{edge_id}' now alive")

    elif topic.endswith("/model_update"):
        # payload: { edge_id, round, exit_depth, layers: {layer_name: [..] } }
        if not payload:
            return
        edge_id = payload.get("edge_id")
        r = int(payload.get("round", current_round))
        layers = payload.get("layers", {})
        if not layers:
            return
        # store arrays in collected_updates[r][layer_name]
        for lname, arr in layers.items():
            collected_updates[r][lname].append(np.array(arr, dtype=np.float32))
        contributors[r].add(edge_id)
        print(f"[fog] Received model update from {edge_id} for round {r}, layers: {list(layers.keys())}")

# Aggregation thread
def aggregator_loop():
    global current_round, global_model
    while True:
        # wait window for updates of this round
        r = current_round
        time.sleep(AGGREGATION_WAIT)  # accumulate updates for this round
        updates_for_round = collected_updates.pop(r, {})
        if updates_for_round:
            # layer-wise averaging
            for lname, arrs in updates_for_round.items():
                if arrs:
                    stacked = np.stack(arrs, axis=0)
                    mean_w = np.mean(stacked, axis=0)
                    # update global
                    if lname in global_model:
                        # simple average between existing and new aggregated (weight equally)
                        global_model[lname] = (global_model[lname] + mean_w) / 2.0
                    else:
                        global_model[lname] = mean_w
            # publish aggregated model
            payload = {
                "ts": datetime.utcnow().isoformat()+"Z",
                "round": r,
                "contributors": list(contributors[r]),
                "layers": {k: v.tolist() for k, v in global_model.items()}
            }
            client.publish("fog/global_model", json.dumps(payload), qos=0)
            print(f"[fog] Published aggregated global model for round {r} (contributors: {len(contributors[r])})")
        else:
            # nothing to aggregate this round
            pass
        current_round += 1

# periodic fleet snapshot publisher (existing behavior)
def publish_fleet_status_loop():
    while True:
        time.sleep(STATUS_PUBLISH_INTERVAL)
        edges_list = []
        for edge_id, status in edge_statuses.items():
             edges_list.append({"edge_id": edge_id, "status": status})
        snapshot = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "fleet_size": len(edge_statuses),
            "edges": edges_list
        }
        client.publish("cloud/fleet_snapshot", json.dumps(snapshot), qos=0)

if __name__ == "__main__":
    print(f"[fog] Connecting to MQTT at {MQTT_HOST}:{MQTT_PORT}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    # launch aggregator
    threading.Thread(target=aggregator_loop, daemon=True).start()
    threading.Thread(target=publish_fleet_status_loop, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
