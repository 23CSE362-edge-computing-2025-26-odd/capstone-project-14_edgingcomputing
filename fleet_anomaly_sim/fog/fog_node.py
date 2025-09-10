import os, json, time, threading
import paho.mqtt.client as mqtt
from datetime import datetime

# --- Environment Configuration ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "10")) # Seconds to wait for heartbeat
STATUS_PUBLISH_INTERVAL = int(os.getenv("STATUS_PUBLISH_INTERVAL", "5")) # Seconds

# --- Data Stores ---
last_heartbeats = {} # {edge_id: timestamp}
edge_statuses = {} # {edge_id: "alive" | "unresponsive"}

# --- MQTT Client Setup ---
client = mqtt.Client(client_id="fog-controller", clean_session=True)

def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to the MQTT broker."""
    print("[fog] Connected to MQTT Broker.")
    # Only subscribe to heartbeat topics
    client.subscribe("edge/+/heartbeat")

def on_message(client, userdata, msg):
    """Callback to process incoming heartbeat messages from edge nodes."""
    topic_parts = msg.topic.split('/')
    edge_id = topic_parts[1]
    
    if topic_parts[-1] == "heartbeat":
        # Record the time of the heartbeat
        last_heartbeats[edge_id] = time.time()
        # If the node was previously unresponsive, mark it as alive
        if edge_statuses.get(edge_id) != "alive":
            edge_statuses[edge_id] = "alive"
            print(f"[fog] Edge node '{edge_id}' is now alive.")

client.on_connect = on_connect
client.on_message = on_message

def check_heartbeats():
    """Periodically checks if edge nodes are still sending heartbeats."""
    while True:
        now = time.time()
        # Iterate over a copy of the items to allow modification during iteration
        for edge_id, last_ts in list(last_heartbeats.items()):
            if now - last_ts > HEARTBEAT_TIMEOUT:
                # If the node has missed a heartbeat, mark it as unresponsive
                if edge_statuses.get(edge_id) != "unresponsive":
                    edge_statuses[edge_id] = "unresponsive"
                    print(f"[fog] Edge node '{edge_id}' has become unresponsive.")
        time.sleep(HEARTBEAT_TIMEOUT / 2)

def publish_fleet_status():
    """Periodically publishes the status of all known edge nodes to the cloud."""
    while True:
        time.sleep(STATUS_PUBLISH_INTERVAL)
        
        edges_list = []
        for edge_id, status in edge_statuses.items():
             edges_list.append({
                 "edge_id": edge_id,
                 "status": status
             })

        snapshot = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "fleet_size": len(edge_statuses),
            "edges": edges_list
        }
        client.publish("cloud/fleet_snapshot", json.dumps(snapshot), qos=0)


if __name__ == "__main__":
    print(f"[fog] Connecting to MQTT at {MQTT_HOST}:{MQTT_PORT}")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    # Start the heartbeat checker in a background thread
    threading.Thread(target=check_heartbeats, daemon=True).start()
    # Start the fleet status publisher in a background thread
    threading.Thread(target=publish_fleet_status, daemon=True).start()
    
    print("[fog] Fog controller started for heartbeat monitoring.")
    while True:
        time.sleep(1)

