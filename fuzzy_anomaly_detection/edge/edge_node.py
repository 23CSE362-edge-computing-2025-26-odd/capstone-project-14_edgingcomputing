import os
import json
import time
from datetime import datetime
from pathlib import Path
import random
import pandas as pd
import paho.mqtt.client as mqtt

# --- Configuration from Environment ---
EDGE_ID = os.getenv("EDGE_ID", "edge_default")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DATA_PATH = Path(__file__).resolve().parent / "Data" / f"simulated_vibration_{EDGE_ID}.csv"
PUBLISH_INTERVAL = 5 # seconds

# --- MQTT Client Setup ---
client = mqtt.Client(client_id=f"edge-publisher-{EDGE_ID}")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[{EDGE_ID}] Connected to MQTT Broker.")

client.on_connect = on_connect

def main():
    print(f"[{EDGE_ID}] Starting Edge Data Forwarder.")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    # Load simulated data from CSV
    try:
        df = pd.read_csv(DATA_PATH)
        data = df.iloc[:, 0].values.tolist()
    except FileNotFoundError:
        print(f"[{EDGE_ID}] Data file not found at {DATA_PATH}. Using random data.")
        data = [random.uniform(0.1, 0.5) for _ in range(100)]
    
    # Main loop to publish data periodically
    try:
        while True:
            payload = {
                "edge_id": EDGE_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "metrics": data
            }
            client.publish(f"edge/{EDGE_ID}/data", json.dumps(payload))
            print(f"[{EDGE_ID}] Published metrics data.")
            time.sleep(PUBLISH_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"[{EDGE_ID}] Shutting down.")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()