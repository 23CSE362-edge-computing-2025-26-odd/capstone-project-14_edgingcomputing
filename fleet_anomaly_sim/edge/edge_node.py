import os, json, time, random, math, collections
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime

# --- Environment Configuration ---
EDGE_ID = os.getenv("EDGE_ID", "edge-1")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ANOMALY_RATE = float(os.getenv("ANOMALY_RATE", "0.01"))  # Probability to inject an anomaly
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50")) # Local buffer size for anomaly detection
Z_THRESH = float(os.getenv("Z_THRESH", "2.5")) # Z-score threshold for anomaly detection

# --- MQTT Client Setup ---
client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)
client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
client.loop_start()

# --- Local Data Buffer ---
sensor_buffer = collections.deque(maxlen=WINDOW_SIZE)
metrics = ["temperature", "humidity", "pressure", "electrical", "vibration"]

def simulate_sensors(t):
    """Generates simulated sensor data with occasional anomalies."""
    # Base normal signals
    temp = 40 + 5*math.sin(t/30.0) + random.gauss(0, 0.2)
    humidity = 55 + 10*math.sin(t/45.0) + random.gauss(0, 0.5)
    pressure = 1.2 + 0.05*math.sin(t/25.0) + random.gauss(0, 0.01)
    electrical = 10 + 0.8*math.sin(t/5.0) + random.gauss(0, 0.05)
    vibration = 0.6 + 0.15*math.sin(t/7.0) + random.gauss(0, 0.01)

    # Inject anomalies
    if random.random() < ANOMALY_RATE:
        kind = random.choice(metrics)
        if kind == "temperature": temp += random.uniform(8, 15)
        elif kind == "humidity": humidity += random.uniform(20, 35)
        elif kind == "pressure": pressure += random.uniform(0.3, 0.7)
        elif kind == "electrical": electrical += random.uniform(3, 6)
        elif kind == "vibration": vibration += random.uniform(0.5, 1.0)
        print(f"[{EDGE_ID}] Injected '{kind}' anomaly.")

    return {
        "temperature": round(temp, 3), "humidity": round(humidity, 3),
        "pressure": round(pressure, 3), "electrical": round(electrical, 3),
        "vibration": round(vibration, 3)
    }

def compute_and_publish_report():
    """Analyzes local data and publishes a full report to the cloud."""
    if len(sensor_buffer) < WINDOW_SIZE:
        # Don't analyze until the buffer is full
        return

    df = pd.DataFrame(list(sensor_buffer))
    current_reading = sensor_buffer[-1]
    
    anom_count = 0
    # Calculate z-score for the latest reading against its own history
    for metric in metrics:
        mu = df[metric].mean()
        sigma = df[metric].std(ddof=0) or 1e-6
        z_score = (current_reading[metric] - mu) / sigma
        if abs(z_score) > Z_THRESH:
            anom_count += 1
            
    anom_score = anom_count / len(metrics)
    decision = "ALERT" if anom_score > 0 else "OK"

    # Publish a detailed report directly to the cloud
    report = {
        "edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z",
        "decision": decision, "score": round(float(anom_score), 3),
        "metrics": current_reading
    }
    client.publish("cloud/reports", json.dumps(report), qos=0)


def publish_heartbeat():
    """Publishes a heartbeat message to the fog."""
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z", "status": "alive"}
    client.publish(f"edge/{EDGE_ID}/heartbeat", json.dumps(msg), qos=0, retain=False)

def on_message(client, userdata, msg):
    """Callback for receiving actuator commands from the fog."""
    if msg.topic.startswith(f"fog/actuators/{EDGE_ID}"):
        try:
            data = json.loads(msg.payload.decode())
        except: data = {"raw": msg.payload.decode()}
        print(f"[{EDGE_ID}] ACTUATOR CMD RECEIVED: {data}")

client.on_message = on_message
client.subscribe(f"fog/actuators/{EDGE_ID}")

if __name__ == "__main__":
    print(f"Starting Edge Node '{EDGE_ID}'")
    print(f"Publishing to MQTT at {MQTT_HOST}:{MQTT_PORT}")
    while True:
        # Simulate sensor data and add to local buffer
        payload = simulate_sensors(time.time())
        sensor_buffer.append(payload)
        
        # Perform local analysis and publish a report to the cloud
        compute_and_publish_report()
        
        # Publish heartbeat to the fog
        publish_heartbeat()
        
        time.sleep(1.0)

