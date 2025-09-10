
import os, json, time, random, math
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime

EDGE_ID = os.getenv("EDGE_ID", "edgeX")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ANOMALY_RATE = float(os.getenv("ANOMALY_RATE", "0.01"))  # probability to inject anomaly per message

client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)
client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
client.loop_start()

def simulate_sensors(t):
    # Base normal signals
    temp = 40 + 5*math.sin(t/30.0) + random.gauss(0, 0.2)      # deg C
    humidity = 55 + 10*math.sin(t/45.0) + random.gauss(0, 0.5) # %
    pressure = 1.2 + 0.05*math.sin(t/25.0) + random.gauss(0, 0.01) # bar
    electrical = 10 + 0.8*math.sin(t/5.0) + random.gauss(0, 0.05)  # amps (proxy)
    vibration = 0.6 + 0.15*math.sin(t/7.0) + random.gauss(0, 0.01) # g

    # occasional anomalies
    if random.random() < ANOMALY_RATE:
        kind = random.choice(["temp","humidity","pressure","electrical","vibration"])
        if kind == "temp":
            temp += random.uniform(8, 15)
        elif kind == "humidity":
            humidity += random.uniform(20, 35)
        elif kind == "pressure":
            pressure += random.uniform(0.3, 0.7)
        elif kind == "electrical":
            electrical += random.uniform(3, 6)
        elif kind == "vibration":
            vibration += random.uniform(0.5, 1.0)

    return {
        "temperature": round(temp,3),
        "humidity": round(humidity,3),
        "pressure": round(pressure,3),
        "electrical": round(electrical,3),
        "vibration": round(vibration,3)
    }

def publish_heartbeat():
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z", "status":"alive"}
    client.publish(f"edge/{EDGE_ID}/heartbeat", json.dumps(msg), qos=0, retain=False)

def publish_sensor():
    t = time.time()
    payload = simulate_sensors(t)
    payload.update({"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z"})
    client.publish(f"edge/{EDGE_ID}/sensors", json.dumps(payload), qos=0, retain=False)

def on_message(client, userdata, msg):
    if msg.topic.startswith(f"fog/actuators/{EDGE_ID}"):
        try:
            data = json.loads(msg.payload.decode())
        except:
            data = {"raw": msg.payload.decode()}
        print(f"[{EDGE_ID}] ACTUATOR CMD: {data}")

client.on_message = on_message
client.subscribe(f"fog/actuators/{EDGE_ID}")

if __name__ == "__main__":
    print(f"Starting {EDGE_ID} → MQTT at {MQTT_HOST}:{MQTT_PORT}")
    while True:
        publish_sensor()
        publish_heartbeat()
        time.sleep(1.0)
