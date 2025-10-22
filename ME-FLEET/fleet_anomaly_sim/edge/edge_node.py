import os, json, time, random, math, base64
import numpy as np
import paho.mqtt.client as mqtt
from datetime import datetime
import torch
import torch.nn as nn
import msgpack_numpy as m

from model import MultiExitNet   # your multi-exit model

# -------------------- CONFIG --------------------
EDGE_ID = os.getenv("EDGE_ID", "edgeX")
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
ANOMALY_RATE = float(os.getenv("ANOMALY_RATE", "0.01"))
EXIT_DEPTH = int(os.getenv("EXIT_DEPTH", "1"))  # 1=shallow, 2=mid, 3=deep
TRAIN_INTERVAL = int(os.getenv("TRAIN_INTERVAL", "20"))  # seconds between updates
BATCH_SIZE = 32

# -------------------- INIT MQTT --------------------
client = mqtt.Client(client_id=f"{EDGE_ID}-pub", clean_session=True)

# -------------------- INIT MODEL --------------------
model = MultiExitNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
round_num = 0

# -------------------- SENSOR SIMULATION --------------------
def simulate_sensors(t):
    temp = 40 + 5*math.sin(t/30.0) + random.gauss(0, 0.2)
    humidity = 55 + 10*math.sin(t/45.0) + random.gauss(0, 0.5)
    pressure = 1.2 + 0.05*math.sin(t/25.0) + random.gauss(0, 0.01)
    electrical = 10 + 0.8*math.sin(t/5.0) + random.gauss(0, 0.05)
    vibration = 0.6 + 0.15*math.sin(t/7.0) + random.gauss(0, 0.01)

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

# -------------------- PUBLISH UTILS --------------------
def publish_heartbeat():
    msg = {"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z", "status":"alive"}
    client.publish(f"edge/{EDGE_ID}/heartbeat", json.dumps(msg))

def publish_sensor(sensor):
    payload = sensor.copy()
    payload.update({"edge_id": EDGE_ID, "ts": datetime.utcnow().isoformat()+"Z"})
    client.publish(f"edge/{EDGE_ID}/sensors", json.dumps(payload))

def publish_model_update(state_dict):
    global round_num
    round_num += 1

    if EXIT_DEPTH == 1:
        keys = ["conv1", "exit1"]
    elif EXIT_DEPTH == 2:
        keys = ["conv1","conv2","exit2"]
    else:
        keys = ["conv1","conv2","conv3","exit3"]

    selected = {k: v.cpu().numpy() for k,v in state_dict.items() if any(k.startswith(l) for l in keys)}
    packed = m.packb(selected)
    b64 = base64.b64encode(packed).decode("ascii")

    payload = {
        "edge_id": EDGE_ID,
        "round": round_num,
        "exit_depth": EXIT_DEPTH,
        "layers": keys,
        "weights_b64": b64
    }
    client.publish(f"edge/{EDGE_ID}/model_update", json.dumps(payload))
    print(f"[{EDGE_ID}] Published model update for round {round_num}")

# -------------------- TRAINING --------------------
def local_train(batch_x, batch_y):
    global model, optimizer     
    model.train()
    x = torch.tensor(batch_x).float().unsqueeze(1)  # [B,1,window_len]
    y = torch.tensor(batch_y).long()
    out = model(x, exit_depth=EXIT_DEPTH)
    loss = criterion(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model.state_dict()

# -------------------- MQTT HANDLERS --------------------
def on_message(client, userdata, msg):
    global model
    if msg.topic.startswith(f"fog/actuators/{EDGE_ID}"):
        try:
            data = json.loads(msg.payload.decode())
        except:
            data = {"raw": msg.payload.decode()}
        print(f"[{EDGE_ID}] ACTUATOR CMD: {data}")

    elif msg.topic.startswith("fog/global_model"):
        data = json.loads(msg.payload.decode())
        weights = m.unpackb(base64.b64decode(data["weights_b64"]))
        sd = model.state_dict()
        for k,v in weights.items():
            if k in sd:
                sd[k] = torch.tensor(v)
        model.load_state_dict(sd, strict=False)
        print(f"[{EDGE_ID}] Synced with global model round {data['round']}")

client.on_message = on_message

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    print(f"Starting {EDGE_ID} → MQTT at {MQTT_HOST}:{MQTT_PORT}")
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()

    client.subscribe(f"fog/actuators/{EDGE_ID}")
    client.subscribe("fog/global_model")

    buffer_x, buffer_y = [], []
    last_train_time = time.time()

    while True:
        sensor = simulate_sensors(time.time())
        publish_sensor(sensor)
        publish_heartbeat()

        # crude label for demo: anomaly if high temp or high vibration
        label = 1 if sensor["temperature"] > 55 or sensor["vibration"] > 1.0 else 0
        buffer_x.append([sensor["vibration"]])  # using vibration feature as example
        buffer_y.append(label)

        if time.time() - last_train_time > TRAIN_INTERVAL and len(buffer_x) >= BATCH_SIZE:
            batch_x = np.array(buffer_x[-BATCH_SIZE:])
            batch_y = np.array(buffer_y[-BATCH_SIZE:])
            sd = local_train(batch_x, batch_y)
            publish_model_update(sd)
            last_train_time = time.time()

        time.sleep(1.0)
