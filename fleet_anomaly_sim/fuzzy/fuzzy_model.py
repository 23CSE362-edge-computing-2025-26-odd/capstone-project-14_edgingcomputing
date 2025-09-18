import os, json, time, threading, collections
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
from datetime import datetime
import skfuzzy as fuzz
from skfuzzy import control as ctrl

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))
NUM_CLUSTERS = int(os.getenv("NUM_CLUSTERS", "2"))

METRICS = ["temperature", "humidity", "pressure", "electrical", "vibration"]
buffers = collections.defaultdict(lambda: collections.deque(maxlen=WINDOW_SIZE))
client = mqtt.Client(client_id="fog-collector", clean_session=True)

def classify_anomalies_fuzzy(df, num_clusters):
    """
    This is the new anomaly classification function using fuzzy logic.
    It replaces the Z-score calculation.

    Args:
        df (pd.DataFrame): DataFrame containing the latest metrics for each machine.
        num_clusters (int): The number of clusters for Fuzzy C-Means.

    Returns:
        list: A list of health scores (0-100) for each machine.
    """
    if len(df) < num_clusters:
        return [100.0] * len(df)

    metric_data = df[METRICS].values
    
    normalized_data = (metric_data - metric_data.mean(axis=0)) / (metric_data.std(axis=0) + 1e-9)
    data_for_fcm = normalized_data.T
    
    try:
        _, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            data_for_fcm, num_clusters, 2, error=0.005, maxiter=1000, init=None
        )
    except Exception as e:
        print(f"[fog-fuzzy] Clustering failed: {e}. Assuming healthy.")
        return [100.0] * len(df)

    anomaly_score_antecedent = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'anomaly_score')
    health_status_consequent = ctrl.Consequent(np.arange(0, 101, 1), 'health_status')

    anomaly_score_antecedent['low'] = fuzz.trimf(anomaly_score_antecedent.universe, [0, 0, 0.5])
    anomaly_score_antecedent['medium'] = fuzz.trimf(anomaly_score_antecedent.universe, [0.2, 0.5, 0.8])
    anomaly_score_antecedent['high'] = fuzz.trimf(anomaly_score_antecedent.universe, [0.5, 1, 1])

    health_status_consequent['faulty'] = fuzz.trimf(health_status_consequent.universe, [0, 0, 40])
    health_status_consequent['warning'] = fuzz.trimf(health_status_consequent.universe, [20, 50, 80])
    health_status_consequent['healthy'] = fuzz.trimf(health_status_consequent.universe, [60, 100, 100])
    
    rule1 = ctrl.Rule(anomaly_score_antecedent['low'], health_status_consequent['healthy'])
    rule2 = ctrl.Rule(anomaly_score_antecedent['medium'], health_status_consequent['warning'])
    rule3 = ctrl.Rule(anomaly_score_antecedent['high'], health_status_consequent['faulty'])

    health_ctrl_system = ctrl.ControlSystem([rule1, rule2, rule3])
    health_simulation = ctrl.ControlSystemSimulation(health_ctrl_system)

    num_machines = u.shape[1]
    cluster_sizes = np.sum(u, axis=1)
    largest_cluster_idx = np.argmax(cluster_sizes)
    
    health_scores = []
    for i in range(num_machines):
        membership_in_healthy = u[largest_cluster_idx, i]
        fuzzy_anomaly_score = 1.0 - membership_in_healthy
        
        health_simulation.input['anomaly_score'] = fuzzy_anomaly_score
        health_simulation.compute()
        health_scores.append(health_simulation.output['health_status'])
        
    return health_scores

def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("edge/+/sensors")
    client.subscribe("edge/+/heartbeat")

def on_message(client, userdata, msg):
    if msg.topic.endswith("/sensors"):
        try:
            data = json.loads(msg.payload.decode())
            buffers[data["edge_id"]].append(data)
        except json.JSONDecodeError:
            print(f"[fog] Could not decode JSON from topic {msg.topic}")

def compute_anomalies():
    while True:
        time.sleep(2.0)
        
        latest = []
        for edge_id, buf in buffers.items():
            if len(buf) > 0:
                latest.append({ "edge_id": edge_id, **buf[-1] })
        
        if not latest:
            continue

        df = pd.DataFrame(latest)
        df_metrics = df.set_index("edge_id")[METRICS]

        health_scores = classify_anomalies_fuzzy(df_metrics, NUM_CLUSTERS)
        df_metrics["health_score"] = health_scores
        
        for edge_id, row in df_metrics.iterrows():
            score = row["health_score"]
            decision = "OK"
            if score < 40: # Corresponds to 'faulty' in the FIS
                decision = "ALERT"
                cmd = {"edge_id": edge_id, "ts": datetime.utcnow().isoformat()+"Z", "alarm": True, "reason": "Low health score detected", "score": score}
                client.publish(f"fog/actuators/{edge_id}", json.dumps(cmd), qos=0)
            elif score < 80: # Corresponds to 'warning'
                decision = "WARNING"
                
            report = {
                "edge_id": edge_id,
                "ts": datetime.utcnow().isoformat()+"Z",
                "decision": decision,
                "health_score": round(float(score), 2),
                "metrics": {m: float(row[m]) for m in METRICS}
            }
            client.publish("cloud/reports", json.dumps(report), qos=0)
        
        snapshot = {
            "ts": datetime.utcnow().isoformat()+"Z",
            "fleet_size": len(df_metrics),
            "model_type": "FuzzyLogic",
            "edges": df_metrics.reset_index()[["edge_id", "health_score"]].to_dict(orient="records")
        }
        client.publish("cloud/fleet_snapshot", json.dumps(snapshot), qos=0)

if __name__ == "__main__":
    print(f"[fog] Connecting to MQTT at {MQTT_HOST}:{MQTT_PORT}")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_start()
    
    processing_thread = threading.Thread(target=compute_anomalies, daemon=True)
    processing_thread.start()
    
    print(f"[fog] Fog node started. Analyzing fleet with Fuzzy Logic using {NUM_CLUSTERS} clusters.")
    while True:
        time.sleep(1)