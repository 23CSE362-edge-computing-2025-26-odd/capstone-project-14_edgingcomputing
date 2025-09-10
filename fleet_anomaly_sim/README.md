
# Fleet Anomaly Detection — Edge ⇢ Fog ⇢ Cloud (Simulation)

This repo simulates your architecture diagram using **Python + Docker**:
- **Edge (STM32/RPi simulated)**: 3 containers publishing sensor data to MQTT.
- **Fog (Industrial PC)**: aggregates fleet data, computes simple z-score anomalies, publishes actuator commands.
- **Cloud (AWS/Azure/GCP simulated)**: Streamlit dashboard subscribing to MQTT and visualizing fleet status.
- **Broker**: Eclipse Mosquitto (MQTT).

## Quick Start

```bash
# 1) cd into the folder
cd fleet_anomaly_sim

# 2) Launch the stack
docker compose up --build

# 3) Open the dashboard
#    http://localhost:8501
```

You should see `edge1/2/3` printing actuator commands when anomalies are detected,
and the **Cloud dashboard** will show real-time reports.

## Topics

- Edge publishes:
  - `edge/{edge_id}/sensors` → JSON with temperature, humidity, pressure, electrical, vibration
  - `edge/{edge_id}/heartbeat`
- Fog publishes:
  - `fog/actuators/{edge_id}` → JSON commands (e.g., `{ "alarm": true }`)
  - `cloud/reports` → JSON status per edge
  - `cloud/fleet_snapshot` → fleet-level snapshot

## Tuning

- `fog` service env:
  - `WINDOW_SIZE` (default 50 samples)
  - `Z_THRESH` (default 2.5)
- `edge` services env:
  - `ANOMALY_RATE` (probability of anomaly injection per message)

## Extending Toward the Paper

- Replace fog z-score logic with:
  - **DTW/W-DTW warping amount** between edges (Block 1)
  - **Hierarchical clustering + cophenetic correlation** to pick clusters (Block 2)
  - **Anomaly score = 1 - (cluster_size / fleet_size)** (Block 3)
- Keep MQTT topics the same — only swap `fog_node.py` internals.

## Files

- `docker-compose.yml`
- `mosquitto/mosquitto.conf`
- `edge/` → `Dockerfile`, `edge_node.py`
- `fog/` → `Dockerfile`, `fog_node.py`
- `cloud/` → `Dockerfile`, `cloud_app.py`
```

