**Fleet-Based Anomaly Detection with ME-FEEL (Multi-Exit Federated Edge Learning)**

This project simulates a fleet-level anomaly detection system across Edge → Fog → Cloud, using MQTT for communication and ME-FEEL for federated learning.
It demonstrates how heterogeneous edge devices can collaboratively train lightweight models while keeping bandwidth low and improving anomaly detection robustness.
**
Architecture
**
 ┌───────────┐        ┌─────────┐        ┌──────────┐
 │  Edge N   │  --->  │   Fog   │  --->  │  Cloud   │
 └───────────┘        └─────────┘        └──────────┘
     │ Sensors            │ Aggregation       │ Dashboard
     │ Local model        │ Anomaly detection │ Fleet view
     │ ME-FEEL exits      │ Global sync       │ Streamlit

**
Edge**

Simulates sensor data (temperature, vibration, electrical load, etc.)

Runs partial ME-FEEL training (light devices → shallow exits, stronger devices → deeper exits)

Publishes sensor readings + local model updates

**Fog
**
Collects edge data and anomaly signals

Performs layer-wise averaging for global ME-FEEL aggregation

Sends actuator commands back to edges

Publishes fleet snapshots + updated global model
**
Cloud**

Interactive Streamlit dashboard

Displays fleet status, anomaly scores, federated learning progress, and pr-edge contributions

**Tech Stack**

Messaging: Eclipse Mosquitto (MQTT broker)

Edge ML: PyTorch (ME-FEEL model with multi-exits)

Fog Processing: Pandas + msgpack-numpy for anomaly detection & aggregation

Cloud Dashboard: Streamlit + Pandas

Containerization: Docker + Docker Compose

**Setup Instructions**

1. Clone the Repo

git clone https://github.com/yourusername/fleet_anomaly_sim.git
cd fleet_anomaly_sim


2. Build & Run With Docker

git clone https://github.com/yourusername/fleet_anomaly_sim.git
cd fleet_anomaly_sim



**
This will start:**

mqtt → Eclipse Mosquitto broker

edge1, edge2, edge3 → Edge nodes simulating sensors + ME-FEEL exits

fog → Aggregation & anomaly detection node

cloud → Dashboard at http://localhost:8501

**Dashboard Preview**

**Fleet Snapshot:** Bar chart of anomaly scores per edge

**Recent Reports:** Live feed of anomaly detection outputs

**Federated Learning Progress:** Tracking global model updates from fog

**Per-Edge Training Contributions:** Visualization of training depth across devices
**
Key Features**

Realistic Fleet Simulation with multiple heterogeneous edge nodes

ME-FEEL Integration for federated learning with exit points

Bandwidth Efficiency — edges only send model updates, not raw data

Fault-Tolerant Learning — weaker nodes can still contribute shallow training

Visual Dashboard for monitoring fleet anomalies in real time

**Example Use Cases**

Predictive maintenance of industrial machines

Fleet monitoring of vehicles / IoT devices

Smart grid and energy anomaly detection

Remote sensor health monitoring

**Next Steps**

Add adaptive anomaly thresholds per edge

Extend ME-FEEL model with more exit strategies

Deploy on real edge hardware (e.g., Raspberry Pi + STM32 mix)
