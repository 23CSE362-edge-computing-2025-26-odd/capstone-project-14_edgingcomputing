# ME-FLEET — Quick Glimpse

ME-FLEET is a compact framework for fleet-based anomaly detection using multi-exit neural models and federated learning across Edge–Fog–Cloud tiers. It focuses on low-latency, privacy-preserving inference with optional early exits on resource-constrained devices.

## Key Points
- Multi-exit models for early, efficient inference.
- Federated training to keep data on-device.
- Edge–Fog–Cloud orchestration for scalable deployment.

## Quick Run (Local)
1. Create & activate a venv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a local simulation:
   ```bash
   python simulate_federated.py --clients 10 --rounds 20
   ```

That's it — use `config/*.yaml` to tune clients, exits, and training settings.
