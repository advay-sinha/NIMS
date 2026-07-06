"""Engine B — Network Health Intelligence over SNMP/MIB telemetry.

CSV-based telemetry pipeline mirroring Engine A's design: configurable
schema validation, leakage-free chronological preprocessing (counter deltas,
rates, splits), health feature engineering and an Isolation Forest anomaly
baseline, all persisting reproducible artefacts under
``outputs/network_health/``. Live SNMP polling, the LSTM autoencoder and the
correlation engine are later phases.
"""
