"""Long-running device services: supervised threads composing the SensorDyme runtime.

Layout:
    supervisor.py    -- Service base class + ordered start/stop with crash restarts
    capture.py       -- owns the camera; publishes frames to the FrameBus
    provisioning.py  -- Wi-Fi state machine; opens BLE provisioning when offline
    ble_gatt.py      -- BlueZ GATT server (Pi-only; import-guarded)
    runtime.py       -- builds and wires everything from config; called by main.py
"""
