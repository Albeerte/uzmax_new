"""
UzMAX Raspberry Pi thermal sender.

Run this on the Raspberry Pi that has the MLX90640 connected over I2C.
It pushes live 32x24 thermal frames to the UzMAX FastAPI server, so the
dashboard can be opened from any browser link.

Install on Raspberry Pi:
    pip3 install requests numpy adafruit-circuitpython-mlx90640 adafruit-blinka --break-system-packages

Run:
    UZMAX_SERVER_URL=http://SERVER_IP:5000 python3 thermal_agent.py
"""

import os
import time

import numpy as np
import requests

import board
import busio
import adafruit_mlx90640


ROWS = 24
COLS = 32
SERVER_URL = os.getenv("UZMAX_SERVER_URL", "http://127.0.0.1:5000").rstrip("/")
PUSH_URL = f"{SERVER_URL}/api/thermal/push"
PUSH_INTERVAL_SECONDS = float(os.getenv("UZMAX_THERMAL_INTERVAL", "0.25"))


def build_sensor():
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400_000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
    return mlx


def main():
    print(f"UzMAX thermal agent pushing to {PUSH_URL}")
    mlx = build_sensor()
    frame = [0.0] * (ROWS * COLS)

    while True:
        try:
            mlx.getFrame(frame)
            arr = np.array(frame, dtype=np.float32).reshape((ROWS, COLS))
            payload = {
                "source": "raspberry_pi",
                "rows": ROWS,
                "cols": COLS,
                "frame": arr.flatten().round(2).tolist(),
                "max": round(float(arr.max()), 1),
                "min": round(float(arr.min()), 1),
                "avg": round(float(arr.mean()), 1),
                "center": round(float(arr[ROWS // 2, COLS // 2]), 1),
            }
            response = requests.post(PUSH_URL, json=payload, timeout=3)
            response.raise_for_status()
            print(f"OK max={payload['max']} center={payload['center']}")
        except Exception as exc:
            print(f"THERMAL PUSH ERROR: {exc}")

        time.sleep(PUSH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
