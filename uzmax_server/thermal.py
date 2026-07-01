"""
UzMAX MLX90640 thermal camera test/agent.

Run on the Raspberry Pi connected to the MLX90640:
    python thermal.py

To push frames into the UzMAX web dashboard running on another machine/server:
    python thermal.py --server http://YOUR_SERVER_IP:5000 --no-window
"""

import argparse
import time

import adafruit_mlx90640
import board
import busio
import cv2
import numpy as np
import requests


ROWS = 24
COLS = 32
FRAME_SIZE = ROWS * COLS


def parse_args():
    parser = argparse.ArgumentParser(description="UzMAX MLX90640 thermal camera")
    parser.add_argument("--server", default="", help="Optional UzMAX server URL, for example http://127.0.0.1:5000")
    parser.add_argument("--source", default="raspberry_pi", help="Source name sent to /api/thermal/push")
    parser.add_argument("--no-window", action="store_true", help="Do not open the OpenCV preview window")
    parser.add_argument("--i2c-frequency", type=int, default=100_000, help="I2C frequency in Hz")
    parser.add_argument("--push-interval", type=float, default=0.25, help="Minimum seconds between server pushes")
    return parser.parse_args()


def open_sensor(i2c_frequency: int):
    i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_frequency)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
    return mlx


def render_window(thermal: np.ndarray, min_temp: float, max_temp: float, center_temp: float) -> bool:
    img = cv2.normalize(thermal, None, 0, 255, cv2.NORM_MINMAX)
    img = np.uint8(img)
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    cv2.putText(
        img,
        f"MIN: {min_temp:.1f}C  MAX: {max_temp:.1f}C  CENTER: {center_temp:.1f}C",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.imshow("UZMAX Thermal Camera", img)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def push_frame(server: str, source: str, thermal: np.ndarray, min_temp: float, max_temp: float, avg_temp: float, center_temp: float):
    url = server.rstrip("/") + "/api/thermal/push"
    payload = {
        "source": source,
        "frame": thermal.flatten().round(2).tolist(),
        "min": round(float(min_temp), 1),
        "max": round(float(max_temp), 1),
        "avg": round(float(avg_temp), 1),
        "center": round(float(center_temp), 1),
    }
    requests.post(url, json=payload, timeout=3).raise_for_status()


def main():
    args = parse_args()
    mlx = open_sensor(args.i2c_frequency)
    frame = [0.0] * FRAME_SIZE
    last_push = 0.0

    print("MLX90640 started")
    if args.server:
        print(f"Pushing frames to {args.server.rstrip('/')}/api/thermal/push")
    if not args.no_window:
        print("Press Q to exit")

    try:
        while True:
            try:
                mlx.getFrame(frame)
            except ValueError:
                print("Frame read error, retrying...")
                time.sleep(0.2)
                continue

            thermal = np.array(frame, dtype=np.float32).reshape((ROWS, COLS))
            min_temp = float(np.min(thermal))
            max_temp = float(np.max(thermal))
            avg_temp = float(np.mean(thermal))
            center_temp = float(thermal[ROWS // 2, COLS // 2])

            if args.server and time.time() - last_push >= args.push_interval:
                try:
                    push_frame(args.server, args.source, thermal, min_temp, max_temp, avg_temp, center_temp)
                    last_push = time.time()
                except Exception as exc:
                    print("Server push error:", exc)

            if not args.no_window:
                if not render_window(thermal, min_temp, max_temp, center_temp):
                    break

    finally:
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
