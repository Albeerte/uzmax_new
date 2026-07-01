# UzMAX Server

Unified FastAPI server for the UzMAX medical assistant, thermal camera, face
recognition, and ESP32 robot controls.

Run from the repository root:

```powershell
python uzmax_server/main.py
```

Open:

```text
http://127.0.0.1:5000/
```

The server listens on `0.0.0.0:5000` by default, so devices on the same network
can open `http://YOUR_SERVER_IP:5000/`.
