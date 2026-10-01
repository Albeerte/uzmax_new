#!/usr/bin/env bash
# Start the UzMAX server (if it is not already running) and open the UI
# full-screen in Chromium. Used by the RoboMed desktop icon.
set -u

REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
PORT="${UZMAX_PORT:-5000}"
URL="http://localhost:${PORT}"
LOG="${HOME}/robomed-server.log"
PYTHON="${REPO}/.venv/bin/python"

notify() {
    command -v notify-send >/dev/null 2>&1 && notify-send -i "${REPO}/scripts/rpi/uzmax.svg" "RoboMed" "$1"
    echo "$1"
}

server_up() {
    curl -fs -o /dev/null --max-time 2 "${URL}/health"
}

if ! server_up; then
    if [ ! -x "${PYTHON}" ]; then
        notify "Python muhiti topilmadi: ${PYTHON}"
        exit 1
    fi
    notify "Server ishga tushirilmoqda..."
    cd "${REPO}"
    nohup "${PYTHON}" -u uzmax_server/main.py >>"${LOG}" 2>&1 &

    # The face model takes a while to load on a Pi.
    for _ in $(seq 1 120); do
        server_up && break
        sleep 1
    done
    if ! server_up; then
        notify "Server ishga tushmadi. Logni ko'ring: ${LOG}"
        exit 1
    fi
fi

BROWSER="$(command -v chromium || command -v chromium-browser || true)"
if [ -z "${BROWSER}" ]; then
    xdg-open "${URL}"
    exit 0
fi

# Separate profile so camera/mic permissions are remembered and the user's
# normal browser session is not touched.
exec "${BROWSER}" \
    --user-data-dir="${HOME}/.config/uzmax-chromium" \
    --app="${URL}" \
    --start-fullscreen \
    --use-fake-ui-for-media-stream \
    --autoplay-policy=no-user-gesture-required \
    --no-first-run \
    --noerrdialogs \
    --disable-session-crashed-bubble \
    --password-store=basic
