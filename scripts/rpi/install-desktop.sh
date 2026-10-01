#!/usr/bin/env bash
# Add a RoboMed icon to the Raspberry Pi desktop and application menu.
# Run once:  bash scripts/rpi/install-desktop.sh
set -eu

REPO="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
LAUNCHER="${REPO}/scripts/rpi/uzmax-start.sh"
ICON="${REPO}/scripts/rpi/uzmax.svg"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${HOME}/Desktop")"
APPS_DIR="${HOME}/.local/share/applications"

chmod +x "${LAUNCHER}"
mkdir -p "${DESKTOP_DIR}" "${APPS_DIR}"

ENTRY="[Desktop Entry]
Type=Application
Name=RoboMed
Comment=UzMAX kasalxona robotini ishga tushirish
Exec=${LAUNCHER}
Icon=${ICON}
Terminal=false
Categories=Utility;"

printf '%s\n' "${ENTRY}" > "${APPS_DIR}/robomed.desktop"
printf '%s\n' "${ENTRY}" > "${DESKTOP_DIR}/robomed.desktop"
chmod +x "${APPS_DIR}/robomed.desktop" "${DESKTOP_DIR}/robomed.desktop"

# Mark the desktop file as trusted so it runs on double-click without a prompt.
gio set "${DESKTOP_DIR}/robomed.desktop" metadata::trusted true 2>/dev/null || true

# Raspberry Pi OS file manager: launch executables without asking.
for conf in "${HOME}/.config/libfm/libfm.conf"; do
    if [ -f "${conf}" ]; then
        if grep -q '^quick_exec=' "${conf}"; then
            sed -i 's/^quick_exec=.*/quick_exec=1/' "${conf}"
        else
            sed -i '/^\[config\]/a quick_exec=1' "${conf}"
        fi
    fi
done

echo "Tayyor: ${DESKTOP_DIR}/robomed.desktop"
echo "Ish stolidagi RoboMed belgisini ikki marta bosing."
