// ================================================================
//  AXON – Unified Control Client
//  • USB port selector (scan / connect / disconnect per ESP32)
//  • Movement D-pad + WASD keyboard
//  • Hand servo sliders + quick-action buttons
//  • Head: 360° servo time-control + 84 LED strip
// ================================================================

// ─── Tab switching ────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => {
    t.classList.remove('active');
    t.setAttribute('aria-selected', 'false');
  });
  document.getElementById('pane-' + name).classList.add('active');
  const tabBtn = document.getElementById('tab-' + name);
  tabBtn.classList.add('active');
  tabBtn.setAttribute('aria-selected', 'true');
}

// ════════════════════════════════════════════════════════════════
//  PORT MANAGEMENT
// ════════════════════════════════════════════════════════════════

const ALL_DEVICES = ['move', 'hand', 'head'];
let portsPanelOpen = false;

function togglePortsPanel() {
  portsPanelOpen = !portsPanelOpen;
  const panel = document.getElementById('ports-panel');
  const btn   = document.getElementById('ports-toggle-btn');
  if (portsPanelOpen) {
    panel.removeAttribute('hidden');
    btn.classList.add('open');
    scanPorts();
  } else {
    panel.setAttribute('hidden', '');
    btn.classList.remove('open');
  }
}

// ── Scan available COM ports ───────────────────────────────────
async function scanPorts() {
  document.querySelectorAll('.icon-btn').forEach(b => {
    b.classList.add('spin');
    setTimeout(() => b.classList.remove('spin'), 500);
  });
  try {
    const res   = await fetch('/ports');
    const ports = await res.json();
    ALL_DEVICES.forEach(dev => {
      const sel     = document.getElementById(dev + '-port-select');
      const current = sel.value;
      sel.innerHTML = '<option value="">— select port —</option>';
      ports.forEach(p => {
        const opt = document.createElement('option');
        opt.value       = p.device;
        opt.textContent = `${p.device}  —  ${p.description}`;
        if (p.device === current) opt.selected = true;
        sel.appendChild(opt);
      });
    });
  } catch (e) { console.warn('Port scan failed:', e); }
}

// ── Update card / tab UI for a device ─────────────────────────
function applyConnectionState(device, connected, port) {
  const label  = document.getElementById(device + '-conn-label');
  const dot    = document.getElementById(device + '-dot');
  const tabDot = document.getElementById(device + '-tab-dot');
  const card   = document.getElementById('port-card-' + device);
  const conBtn = document.getElementById(device + '-connect-btn');
  const disBtn = document.getElementById(device + '-disconnect-btn');
  if (!label) return;

  if (connected) {
    label.textContent = `Connected  ·  ${port}`;
    label.className   = 'port-conn-label online';
    dot.className     = 'status-dot online';
    tabDot.className  = 'tab-dot online';
    card.classList.add('connected');
    conBtn.disabled   = true;
    disBtn.disabled   = false;
  } else {
    label.textContent = 'Not connected';
    label.className   = 'port-conn-label offline';
    dot.className     = 'status-dot offline';
    tabDot.className  = 'tab-dot offline';
    card.classList.remove('connected');
    conBtn.disabled   = false;
    disBtn.disabled   = true;
  }
}

// ── Show / hide inline error under a port card ─────────────────
function showPortError(device, message) {
  const el = document.getElementById(device + '-port-error');
  if (!el) return;
  el.textContent = '⚠ ' + friendlyError(message);
  el.removeAttribute('hidden');
}

function clearPortError(device) {
  const el = document.getElementById(device + '-port-error');
  if (el) el.setAttribute('hidden', '');
}

// Turn raw Python exceptions into readable one-liners
function friendlyError(msg) {
  if (!msg) return 'Unknown error';
  // PermissionError(13, 'A device attached...')
  if (msg.includes('PermissionError') || msg.includes('not functioning'))
    return 'Device not responding — unplug & replug the ESP32, then try again.';
  if (msg.includes('FileNotFoundError') || msg.includes('could not open port'))
    return 'Port not found — check the COM port and try Refresh.';
  if (msg.includes('SerialException') || msg.includes('Access is denied'))
    return 'Port in use — close Arduino IDE / Serial Monitor and retry.';
  if (msg.includes('No such file') || msg.includes('does not exist'))
    return 'COM port disappeared — reconnect the USB cable.';
  // fallback: strip the Python class name, show the inner message only
  const inner = msg.match(/'([^']+)'/);
  return inner ? inner[1] : msg;
}

// ── Connect a device ───────────────────────────────────────────
async function connectDevice(device) {
  const sel  = document.getElementById(device + '-port-select');
  const port = sel.value;
  if (!port) {
    showPortError(device, 'Please select a COM port first.');
    return;
  }

  clearPortError(device);
  const conBtn = document.getElementById(device + '-connect-btn');
  conBtn.textContent = 'Connecting…';
  conBtn.disabled    = true;

  try {
    const res  = await fetch('/connect', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ device, port })
    });
    const data = await res.json();
    if (data.ok) {
      applyConnectionState(device, true, port);
    } else {
      applyConnectionState(device, false, null);
      showPortError(device, data.message);
    }
  } catch (e) {
    applyConnectionState(device, false, null);
    showPortError(device, 'Network error — is the server running?');
  }
  conBtn.textContent = 'Connect';
}


// ── Disconnect a device ────────────────────────────────────────
async function disconnectDevice(device) {
  try {
    await fetch('/disconnect', {
      method : 'POST',
      headers: { 'Content-Type': 'application/json' },
      body   : JSON.stringify({ device })
    });
  } catch (e) { /* ignore */ }
  applyConnectionState(device, false, null);
}

// ── Poll /status every 5 s ────────────────────────────────────
async function pollStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();
    ALL_DEVICES.forEach(dev => {
      if (data[dev]) {
        applyConnectionState(dev, data[dev].connected, data[dev].port);
        const port = data[dev].port;
        if (port) {
          const sel = document.getElementById(dev + '-port-select');
          if (sel && !sel.querySelector(`option[value="${port}"]`)) {
            const opt = document.createElement('option');
            opt.value = port; opt.textContent = port; opt.selected = true;
            sel.appendChild(opt);
          } else if (sel) { sel.value = port; }
        }
      }
    });
  } catch (e) { /* server unreachable */ }
}

pollStatus();
setInterval(pollStatus, 5000);

// ════════════════════════════════════════════════════════════════
//  MOVEMENT CONTROL
// ════════════════════════════════════════════════════════════════

const moveStatus = document.getElementById('move-status');

const CMD_LABELS = {
  robot_fwd: 'Forward',   robot_back: 'Reverse',
  spin_ccw:  'Left',      spin_cw:    'Right',
  fwd_left:  'Fwd-Left',  fwd_right:  'Fwd-Right',
  back_left: 'Rev-Left',  back_right: 'Rev-Right',
  stop:      'Stop'
};

const CMD_BTN_MAP = {
  robot_fwd: 'btn-fwd',  robot_back: 'btn-rev',
  spin_ccw:  'btn-left', spin_cw:    'btn-right',
  fwd_left:  'btn-fl',   fwd_right:  'btn-fr',
  back_left: 'btn-bl',   back_right: 'btn-br',
  stop:      'btn-stop'
};

let holdTimeout;
let lastSentCmd = '';

function issueMovement(cmd) {
  fetch('/api/command', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ command: cmd })
  })
  .then(r => r.json())
  .then(d => applyConnectionState('move', d.serial_connected,
             d.serial_connected
               ? document.getElementById('move-port-select').value || null
               : null))
  .catch(() => {});
}

function setDpadActive(cmd) {
  document.querySelectorAll('.dpad-grid .btn').forEach(b => b.classList.remove('active'));
  const btnId = CMD_BTN_MAP[cmd];
  if (btnId) document.getElementById(btnId)?.classList.add('active');
}

function engageDrive(cmd) {
  if (cmd === 'stop') {
    issueMovement('stop');
    lastSentCmd = '';
    setDpadActive('stop');
    moveStatus.innerHTML = '&#x25A0; Emergency Stop';
    moveStatus.className = 'status-pill stopped';
    setTimeout(() => {
      moveStatus.innerHTML = '&#x25CF; System Idle';
      moveStatus.className = 'status-pill';
      setDpadActive('');
    }, 800);
    return;
  }
  if (cmd === lastSentCmd) return;
  lastSentCmd = cmd;
  issueMovement(cmd);
  setDpadActive(cmd);
  moveStatus.innerHTML = `&#x25B6; ${CMD_LABELS[cmd] || cmd}`;
  moveStatus.className = 'status-pill active-cmd';
  clearTimeout(holdTimeout);
  holdTimeout = setTimeout(() => {
    issueMovement('stop');
    lastSentCmd = '';
    moveStatus.innerHTML = '&#x23F1; Auto-Stopped (3 s)';
    moveStatus.className = 'status-pill stopped';
    setDpadActive('');
    setTimeout(() => {
      moveStatus.innerHTML = '&#x25CF; System Idle';
      moveStatus.className = 'status-pill';
    }, 1500);
  }, 3000);
}

function disengageDrive() {
  clearTimeout(holdTimeout);
  issueMovement('stop');
  lastSentCmd = '';
  setDpadActive('');
  moveStatus.innerHTML = '&#x25CF; System Idle';
  moveStatus.className = 'status-pill';
}

document.querySelectorAll('.dpad-grid .btn').forEach(btn => {
  const command = btn.getAttribute('data-cmd');
  btn.addEventListener('mousedown',  () => engageDrive(command));
  btn.addEventListener('mouseup',    disengageDrive);
  btn.addEventListener('mouseleave', disengageDrive);
  btn.addEventListener('touchstart', e => { e.preventDefault(); engageDrive(command); });
  btn.addEventListener('touchend',   e => { e.preventDefault(); disengageDrive(); });
});

const keysHeld = new Set();
function resolveCmd() {
  const w = keysHeld.has('w'), a = keysHeld.has('a');
  const s = keysHeld.has('s'), d = keysHeld.has('d');
  if (w && a) return 'fwd_left';
  if (w && d) return 'fwd_right';
  if (s && a) return 'back_left';
  if (s && d) return 'back_right';
  if (w) return 'robot_fwd';
  if (s) return 'robot_back';
  if (a) return 'spin_ccw';
  if (d) return 'spin_cw';
  return null;
}
document.addEventListener('keydown', e => {
  const key = e.key.toLowerCase();
  if (key === ' ') { e.preventDefault(); engageDrive('stop'); return; }
  if ('wasd'.includes(key) && !keysHeld.has(key)) {
    keysHeld.add(key);
    const cmd = resolveCmd();
    if (cmd) engageDrive(cmd);
  }
});
document.addEventListener('keyup', e => {
  const key = e.key.toLowerCase();
  if ('wasd'.includes(key)) {
    keysHeld.delete(key);
    const cmd = resolveCmd();
    if (cmd) { lastSentCmd = ''; engageDrive(cmd); }
    else disengageDrive();
  }
});

// ════════════════════════════════════════════════════════════════
//  HAND / SERVO CONTROL
// ════════════════════════════════════════════════════════════════

const handStatus = document.getElementById('hand-status');
let servoTimer = null;

function sendServo(cmd) {
  fetch('/send', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ command: cmd })
  })
  .then(r => r.json())
  .then(d => {
    handStatus.textContent = d.response || 'OK';
    handStatus.className   = 'status-pill';
    applyConnectionState('hand', d.serial_connected,
      d.serial_connected
        ? document.getElementById('hand-port-select').value || null
        : null);
  })
  .catch(err => {
    handStatus.textContent = 'Error: ' + err;
    handStatus.className   = 'status-pill error';
  });
}

function updateServo(name, value) {
  document.getElementById(name + '_val').textContent = value;
  // Convert "R1" → "r 1 90",  "L6" → "l 6 45"  (ESP32 format)
  const side = name[0].toLowerCase();   // 'r' or 'l'
  const num  = name[1];                  // '1' – '6'
  const cmd  = `${side} ${num} ${value}`;
  clearTimeout(servoTimer);
  servoTimer = setTimeout(() => sendServo(cmd), 120);
}

// ════════════════════════════════════════════════════════════════
//  HEAD CONTROL  (360° Servo + 84 LED Strip)
// ════════════════════════════════════════════════════════════════

const headStatus  = document.getElementById('head-status');
const headLogText = document.getElementById('head-log-text');

// ── Send raw command to head ESP32 ─────────────────────────────
function headCmd(cmd) {
  headStatus.textContent = `Sending: ${cmd}`;
  headStatus.className   = 'status-pill active-cmd';

  fetch('/head', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ command: cmd })
  })
  .then(r => r.json())
  .then(d => {
    headLogText.textContent = d.response || '—';
    headStatus.textContent  = d.serial_connected ? 'OK' : 'Not connected';
    headStatus.className    = d.serial_connected ? 'status-pill' : 'status-pill error';
    applyConnectionState('head', d.serial_connected,
      d.serial_connected
        ? document.getElementById('head-port-select').value || null
        : null);
  })
  .catch(err => {
    headLogText.textContent = 'Error: ' + err;
    headStatus.textContent  = 'Network error';
    headStatus.className    = 'status-pill error';
  });
}

// ── Servo helpers ──────────────────────────────────────────────
function headCW() {
  const ms = parseInt(document.getElementById('cw-ms').value) || 1000;
  headCmd(`cw ${ms}`);
}

function headCCW() {
  const ms = parseInt(document.getElementById('ccw-ms').value) || 1000;
  headCmd(`ccw ${ms}`);
}

function headRun() {
  const val = parseInt(document.getElementById('run-val').value);
  const ms  = parseInt(document.getElementById('run-ms').value) || 1000;
  headCmd(`run ${val} ${ms}`);
}

function headSetStop() {
  const val = parseInt(document.getElementById('stop-val').value);
  headCmd(`stop ${val}`);
}

// ── LED helpers ────────────────────────────────────────────────

// Convert hex color (#rrggbb) → {r,g,b}
function hexToRgb(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return { r, g, b };
}

function updateRgbReadout(r, g, b) {
  document.getElementById('led-r').textContent = r;
  document.getElementById('led-g').textContent = g;
  document.getElementById('led-b').textContent = b;
}

// Called when color picker changes
function ledPickerChange(hex) {
  const { r, g, b } = hexToRgb(hex);
  updateRgbReadout(r, g, b);
}

// Send current picker color
function ledSend() {
  const hex    = document.getElementById('led-color-picker').value;
  const { r, g, b } = hexToRgb(hex);
  headCmd(`led ${r} ${g} ${b}`);
}

// Preset color buttons
function ledPreset(r, g, b) {
  updateRgbReadout(r, g, b);
  // Update picker to match (convert back to hex)
  const hex = '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
  document.getElementById('led-color-picker').value = hex;
  headCmd(`led ${r} ${g} ${b}`);
}

// Brightness slider (debounced)
let brightTimer = null;
function updateBrightness(value) {
  document.getElementById('bright-val').textContent = value;
  clearTimeout(brightTimer);
  brightTimer = setTimeout(() => headCmd(`brightness ${value}`), 150);
}
