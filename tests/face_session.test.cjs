const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../uzmax_server/static/index.html'), 'utf8');
const script = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)].map(match => match[1]).join('\n');
new vm.Script(script);
const between = (start, end) => script.slice(script.indexOf(start), script.indexOf(end));

function faceSession() {
  const elements = new Map();
  const sent = [];
  let now = 10000;
  const context = vm.createContext({
    console,
    Date: { now: () => now },
    document: { getElementById: id => {
      if (!elements.has(id)) elements.set(id, {
        textContent: '', style: {}, classList: { contains: () => false },
      });
      return elements.get(id);
    } },
    location: { protocol: 'http:' },
    WebSocket: class {
      send(message) { sent.push(JSON.parse(message)); }
    },
    setWS() {}, chatMsg() {}, enableLiveConversation() {},
    updatePersonVitals() {}, patientThermalPayload: () => null,
    sendChatJson: message => sent.push(message),
  });
  vm.runInContext(`
    let ws, liveMode = false, allowInterrupt = false, pendingChatMessages = [];
    const WS_ORIGIN = 'localhost';
    ${between('let faceEnabled = false', 'function faceBoxKey')}
    function drawFaceBoxes() {}
    ${between('function faceSampleScore', 'async function resetFaceId')}
    ${between('async function runFaceIdentify', '// Fast path:')}
    ${between('function wsInit()', 'function setWS(')}
    faceEnabled = true;
    wsInit();
  `, context);
  return { context, sent, elements, run: code => vm.runInContext(code, context), tick: ms => { now += ms; } };
}

test('registration updates the displayed name and discards an in-flight unknown result', async () => {
  const session = faceSession();
  let release;
  session.context.fetch = () => new Promise(resolve => { release = resolve; });
  const request = session.run('runFaceIdentify("image")');
  await session.run(`ws.onmessage({data: JSON.stringify({
    type: 'patient_registered', person: {person_id: 'patient-1', full_name: 'Xarilov Jamshid'}
  })})`);
  release({ json: async () => ({ faces: [{ status: 'unknown', snapshot_path: 'stale.jpg' }] }) });
  await request;
  assert.equal(session.elements.get('p-name').textContent, 'Xarilov Jamshid');
  assert.equal(session.run('activeKnownPersonId'), 'patient-1');
  assert.equal(session.run('pendingUnknownFace'), false);
  assert.equal(session.sent.length, 0);
});

test('brief unknown frames retain the conversation; a sustained new face starts onboarding', () => {
  const session = faceSession();
  session.run(`notifyFaceIdentity({status: 'known', person: {person_id: 'patient-1', full_name: 'Xarilov Jamshid'}})`);
  session.sent.length = 0;
  session.run(`notifyFaceIdentity({status: 'unknown', snapshot_path: 'new.jpg'})`);
  session.tick(1000);
  session.run(`notifyFaceIdentity({status: 'unknown', snapshot_path: 'new2.jpg'})`);
  assert.equal(session.sent.length, 0);
  assert.equal(session.run('activeKnownPersonId'), 'patient-1');
  session.tick(2500);
  session.run(`notifyFaceIdentity({status: 'unknown', snapshot_path: 'new3.jpg'})`);
  assert.equal(session.run('activeKnownPersonId'), null);
  assert.equal(session.sent[0].type, 'face_identity');
  assert.ok(session.sent[0].pending_registration);
});

test('a reconnected chat receives fresh face identity and an absent face clears it', () => {
  const session = faceSession();
  const known = `notifyFaceIdentity({status: 'known', person: {person_id: 'patient-1', full_name: 'Xarilov Jamshid'}})`;
  session.run(known);
  session.sent.length = 0;
  session.run('ws.onopen()');
  session.run(known);
  assert.equal(session.sent.filter(message => message.type === 'face_identity').length, 1);
  session.tick(2000);
  session.run(`notifyFaceIdentity({status: 'no_face'})`);
  assert.equal(session.run('activeKnownPersonId'), null);
  assert.equal(session.run('activeKnownPerson'), null);
  assert.equal(session.sent.at(-1).type, 'person_left');
});

function cameraSession() {
  const session = faceSession();
  const timers = [];
  session.context.clearTimeout = () => {};
  session.context.setTimeout = callback => { timers.push(callback); return timers.length; };
  session.context.navigator = { mediaDevices: {} };
  session.run(between('function cameraErrorMessage', "document.getElementById('tgl-face').addEventListener"));
  session.run(between('function captureFaceFrameB64', '// Heavy path:'));
  session.run(between('async function detectFace(', "const DEVS ="));
  return {...session, timers};
}

test('a camera opened after being switched off is released without starting detection', async () => {
  const session = cameraSession();
  let release;
  let stops = 0;
  session.context.navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { release = resolve; });
  const start = session.run('startFaceCamera()');
  session.run('stopFaceCamera()');
  release({getTracks: () => [{stop: () => { stops++; }}]});
  await start;
  assert.equal(stops, 1);
  assert.equal(session.run('faceEnabled'), false);
  assert.equal(session.run('faceCamSt'), null);
  assert.equal(session.timers.length, 0);
});

test('busy camera and playback failures leave no stream or checked toggle', async () => {
  const session = cameraSession();
  session.context.navigator.mediaDevices.getUserMedia = async () => {
    throw Object.assign(new Error('Device in use'), {name: 'NotReadableError'});
  };
  await session.run('startFaceCamera()');
  assert.match(session.elements.get('p-sub').textContent, /boshqa dasturda band/);
  assert.equal(session.elements.get('tgl-face').checked, false);
  let stops = 0;
  session.context.navigator.mediaDevices.getUserMedia = async () => ({
    getTracks: () => [{stop: () => { stops++; }}],
  });
  session.elements.get('face-video').play = async () => { throw new Error('Playback failed'); };
  await session.run('startFaceCamera()');
  assert.equal(stops, 1);
  assert.equal(session.run('faceCamSt'), null);
  assert.equal(session.elements.get('face-video').srcObject, null);
  assert.equal(session.elements.get('tgl-face').checked, false);
});

test('detection waits for video frames and keeps retrying after a capture error', async () => {
  const session = cameraSession();
  session.elements.set('face-video', {readyState: 0, videoWidth: 0, videoHeight: 0});
  await session.run('detectFace()');
  assert.equal(session.timers.length, 1);
  session.run('captureFaceFrameB64 = () => { throw new Error("Frame not ready"); }');
  session.context.console = {warn() {}};
  await session.timers[0]();
  assert.equal(session.timers.length, 2);
  session.run('stopFaceCamera()');
  await session.timers[1]();
  assert.equal(session.timers.length, 2);
});

test('microphone denial stops automatic voice retries without disabling the camera', () => {
  const session = faceSession();
  session.context.micState = () => {};
  session.run('let microphoneAutoBlocked = false, pendingAutoListen = true; liveMode = true;');
  session.run(between('function enableLiveConversation', 'function sendChatJson'));
  session.run(between('function disableUnavailableMicrophone', "document.getElementById('tgl-live').addEventListener"));
  session.run('disableUnavailableMicrophone({name: "NotAllowedError"}); enableLiveConversation();');
  assert.equal(session.run('liveMode'), false);
  assert.equal(session.run('pendingAutoListen'), false);
  assert.equal(session.run('faceEnabled'), true);
  assert.match(session.elements.get('stt-partial').textContent, /ruxsat berilmagan/);
  assert.equal(session.sent.filter(message => message.live_mode === true).length, 0);
});
