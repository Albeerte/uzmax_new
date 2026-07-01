// ============================================================
// UzMAX MOVE CONTROLLER v4 - ESP32
// Hold-to-move dashboard control for 2 stepper motors.
//
// Dashboard protocol:
//   PING              -> DEVICE:MOVE
//   MOVE SPEED 60     -> speed 0..100
//   MOVE FWD          -> start moving forward
//   MOVE BACK         -> start moving backward
//   MOVE LEFT         -> pivot/turn left
//   MOVE RIGHT        -> pivot/turn right
//   MOVE PING         -> keepalive while button/key is held
//   MOVE STOP         -> stop when button/key is released
//
// Also supports test commands:
//   left_fwd, left_back, right_fwd, right_back
//   robot_fwd, robot_back, spin_cw, spin_ccw, demo, stop
// ============================================================

#include <Arduino.h>

#define DEVICE_NAME "MOVE"

#define LEFT_STEP_PIN   18
#define LEFT_DIR_PIN    19
#define LEFT_EN_PIN     21

#define RIGHT_STEP_PIN  22
#define RIGHT_DIR_PIN   23
#define RIGHT_EN_PIN    25

#define DRIVER_ENABLE_LEVEL  LOW
#define DRIVER_DISABLE_LEVEL HIGH

// Speed is pulse delay in microseconds: lower = faster.
#define PULSE_SLOW          3000
#define PULSE_FAST           300
#define ACCEL_STEPS          250
#define DECEL_STEPS          120
#define SERIAL_CHECK_STEPS    10
#define KEEPALIVE_TIMEOUT_MS 350

String queuedCommand = "";
volatile bool stopRequested = false;
bool motorRunning = false;
int currentSpeed = 60;
unsigned long lastKeepAlive = 0;

int speedToPulseDelay(int speed) {
  speed = constrain(speed, 0, 100);
  return map(speed, 0, 100, PULSE_SLOW, PULSE_FAST);
}

float smoothStep(float t) {
  t = constrain(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

int rampDelay(int stepIndex, int targetDelay) {
  if (stepIndex >= ACCEL_STEPS) return targetDelay;
  float t = smoothStep((float)stepIndex / ACCEL_STEPS);
  return (int)(PULSE_SLOW - t * (PULSE_SLOW - targetDelay));
}

int stopDelay(int stepIndex, int startDelay) {
  if (stepIndex >= DECEL_STEPS) return PULSE_SLOW;
  float t = smoothStep((float)stepIndex / DECEL_STEPS);
  return (int)(startDelay + t * (PULSE_SLOW - startDelay));
}

void enableLeftDriver() {
  digitalWrite(LEFT_EN_PIN, DRIVER_ENABLE_LEVEL);
}

void disableLeftDriver() {
  digitalWrite(LEFT_EN_PIN, DRIVER_DISABLE_LEVEL);
}

void enableRightDriver() {
  digitalWrite(RIGHT_EN_PIN, DRIVER_ENABLE_LEVEL);
}

void disableRightDriver() {
  digitalWrite(RIGHT_EN_PIN, DRIVER_DISABLE_LEVEL);
}

void enableAllDrivers() {
  enableLeftDriver();
  enableRightDriver();
}

void disableAllDrivers() {
  disableLeftDriver();
  disableRightDriver();
}

void pulsePin(int stepPin, int delayUs) {
  digitalWrite(stepPin, HIGH);
  delayMicroseconds(delayUs);
  digitalWrite(stepPin, LOW);
  delayMicroseconds(delayUs);
}

void pulseBoth(int delayUs) {
  digitalWrite(RIGHT_STEP_PIN, HIGH);
  digitalWrite(LEFT_STEP_PIN, HIGH);
  delayMicroseconds(delayUs);
  digitalWrite(RIGHT_STEP_PIN, LOW);
  digitalWrite(LEFT_STEP_PIN, LOW);
  delayMicroseconds(delayUs);
}

void applyBothDirection(bool leftForward, bool rightForward) {
  // Direction mapping follows your corrected test code.
  digitalWrite(RIGHT_DIR_PIN, leftForward ? HIGH : LOW);
  digitalWrite(LEFT_DIR_PIN, rightForward ? HIGH : LOW);
}

void checkSerial() {
  if (motorRunning && millis() - lastKeepAlive > KEEPALIVE_TIMEOUT_MS) {
    stopRequested = true;
    return;
  }

  if (!Serial.available()) return;

  String incoming = Serial.readStringUntil('\n');
  incoming.trim();
  incoming.toUpperCase();
  if (incoming.length() == 0) return;

  if (incoming == "PING") {
    Serial.println("DEVICE:MOVE");
    return;
  }

  if (incoming == "MOVE PING") {
    lastKeepAlive = millis();
    return;
  }

  if (incoming.startsWith("MOVE SPEED ")) {
    currentSpeed = constrain(incoming.substring(11).toInt(), 0, 100);
    Serial.print("OK:SPEED ");
    Serial.println(currentSpeed);
    return;
  }

  if (incoming == "MOVE STOP" || incoming == "STOP") {
    stopRequested = true;
    return;
  }

  queuedCommand = incoming;
  stopRequested = true;
}

void continuousBoth(bool leftForward, bool rightForward) {
  enableAllDrivers();
  applyBothDirection(leftForward, rightForward);

  motorRunning = true;
  stopRequested = false;
  lastKeepAlive = millis();

  int accelStep = 0;
  int decelStep = 0;
  int serialStep = 0;
  int currentDelay = PULSE_SLOW;
  bool decelerating = false;

  while (true) {
    int targetDelay = speedToPulseDelay(currentSpeed);

    if (!decelerating) {
      currentDelay = rampDelay(accelStep++, targetDelay);
    }

    if (stopRequested) decelerating = true;

    if (decelerating) {
      currentDelay = stopDelay(decelStep++, currentDelay);
      if (decelStep >= DECEL_STEPS) break;
    }

    pulseBoth(currentDelay);

    if (++serialStep >= SERIAL_CHECK_STEPS) {
      serialStep = 0;
      checkSerial();
    }
  }

  motorRunning = false;
  disableAllDrivers();
  Serial.println("OK:STOP");
}

void continuousSingle(int stepPin, int dirPin, int enPin, bool dir) {
  digitalWrite(enPin, DRIVER_ENABLE_LEVEL);
  digitalWrite(dirPin, dir ? HIGH : LOW);

  motorRunning = true;
  stopRequested = false;
  lastKeepAlive = millis();

  int accelStep = 0;
  int decelStep = 0;
  int serialStep = 0;
  int currentDelay = PULSE_SLOW;
  bool decelerating = false;

  while (true) {
    int targetDelay = speedToPulseDelay(currentSpeed);

    if (!decelerating) {
      currentDelay = rampDelay(accelStep++, targetDelay);
    }

    if (stopRequested) decelerating = true;

    if (decelerating) {
      currentDelay = stopDelay(decelStep++, currentDelay);
      if (decelStep >= DECEL_STEPS) break;
    }

    pulsePin(stepPin, currentDelay);

    if (++serialStep >= SERIAL_CHECK_STEPS) {
      serialStep = 0;
      checkSerial();
    }
  }

  motorRunning = false;
  disableAllDrivers();
  Serial.println("OK:STOP");
}

void fixedSingle(int stepPin, int dirPin, int enPin, bool dir, int steps, int pulseDelay) {
  digitalWrite(enPin, DRIVER_ENABLE_LEVEL);
  digitalWrite(dirPin, dir ? HIGH : LOW);
  for (int i = 0; i < steps; i++) {
    pulsePin(stepPin, pulseDelay);
  }
  disableAllDrivers();
}

void fixedBoth(bool leftForward, bool rightForward, int steps, int pulseDelay) {
  enableAllDrivers();
  applyBothDirection(leftForward, rightForward);
  for (int i = 0; i < steps; i++) {
    pulseBoth(pulseDelay);
  }
  disableAllDrivers();
}

void runDemo() {
  Serial.println("DEMO:START");
  fixedSingle(RIGHT_STEP_PIN, RIGHT_DIR_PIN, RIGHT_EN_PIN, true, 1000, 700);
  delay(300);
  fixedSingle(RIGHT_STEP_PIN, RIGHT_DIR_PIN, RIGHT_EN_PIN, false, 1000, 700);
  delay(300);
  fixedSingle(LEFT_STEP_PIN, LEFT_DIR_PIN, LEFT_EN_PIN, false, 1000, 700);
  delay(300);
  fixedSingle(LEFT_STEP_PIN, LEFT_DIR_PIN, LEFT_EN_PIN, true, 1000, 700);
  delay(300);
  fixedBoth(true, false, 1400, 700);
  delay(300);
  fixedBoth(false, true, 1400, 700);
  delay(300);
  fixedBoth(true, true, 1400, 700);
  delay(300);
  fixedBoth(false, false, 1400, 700);
  Serial.println("DEMO:END");
}

String parseMoveDirection(String input) {
  if (input == "W" || input == "ROBOT_FWD") return "FWD";
  if (input == "S" || input == "ROBOT_BACK") return "BACK";
  if (input == "A") return "LEFT";
  if (input == "D") return "RIGHT";
  if (input == "SPIN_CW") return "CW";
  if (input == "SPIN_CCW") return "CCW";

  if (input.startsWith("MOVE ")) {
    String rest = input.substring(5);
    int firstSpace = rest.indexOf(' ');
    return firstSpace < 0 ? rest : rest.substring(0, firstSpace);
  }

  return "";
}

void handleCommand(String input) {
  input.trim();
  input.toUpperCase();
  if (input.length() == 0) return;

  if (input == "PING") {
    Serial.println("DEVICE:MOVE");
    return;
  }

  if (input == "MOVE PING") {
    lastKeepAlive = millis();
    return;
  }

  if (input == "MOVE STOP" || input == "STOP") {
    stopRequested = true;
    motorRunning = false;
    disableAllDrivers();
    Serial.println("OK:STOP");
    return;
  }

  if (input.startsWith("MOVE SPEED ")) {
    currentSpeed = constrain(input.substring(11).toInt(), 0, 100);
    Serial.print("OK:SPEED ");
    Serial.println(currentSpeed);
    return;
  }

  if (input == "LEFT_FWD") {
    fixedSingle(RIGHT_STEP_PIN, RIGHT_DIR_PIN, RIGHT_EN_PIN, true, 1000, 700);
    Serial.println("OK:LEFT_FWD");
    return;
  }
  if (input == "LEFT_BACK") {
    fixedSingle(RIGHT_STEP_PIN, RIGHT_DIR_PIN, RIGHT_EN_PIN, false, 1000, 700);
    Serial.println("OK:LEFT_BACK");
    return;
  }
  if (input == "RIGHT_FWD") {
    fixedSingle(LEFT_STEP_PIN, LEFT_DIR_PIN, LEFT_EN_PIN, false, 1000, 700);
    Serial.println("OK:RIGHT_FWD");
    return;
  }
  if (input == "RIGHT_BACK") {
    fixedSingle(LEFT_STEP_PIN, LEFT_DIR_PIN, LEFT_EN_PIN, true, 1000, 700);
    Serial.println("OK:RIGHT_BACK");
    return;
  }
  if (input == "DEMO") {
    runDemo();
    return;
  }

  String direction = parseMoveDirection(input);
  if (direction.length() == 0) {
    Serial.println("ERR:UNKNOWN");
    return;
  }

  stopRequested = false;
  lastKeepAlive = millis();

  if (direction == "FWD") {
    Serial.println("OK:MOVE_FWD");
    continuousBoth(false, true);
  } else if (direction == "BACK") {
    Serial.println("OK:MOVE_BACK");
    continuousBoth(true, false);
  } else if (direction == "LEFT") {
    Serial.println("OK:MOVE_LEFT");
    continuousSingle(RIGHT_STEP_PIN, RIGHT_DIR_PIN, RIGHT_EN_PIN, true);
  } else if (direction == "RIGHT") {
    Serial.println("OK:MOVE_RIGHT");
    continuousSingle(LEFT_STEP_PIN, LEFT_DIR_PIN, LEFT_EN_PIN, false);
  } else if (direction == "CW") {
    Serial.println("OK:MOVE_CW");
    continuousBoth(true, true);
  } else if (direction == "CCW") {
    Serial.println("OK:MOVE_CCW");
    continuousBoth(false, false);
  } else {
    Serial.println("ERR:UNKNOWN_DIR");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(LEFT_STEP_PIN, OUTPUT);
  pinMode(LEFT_DIR_PIN, OUTPUT);
  pinMode(LEFT_EN_PIN, OUTPUT);
  pinMode(RIGHT_STEP_PIN, OUTPUT);
  pinMode(RIGHT_DIR_PIN, OUTPUT);
  pinMode(RIGHT_EN_PIN, OUTPUT);

  digitalWrite(LEFT_STEP_PIN, LOW);
  digitalWrite(LEFT_DIR_PIN, LOW);
  digitalWrite(RIGHT_STEP_PIN, LOW);
  digitalWrite(RIGHT_DIR_PIN, LOW);

  disableAllDrivers();
  Serial.println("DEVICE:MOVE");
  Serial.println("MOVE VERSION 4 READY");
}

void loop() {
  if (queuedCommand.length() > 0) {
    String next = queuedCommand;
    queuedCommand = "";
    handleCommand(next);
    return;
  }

  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  command.trim();
  handleCommand(command);
}
