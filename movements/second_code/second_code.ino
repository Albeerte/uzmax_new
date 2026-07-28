// ============================================================
// UzMAX MOVE CONTROLLER - SECOND CODE ESP32
// Hold-to-move control for 2 stepper motors.
//
// Dashboard / Serial protocol:
//   PING              -> DEVICE:MOVE
//   MOVE SPEED 60     -> speed 0..100
//   MOVE FWD          -> both motors forward
//   MOVE BACK         -> both motors backward
//   MOVE LEFT         -> left motor backward, right motor forward
//   MOVE RIGHT        -> left motor forward, right motor backward
//   MOVE PING         -> keepalive while button/key is held
//   MOVE STOP         -> stop when button/key is released
//
// Short aliases:
//   W, S, A, D, STOP
//   ROBOT_FWD, ROBOT_BACK, SPIN_CW, SPIN_CCW
//
// Speed tuning:
//   SPEED 80
//   MOVE SPEED 80
//   MOVE FAST 220     -> fastest pulse delay in microseconds
//   MOVE SLOW 3000    -> slowest pulse delay in microseconds
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

// If one motor turns opposite to what you expect, flip only that value.
// true means logical forward writes HIGH to DIR pin.
#define LEFT_FORWARD_HIGH   true
#define RIGHT_FORWARD_HIGH  false

// Lower pulse delay = faster motor.
int pulseSlowUs = 3000;
int pulseFastUs = 220;

#define ACCEL_STEPS          300
#define DECEL_STEPS          140
#define SERIAL_CHECK_STEPS     8
#define KEEPALIVE_TIMEOUT_MS 350

String queuedCommand = "";
volatile bool stopRequested = false;
bool motorRunning = false;
int currentSpeed = 60;
unsigned long lastKeepAlive = 0;

int speedToPulseDelay(int speed) {
  speed = constrain(speed, 0, 100);
  return map(speed, 0, 100, pulseSlowUs, pulseFastUs);
}

float smoothStep(float t) {
  t = constrain(t, 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

int rampDelay(int stepIndex, int targetDelay) {
  if (stepIndex >= ACCEL_STEPS) return targetDelay;
  float t = smoothStep((float)stepIndex / ACCEL_STEPS);
  return (int)(pulseSlowUs - t * (pulseSlowUs - targetDelay));
}

int stopDelay(int stepIndex, int startDelay) {
  if (stepIndex >= DECEL_STEPS) return pulseSlowUs;
  float t = smoothStep((float)stepIndex / DECEL_STEPS);
  return (int)(startDelay + t * (pulseSlowUs - startDelay));
}

void enableAllDrivers() {
  digitalWrite(LEFT_EN_PIN, DRIVER_ENABLE_LEVEL);
  digitalWrite(RIGHT_EN_PIN, DRIVER_ENABLE_LEVEL);
}

void disableAllDrivers() {
  digitalWrite(LEFT_EN_PIN, DRIVER_DISABLE_LEVEL);
  digitalWrite(RIGHT_EN_PIN, DRIVER_DISABLE_LEVEL);
}

void setLeftDirection(bool forward) {
  digitalWrite(LEFT_DIR_PIN, forward == LEFT_FORWARD_HIGH ? HIGH : LOW);
}

void setRightDirection(bool forward) {
  digitalWrite(RIGHT_DIR_PIN, forward == RIGHT_FORWARD_HIGH ? HIGH : LOW);
}

void applyDirection(bool leftForward, bool rightForward) {
  setLeftDirection(leftForward);
  setRightDirection(rightForward);
}

void pulseBoth(int delayUs) {
  digitalWrite(LEFT_STEP_PIN, HIGH);
  digitalWrite(RIGHT_STEP_PIN, HIGH);
  delayMicroseconds(delayUs);
  digitalWrite(LEFT_STEP_PIN, LOW);
  digitalWrite(RIGHT_STEP_PIN, LOW);
  delayMicroseconds(delayUs);
}

void printSpeed() {
  Serial.print("OK:SPEED ");
  Serial.println(currentSpeed);
}

void printTiming() {
  Serial.print("OK:TIMING FAST=");
  Serial.print(pulseFastUs);
  Serial.print(" SLOW=");
  Serial.println(pulseSlowUs);
}

void setSpeedValue(int speed) {
  currentSpeed = constrain(speed, 0, 100);
  printSpeed();
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

  if (incoming == "MOVE STOP" || incoming == "STOP") {
    stopRequested = true;
    return;
  }

  if (incoming.startsWith("MOVE SPEED ")) {
    setSpeedValue(incoming.substring(11).toInt());
    return;
  }

  if (incoming.startsWith("SPEED ")) {
    setSpeedValue(incoming.substring(6).toInt());
    return;
  }

  if (incoming.startsWith("MOVE FAST ")) {
    pulseFastUs = constrain(incoming.substring(10).toInt(), 80, 3000);
    if (pulseFastUs > pulseSlowUs) pulseFastUs = pulseSlowUs;
    printTiming();
    return;
  }

  if (incoming.startsWith("MOVE SLOW ")) {
    pulseSlowUs = constrain(incoming.substring(10).toInt(), 200, 8000);
    if (pulseSlowUs < pulseFastUs) pulseSlowUs = pulseFastUs;
    printTiming();
    return;
  }

  queuedCommand = incoming;
  stopRequested = true;
}

String parseMoveDirection(String input) {
  if (input == "W" || input == "ROBOT_FWD") return "FWD";
  if (input == "S" || input == "ROBOT_BACK") return "BACK";
  if (input == "A" || input == "SPIN_CCW") return "LEFT";
  if (input == "D" || input == "SPIN_CW") return "RIGHT";

  if (input.startsWith("MOVE ")) {
    String rest = input.substring(5);
    int firstSpace = rest.indexOf(' ');
    return firstSpace < 0 ? rest : rest.substring(0, firstSpace);
  }

  return "";
}

void continuousMove(bool leftForward, bool rightForward, const char *startedMessage) {
  enableAllDrivers();
  applyDirection(leftForward, rightForward);

  motorRunning = true;
  stopRequested = false;
  lastKeepAlive = millis();
  Serial.println(startedMessage);

  int accelStep = 0;
  int decelStep = 0;
  int serialStep = 0;
  int currentDelay = pulseSlowUs;
  bool decelerating = false;

  while (true) {
    int targetDelay = speedToPulseDelay(currentSpeed);

    if (!decelerating) {
      currentDelay = rampDelay(accelStep++, targetDelay);
    }

    if (stopRequested) {
      decelerating = true;
    }

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
    Serial.println("OK:PING");
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
    setSpeedValue(input.substring(11).toInt());
    return;
  }

  if (input.startsWith("SPEED ")) {
    setSpeedValue(input.substring(6).toInt());
    return;
  }

  if (input.startsWith("MOVE FAST ")) {
    pulseFastUs = constrain(input.substring(10).toInt(), 80, 3000);
    if (pulseFastUs > pulseSlowUs) pulseFastUs = pulseSlowUs;
    printTiming();
    return;
  }

  if (input.startsWith("MOVE SLOW ")) {
    pulseSlowUs = constrain(input.substring(10).toInt(), 200, 8000);
    if (pulseSlowUs < pulseFastUs) pulseSlowUs = pulseFastUs;
    printTiming();
    return;
  }

  String direction = parseMoveDirection(input);
  if (direction.length() == 0) {
    Serial.println("ERR:UNKNOWN");
    return;
  }

  if (direction == "FWD") {
    // Both motors forward.
    continuousMove(true, true, "OK:MOVE_FWD");
  } else if (direction == "BACK") {
    // Both motors backward.
    continuousMove(false, false, "OK:MOVE_BACK");
  } else if (direction == "LEFT") {
    // Left turn: left motor backward, right motor forward.
    continuousMove(false, true, "OK:MOVE_LEFT");
  } else if (direction == "RIGHT") {
    // Right turn: left motor forward, right motor backward.
    continuousMove(true, false, "OK:MOVE_RIGHT");
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
  digitalWrite(RIGHT_STEP_PIN, LOW);
  digitalWrite(LEFT_DIR_PIN, LOW);
  digitalWrite(RIGHT_DIR_PIN, LOW);

  disableAllDrivers();
  Serial.println("DEVICE:MOVE");
  Serial.println("MOVE SECOND CODE READY");
  printSpeed();
  printTiming();
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
