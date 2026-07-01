#include <Arduino.h>
#include <ESP32Servo.h>
#include <Adafruit_NeoPixel.h>

#define BAUD_RATE 115200

#define LED_PIN 4
#define LED_COUNT 84
#define LED_BRIGHTNESS 60

#define HEAD_SERVO_PIN 17
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500

Adafruit_NeoPixel strip(LED_COUNT, LED_PIN, NEO_RGB + NEO_KHZ800);
Servo headServo;

uint8_t brightnessValue = LED_BRIGHTNESS;

int currentHeadAngle = 90;
const int HEAD_MIN_ANGLE = 0;
const int HEAD_MAX_ANGLE = 180;
const int HEAD_CENTER_ANGLE = 90;

void setAll(uint8_t r, uint8_t g, uint8_t b) {
  for (int i = 0; i < LED_COUNT; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }
  strip.show();
}

void ledOff() {
  strip.clear();
  strip.show();
}

bool parseThreeInts(String text, int startIndex, int &a, int &b, int &c) {
  text = text.substring(startIndex);
  text.trim();

  int p1 = text.indexOf(' ');
  int p2 = text.indexOf(' ', p1 + 1);

  if (p1 < 0 || p2 < 0) return false;

  a = constrain(text.substring(0, p1).toInt(), 0, 255);
  b = constrain(text.substring(p1 + 1, p2).toInt(), 0, 255);
  c = constrain(text.substring(p2 + 1).toInt(), 0, 255);

  return true;
}

void servoWriteAngle(int angle) {
  currentHeadAngle = constrain(angle, HEAD_MIN_ANGLE, HEAD_MAX_ANGLE);
  headServo.write(currentHeadAngle);
}

void servoStop() {
  servoWriteAngle(currentHeadAngle);
}

void servoLeft(int step = 15) {
  servoWriteAngle(currentHeadAngle - constrain(step, 1, 90));
}

void servoRight(int step = 15) {
  servoWriteAngle(currentHeadAngle + constrain(step, 1, 90));
}

void rainbow(int waitMs) {
  for (long hue = 0; hue < 65536; hue += 256) {
    if (Serial.available()) return;

    for (int i = 0; i < LED_COUNT; i++) {
      int pixelHue = hue + (i * 65536L / LED_COUNT);
      strip.setPixelColor(i, strip.gamma32(strip.ColorHSV(pixelHue)));
    }

    strip.show();
    delay(waitMs);
  }
}

void printReady() {
  Serial.println("DEVICE:HEAD");
  Serial.println("HEAD READY");
  Serial.println("Commands:");
  Serial.println("PING");
  Serial.println("HEAD LEFT [step]");
  Serial.println("HEAD RIGHT [step]");
  Serial.println("HEAD STOP");
  Serial.println("HEAD CENTER");
  Serial.println("HEAD SERVO 90");
  Serial.println("HEAD LED 255 0 0");
  Serial.println("HEAD LED_OFF");
  Serial.println("HEAD RAINBOW");
}

void handleCommand(String command) {
  command.trim();
  if (command.length() == 0) return;

  String upper = command;
  upper.toUpperCase();

  String lower = command;
  lower.toLowerCase();

  if (upper == "PING") {
    Serial.println("DEVICE:HEAD");
    return;
  }

  if (upper.startsWith("HEAD LEFT")) {
    int step = command.substring(9).toInt();
    if (step <= 0) step = 15;
    servoLeft(step);
    Serial.print("OK:HEAD_LEFT ");
    Serial.println(currentHeadAngle);
    return;
  }

  if (upper.startsWith("HEAD RIGHT")) {
    int step = command.substring(10).toInt();
    if (step <= 0) step = 15;
    servoRight(step);
    Serial.print("OK:HEAD_RIGHT ");
    Serial.println(currentHeadAngle);
    return;
  }

  if (upper == "HEAD STOP") {
    servoStop();
    Serial.print("OK:HEAD_HOLD ");
    Serial.println(currentHeadAngle);
    return;
  }

  if (upper == "HEAD CENTER" || upper == "HEAD NEUTRAL") {
    servoWriteAngle(HEAD_CENTER_ANGLE);
    Serial.println("OK:HEAD_CENTER 90");
    return;
  }

  if (upper.startsWith("HEAD NEUTRAL ")) {
    int angle = command.substring(13).toInt();
    servoWriteAngle(angle);
    Serial.print("OK:HEAD_NEUTRAL ");
    Serial.println(currentHeadAngle);
    return;
  }

  if (upper.startsWith("HEAD SERVO ")) {
    int angle = command.substring(11).toInt();
    servoWriteAngle(angle);
    Serial.print("OK:HEAD_SERVO ");
    Serial.println(currentHeadAngle);
    return;
  }

  if (upper.startsWith("HEAD LED ")) {
    int r, g, b;

    if (!parseThreeInts(command, 9, r, g, b)) {
      Serial.println("ERR:LED_FORMAT");
      return;
    }

    setAll(r, g, b);
    Serial.println("OK:LED");
    return;
  }

  if (upper == "HEAD LED_OFF") {
    ledOff();
    Serial.println("OK:LED_OFF");
    return;
  }

  if (upper == "HEAD RAINBOW") {
    rainbow(5);
    Serial.println("OK:RAINBOW");
    return;
  }

  if (lower == "red") {
    setAll(255, 0, 0);
    Serial.println("OK:RED");
  } else if (lower == "green") {
    setAll(0, 255, 0);
    Serial.println("OK:GREEN");
  } else if (lower == "blue") {
    setAll(0, 0, 255);
    Serial.println("OK:BLUE");
  } else if (lower == "white") {
    setAll(255, 255, 255);
    Serial.println("OK:WHITE");
  } else if (lower == "off") {
    ledOff();
    Serial.println("OK:OFF");
  } else if (lower == "rainbow") {
    rainbow(5);
    Serial.println("OK:RAINBOW");
  } else if (lower.startsWith("brightness")) {
    int value = constrain(command.substring(10).toInt(), 0, 255);
    brightnessValue = value;
    strip.setBrightness(brightnessValue);
    strip.show();

    Serial.print("OK:BRIGHTNESS ");
    Serial.println(brightnessValue);
  } else {
    Serial.println("ERR:UNKNOWN_HEAD_COMMAND");
  }
}

void setup() {
  Serial.begin(BAUD_RATE);
  delay(1000);

  ESP32PWM::allocateTimer(0);

  headServo.setPeriodHertz(50);
  headServo.attach(HEAD_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);

  servoWriteAngle(HEAD_CENTER_ANGLE);

  strip.begin();
  strip.setBrightness(brightnessValue);
  ledOff();

  printReady();
}

void loop() {
  if (!Serial.available()) return;

  String command = Serial.readStringUntil('\n');
  handleCommand(command);
}
