#include <Arduino.h>
#include <Wire.h>
// required for display
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// init oled dimensions
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// define pins for buttons
#define BTN_RESET_BASELINE D3 // to reset the baseline for drowsiness detection
#define BTN_RESET_STATS D4 // to reset the drowsiness score and attentiveness score

#define redPin D5
#define greenPin D6
#define yellowPin D7

// i2c address for ssd1306
#define OLED_ADDR 0x3C

// display object
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// store current status and scores
String currentStatus = "NO DATA";
float attentiveness = 0.0;
float drowsyScore = 0.0;

// store last button states
bool lastBaselineState = HIGH;
bool lastStatBtnState = HIGH;

// store last button press times for debouncing
unsigned long lastBaselinePress = 0;
unsigned long lastStatBtnPress = 0;
const unsigned long debounceMs = 250;

// led state modes
enum LedMode {
  LED_OFF,
  LED_BASELINE,
  LED_ALERT,
  LED_WARNING,
  LED_DROWSY
};

LedMode ledMode = LED_OFF;
unsigned long lastLedToggle = 0;
bool ledOn = false;

// turn all leds off
void offLeds() {
  digitalWrite(redPin, LOW);
  digitalWrite(greenPin, LOW);
  digitalWrite(yellowPin, LOW);
}

// change led behavior based on current driver status
void setLedMode(LedMode mode) {
  if (ledMode == mode) {
    return;
  }

  ledMode = mode;
  ledOn = false;
  lastLedToggle = millis();

  offLeds();

  if (ledMode == LED_ALERT) {
    digitalWrite(greenPin, HIGH);
  }
  else if (ledMode == LED_BASELINE) {
    digitalWrite(yellowPin, HIGH);
  }
}

// update blinking leds without blocking button or serial reads
void updateLeds() {
  if (ledMode == LED_ALERT || ledMode == LED_OFF || ledMode == LED_BASELINE) {
    return;
  }

  unsigned long now = millis();

  int activePin = yellowPin;
  unsigned long interval = 250;

  if (ledMode == LED_WARNING) {
    activePin = yellowPin;
    interval = 250;
  } 
  else if (ledMode == LED_DROWSY) {
    activePin = redPin;
    interval = 100;
  }

  if (now - lastLedToggle >= interval) {
    lastLedToggle = now;
    ledOn = !ledOn;

    offLeds();
    digitalWrite(activePin, ledOn ? HIGH : LOW);
  }
}

// show current status and scores on the display
void drawDisplay() {
  display.clearDisplay();

  display.setTextColor(SSD1306_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("DashSentinel");

  display.setCursor(0, 16);
  display.print("Status:");

  display.setTextSize(2);
  display.setCursor(0, 24);

  if (currentStatus == "LEARNING BASELINE") {
    display.print("LEARNING");
  }
  else if (currentStatus == "VISION DEGRADED") {
    display.print("VISION");
    display.setTextSize(1);
    display.setCursor(0, 42);
    display.print("DEGRADED");
  }
  else {
    display.println(currentStatus);
  }

  display.setTextSize(1);
  display.setCursor(0, 48);
  display.print("Attention: ");
  display.print(attentiveness, 1);
  display.print("%");

  display.setCursor(0, 57);
  display.print("Drowsy: ");
  display.print(drowsyScore, 2);

  display.display();
}

// parse incoming serial data once in format "status,attentiveness,drowsyScore"
void parseLine(String line) {
  line.trim();

  int firstComma = line.indexOf(',');
  int secondComma = line.indexOf(',', firstComma + 1);

  if (firstComma < 0 || secondComma < 0) {
    return;
  }

  currentStatus = line.substring(0, firstComma);
  attentiveness = line.substring(firstComma + 1, secondComma).toFloat();
  drowsyScore = line.substring(secondComma + 1).toFloat();

  if (currentStatus == "ALERT") {
    setLedMode(LED_ALERT);
  }
  else if (currentStatus == "LEARNING BASELINE") {
    setLedMode(LED_BASELINE);
  }
  else if (currentStatus == "WARNING" || currentStatus == "VISION DEGRADED") {
    setLedMode(LED_WARNING);
  } 
  else if (currentStatus == "DROWSY") {
    setLedMode(LED_DROWSY);
  } 
  else {
    setLedMode(LED_OFF);
  }

  drawDisplay();
}

// check button states and print to serial if pressed
void checkButtons() {
  bool baselineState = digitalRead(BTN_RESET_BASELINE);
  bool statBtnState = digitalRead(BTN_RESET_STATS);
  unsigned long now = millis();

  if (lastBaselineState == HIGH && baselineState == LOW && (now - lastBaselinePress > debounceMs)) {
    Serial.println("BTN_RESET_BASELINE");
    lastBaselinePress = now;
  }

  if (lastStatBtnState == HIGH && statBtnState == LOW && (now - lastStatBtnPress > debounceMs)) {
    Serial.println("BTN_RESET_STATS");
    lastStatBtnPress = now;
  }

  lastBaselineState = baselineState;
  lastStatBtnState = statBtnState;
}

void setup() {
  // using internal pullup resistors for buttons
  // so they read high when not pressed and low when pressed
  pinMode(BTN_RESET_BASELINE, INPUT_PULLUP);
  pinMode(BTN_RESET_STATS, INPUT_PULLUP);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(yellowPin, OUTPUT);

  offLeds();

  Serial.begin(115200);
  Wire.begin();

  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    while (true) {
      delay(1000);
    }
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("DashSentinel");
  display.println("waiting for data...");
  display.display();
}

void loop() {
  checkButtons();

  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    parseLine(line);
  }

  updateLeds();
}