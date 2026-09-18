/*
 * ESP32 on-device Human Activity Recognition (AIoT course project)
 *
 * Sensing pipeline
 *   MPU6050 (I2C, polling) -> fixed 50 Hz sampler (micros() deadline scheduler)
 *   -> ring buffer -> 128-sample window (2.56 s) with hop 64 (50 % overlap)
 *   -> per-channel standardisation (train-set mean/std, see har_config.h)
 *   -> TFLite-Micro inference (float32 baseline AND int8 PTQ model, both run on
 *      every window so latency / arena usage can be compared on the same input)
 *
 * Modes (toggle with the push button on GPIO 4, or serial 'r'/'l'):
 *   LIVE   - classify the real-time MPU6050 stream
 *   REPLAY - feed recorded windows from held-out TEST subjects through the
 *            identical preprocessing + inference path (ground truth is known,
 *            so the on-device accuracy of both models can be verified)
 *
 * Concurrency: the ESP32 has two cores. Sampling runs in loop() (core 1) and
 * inference runs in a FreeRTOS task pinned to core 0, so a slow inference never
 * delays the 50 Hz sampler. The IMU and the OLED sit on separate I2C controllers
 * (Wire / Wire1) so a ~25 ms display refresh can never block a sensor read.
 *
 * Serial output is CSV-like so training/05_parse_device_log.py can turn the
 * log into the numbers reported in the report.
 */
#include <Wire.h>
// ---- optional add-on: publish each result over MQTT (publish-subscribe) ----
// Set to 0 for a fully offline build; nothing else changes (sampling/inference are unaffected).
#define ENABLE_MQTT 1
#if ENABLE_MQTT
#include <WiFi.h>
#include <PubSubClient.h>
#endif
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "har_config.h"
#include "model_data.h"
#include "test_windows.h"

// (in a header: the Arduino preprocessor hoists prototypes above this file's type definitions)
#include "har_types.h"

// ---------------- pins / peripherals ----------------
static const int PIN_SDA = 21, PIN_SCL = 22;          // I2C bus 0: MPU6050 only (sampler, core 1)
static const int PIN_OLED_SDA = 25, PIN_OLED_SCL = 26; // I2C bus 1: OLED only (inference task, core 0)
static const int PIN_BUTTON = 4, PIN_LED = 2;
// One "stimulus" button per class: injects a recorded held-out window of that class into the
// on-device pipeline (same preprocessing + inference path). NOT live sensor data - see README.
static const int CLASS_PINS[HAR_N_CLASSES] = {32, 33, 27, 14, 12, 13};
static const uint8_t MPU_ADDR = 0x68, OLED_ADDR = 0x3C;
Adafruit_SSD1306 oled(128, 64, &Wire1, -1);

// ---------------- MPU6050 minimal register driver ----------------
// Configured for +/-2 g (16384 LSB/g) and +/-250 deg/s (131 LSB/(deg/s)),
// digital low-pass filter 21 Hz (DLPF_CFG=4): anti-aliasing below fs/2 = 25 Hz.
static bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}
static bool mpuInit() {
  if (!mpuWrite(0x6B, 0x00)) return false;   // PWR_MGMT_1: wake up
  mpuWrite(0x1A, 0x04);                      // CONFIG: DLPF 21 Hz
  mpuWrite(0x1B, 0x00);                      // GYRO_CONFIG: +/-250 dps
  mpuWrite(0x1C, 0x00);                      // ACCEL_CONFIG: +/-2 g
  return true;
}
// Returns one sample in dataset units/order: ax ay az [g], gx gy gz [rad/s].
// Axis mapping: the dataset's X axis is "vertical when upright" (mean +0.8 g), so the
// sensor is mounted with its Z axis along that direction: dataset(x,y,z)=sensor(z,x,y).
static bool mpuRead(float out[HAR_N_CH]) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) { r[i] = (int16_t)((Wire.read() << 8) | Wire.read()); }
  const float ax = r[0] / 16384.0f, ay = r[1] / 16384.0f, az = r[2] / 16384.0f;
  const float d2r = 3.14159265f / 180.0f;
  const float gx = r[4] / 131.0f * d2r, gy = r[5] / 131.0f * d2r, gz = r[6] / 131.0f * d2r;
  out[0] = az; out[1] = ax; out[2] = ay;
  out[3] = gz; out[4] = gx; out[5] = gy;
  return true;
}

// ---------------- TFLite Micro: two models ----------------
namespace {
tflite::MicroErrorReporter errReporter;
using Resolver = tflite::MicroMutableOpResolver<8>;
Resolver resolver;
constexpr int kArenaF32 = 24 * 1024;
constexpr int kArenaI8  = 12 * 1024;
alignas(16) uint8_t arenaF32[kArenaF32];
alignas(16) uint8_t arenaI8[kArenaI8];
tflite::MicroInterpreter* interpF32 = nullptr;
tflite::MicroInterpreter* interpI8 = nullptr;
}

static tflite::MicroInterpreter* makeInterpreter(const unsigned char* data, uint8_t* arena, int arenaSize, const char* tag) {
  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { Serial.printf("ERR schema %s\r\n", tag); return nullptr; }
  auto* it = new tflite::MicroInterpreter(model, resolver, arena, arenaSize, &errReporter);
  if (it->AllocateTensors() != kTfLiteOk) { Serial.printf("ERR alloc %s\r\n", tag); return nullptr; }
  Serial.printf("MODEL,%s,flash_bytes=%u,arena_used_bytes=%u,arena_size=%d\r\n", tag,
                tag[0]=='f' ? har_float32_tflite_len : har_int8_tflite_len, (unsigned)it->arena_used_bytes(), arenaSize);
  return it;
}

// window: [HAR_WINDOW][HAR_N_CH] raw units. Standardisation + (de)quantisation happen here,
// i.e. identical preprocessing for LIVE and REPLAY inputs.
static Result runModel(tflite::MicroInterpreter* it, const float window[HAR_WINDOW][HAR_N_CH]) {
  TfLiteTensor* in = it->input(0);
  const bool q = (in->type == kTfLiteInt8);
  for (int t = 0; t < HAR_WINDOW; t++)
    for (int c = 0; c < HAR_N_CH; c++) {
      float v = (window[t][c] - HAR_MEAN[c]) / HAR_STD[c];
      if (q) {
        int32_t qv = (int32_t)lroundf(v / in->params.scale) + in->params.zero_point;
        in->data.int8[t * HAR_N_CH + c] = (int8_t)constrain(qv, -128, 127);
      } else in->data.f[t * HAR_N_CH + c] = v;
    }
  uint32_t t0 = micros();
  it->Invoke();
  uint32_t lat = micros() - t0;
  TfLiteTensor* out = it->output(0);
  Result r{0, 0.f, lat};
  for (int k = 0; k < HAR_N_CLASSES; k++) {
    float p = q ? (out->data.int8[k] - out->params.zero_point) * out->params.scale : out->data.f[k];
    if (p > r.conf) { r.conf = p; r.cls = k; }
  }
  return r;
}

// ---------------- sampler / windowing ----------------
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / HAR_FS_HZ;
static float ring[HAR_WINDOW][HAR_N_CH];
static int ringHead = 0, ringCount = 0, sinceLastInfer = 0;
static uint32_t nextSampleUs = 0, rateWindowStart = 0;
static int rateCount = 0, missedDeadlines = 0; static float measuredHz = 0;
static float window[HAR_WINDOW][HAR_N_CH];

static void copyRingToWindow() {
  for (int i = 0; i < HAR_WINDOW; i++) {
    int idx = (ringHead + i) % HAR_WINDOW;           // ringHead == oldest sample
    for (int c = 0; c < HAR_N_CH; c++) window[i][c] = ring[idx][c];
  }
}

// ---------------- UI ----------------
enum Mode { LIVE, REPLAY }; Mode mode = LIVE;
static int replayIdx = 0; static uint32_t lastReplayMs = 0;
static int correctF32 = 0, correctI8 = 0; static uint64_t sumLatF32 = 0, sumLatI8 = 0; static int nReplay = 0;

static void showOled(const char* line1, const Result& f, const Result& q, const char* extra) {
  oled.clearDisplay(); oled.setTextSize(1); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0, 0);
  oled.println(line1);
  oled.setTextSize(2); oled.println(HAR_CLASSES[q.cls]); oled.setTextSize(1);
  oled.printf("int8  %3.0f%% %5lu us\r\n", q.conf * 100, (unsigned long)q.latency_us);
  oled.printf("f32   %3.0f%% %5lu us\r\n", f.conf * 100, (unsigned long)f.latency_us);
  oled.println(extra); oled.display();
}

// ---------------- MQTT add-on ----------------
#if ENABLE_MQTT
static const char* WIFI_SSID = "Wokwi-GUEST", *WIFI_PASS = "";   // Wokwi's simulated open AP
static const char* MQTT_HOST = "broker.hivemq.com"; static const int MQTT_PORT = 1883;
WiFiClient wifiClient; PubSubClient mqtt(wifiClient);
char mqttTopic[48];                       // aiot/har/<chip-id>/activity
static char mqttPending[192]; static volatile bool mqttHasPending = false;   // filled by core 0, sent by core 1
static uint32_t mqttLastTry = 0; static int mqttPublished = 0;

static void mqttSetup() {
  uint32_t id = (uint32_t)(ESP.getEfuseMac() & 0xFFFFFF);
  snprintf(mqttTopic, sizeof mqttTopic, "aiot/har/%06lx/activity", (unsigned long)id);
  WiFi.begin(WIFI_SSID, WIFI_PASS, 6);
  Serial.printf("WIFI,connecting to %s\r\n", WIFI_SSID);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);
  Serial.printf("WIFI,%s,ip=%s\r\n", WiFi.status() == WL_CONNECTED ? "ok" : "FAILED", WiFi.localIP().toString().c_str());
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  Serial.printf("MQTT,broker=%s:%d,topic=%s\r\n", MQTT_HOST, MQTT_PORT, mqttTopic);
}
// Called ONLY from the inference task on core 0 (PubSubClient is not thread-safe). A (re)connect can
// block for ~1 s (DNS + TCP), which must never happen on the sampler core: with the first version of
// this code in loop() it cost 36 missed sample deadlines.
static void mqttService() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (!mqtt.connected()) {
    if (millis() - mqttLastTry < 3000) return;
    mqttLastTry = millis();
    char cid[24]; snprintf(cid, sizeof cid, "esp32-har-%lx", (unsigned long)random(0xFFFFFF));
    if (mqtt.connect(cid)) Serial.print("MQTT,connected\r\n");
    return;
  }
  mqtt.loop();
  if (mqttHasPending) {
    mqtt.publish(mqttTopic, mqttPending);   // QoS 0, ~150 B per window (vs 1536 B of raw IMU per window)
    mqttHasPending = false; mqttPublished++;
  }
}
#endif

// ---------------- inference task (core 0) ----------------
TaskHandle_t inferTask = nullptr;
static volatile bool inferBusy = false;
static int droppedWindows = 0;       // windows skipped because inference was still running
struct Job { char src[24]; int label; int subject; int idx; };
static Job job;

static void classifyAndReport(const char* src, int label, int subject) {
  digitalWrite(PIN_LED, HIGH);
  Result f = runModel(interpF32, window);
  Result q = runModel(interpI8, window);
  digitalWrite(PIN_LED, LOW);
  char extra[32];
  if (label >= 0) {
    if (job.src[0] != 'B') { nReplay++; correctF32 += (f.cls == label); correctI8 += (q.cls == label);
                             sumLatF32 += f.latency_us; sumLatI8 += q.latency_us; }
    Serial.printf("%s,%d,subj=%d,label=%s,f32=%s(%s),f32_conf=%.3f,f32_us=%lu,int8=%s(%s),int8_conf=%.3f,int8_us=%lu\r\n",
                  job.src[0] == 'B' ? "INJECT" : "REPLAY", job.idx, subject, HAR_CLASSES[label], HAR_CLASSES[f.cls], f.cls == label ? "ok" : "WRONG", f.conf, (unsigned long)f.latency_us,
                  HAR_CLASSES[q.cls], q.cls == label ? "ok" : "WRONG", q.conf, (unsigned long)q.latency_us);
    snprintf(extra, sizeof extra, "truth:%.10s", HAR_CLASSES[label]);
  } else {
    float m[3] = {0, 0, 0};
    for (int t = 0; t < HAR_WINDOW; t++) for (int c = 0; c < 3; c++) m[c] += window[t][c] / HAR_WINDOW;
    Serial.printf("LIVE,rate=%.1fHz,missed=%d,dropped=%d,acc_mean_g=[%.2f %.2f %.2f],f32=%s,f32_conf=%.3f,f32_us=%lu,int8=%s,int8_conf=%.3f,int8_us=%lu\r\n",
                  measuredHz, missedDeadlines, droppedWindows, m[0], m[1], m[2], HAR_CLASSES[f.cls], f.conf, (unsigned long)f.latency_us,
                  HAR_CLASSES[q.cls], q.conf, (unsigned long)q.latency_us);
    snprintf(extra, sizeof extra, "fs=%.1fHz miss=%d", measuredHz, missedDeadlines);
  }
  showOled(src, f, q, extra);
#if ENABLE_MQTT
  if (!mqttHasPending) {
    snprintf(mqttPending, sizeof mqttPending,
      "{\"src\":\"%s\",\"truth\":\"%s\",\"int8\":\"%s\",\"int8_conf\":%.2f,\"int8_us\":%lu,\"f32\":\"%s\",\"f32_conf\":%.2f,\"f32_us\":%lu,\"rate_hz\":%.1f}",
      src, label >= 0 ? HAR_CLASSES[label] : "-", HAR_CLASSES[q.cls], q.conf, (unsigned long)q.latency_us,
      HAR_CLASSES[f.cls], f.conf, (unsigned long)f.latency_us, measuredHz);
    mqttHasPending = true;
  }
#endif
}

static void inferTaskFn(void*) {
  for (;;) {
    // wait up to 20 ms for a window; in between, service the network (core 0 only)
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20)) > 0) {
      classifyAndReport(job.src, job.label, job.subject);
      inferBusy = false;
    }
#if ENABLE_MQTT
    mqttService();
#endif
  }
}
// Hand the current `window` buffer to core 0. Returns false if inference is still busy.
static bool submitWindow(const char* src, int label, int subject, int idx = -1) {
  if (inferBusy) return false;
  inferBusy = true;
  snprintf(job.src, sizeof job.src, "%s", src); job.label = label; job.subject = subject; job.idx = idx;
  xTaskNotifyGive(inferTask);
  return true;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_BUTTON, INPUT_PULLUP); pinMode(PIN_LED, OUTPUT);
  for (int k = 0; k < HAR_N_CLASSES; k++) pinMode(CLASS_PINS[k], INPUT_PULLUP);
  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  Wire1.begin(PIN_OLED_SDA, PIN_OLED_SCL, 400000);
  oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
  oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0, 0); oled.println("HAR boot..."); oled.display();
  Serial.printf("\nBOOT,fs=%dHz,window=%d,hop=%d,free_heap=%u\r\n", HAR_FS_HZ, HAR_WINDOW, HAR_HOP, ESP.getFreeHeap());
  Serial.printf("SENSOR,mpu6050=%s\r\n", mpuInit() ? "ok" : "MISSING");

  resolver.AddExpandDims(); resolver.AddConv2D(); resolver.AddReshape(); resolver.AddAdd();
  resolver.AddAveragePool2D(); resolver.AddMean(); resolver.AddFullyConnected(); resolver.AddSoftmax();
  interpF32 = makeInterpreter(har_float32_tflite, arenaF32, kArenaF32, "float32");
  interpI8  = makeInterpreter(har_int8_tflite,   arenaI8,  kArenaI8,  "int8");
  Serial.printf("HEAP,after_alloc=%u\r\n", ESP.getFreeHeap());
  xTaskCreatePinnedToCore(inferTaskFn, "infer", 8192, nullptr, 1, &inferTask, 0);   // core 0
  Serial.printf("TASK,inference_core=0,sampler_core=%d\r\n", xPortGetCoreID());
  Serial.print("\r\n=== LIVE mode: real-time MPU6050 stream (sensing pipeline evidence). Drag the accel sliders to change posture.\r\n");
  Serial.print("=== Class buttons WALK/UP/DOWN/SIT/STAND/LAY inject one recorded window of that class (INJECT lines).\r\n");
  Serial.print("=== Press MODE (or send 'r') for REPLAY: 18 recorded windows from unseen subjects, ground truth known (model evidence).\r\n\r\n");
#if ENABLE_MQTT
  mqttSetup();
#endif
  nextSampleUs = rateWindowStart = micros();
}

void loop() {
  // --- button: toggle mode (debounced) ---
  static uint32_t lastBtnMs = 0; static int lastBtn = HIGH;
  int b = digitalRead(PIN_BUTTON);
  if (b != lastBtn && millis() - lastBtnMs > 200) {
    lastBtnMs = millis();
    if (b == LOW) {
      mode = (mode == LIVE) ? REPLAY : LIVE;
      Serial.printf("MODE,%s\r\n", mode == LIVE ? "LIVE" : "REPLAY");
      if (mode == REPLAY) { replayIdx = 0; nReplay = correctF32 = correctI8 = 0; sumLatF32 = sumLatI8 = 0; lastReplayMs = 0; }
    }
  }
  lastBtn = b;
  // class buttons: inject the next recorded window of that class (cycles through 3 per class)
  static int lastCls[HAR_N_CLASSES] = {HIGH, HIGH, HIGH, HIGH, HIGH, HIGH}; static int nextOfClass[HAR_N_CLASSES] = {0};
  for (int k = 0; k < HAR_N_CLASSES; k++) {
    int v = digitalRead(CLASS_PINS[k]);
    if (v == LOW && lastCls[k] == HIGH && !inferBusy) {
      int hit = 0, idx = -1;
      for (int i = 0; i < N_TEST_WINDOWS; i++) if (TEST_LABELS[i] == k && hit++ == nextOfClass[k]) { idx = i; break; }
      if (idx >= 0) {
        nextOfClass[k] = (nextOfClass[k] + 1) % 3;
        memcpy_P(window, TEST_WINDOWS[idx], sizeof window);
        char src[24]; snprintf(src, sizeof src, "BTN %s", HAR_CLASSES[k]);
        submitWindow(src, TEST_LABELS[idx], TEST_SUBJECTS[idx], idx);
      }
    }
    lastCls[k] = v;
  }
  // serial commands (used by wokwi-cli automation): 'r' -> REPLAY, 'l' -> LIVE
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'l') {
      mode = (c == 'r') ? REPLAY : LIVE;
      Serial.printf("MODE,%s\r\n", mode == LIVE ? "LIVE" : "REPLAY");
      if (mode == REPLAY) { replayIdx = 0; nReplay = correctF32 = correctI8 = 0; sumLatF32 = sumLatI8 = 0; lastReplayMs = 0; }
    }
  }

  if (mode == REPLAY) {
    if (millis() - lastReplayMs >= 1500) {
      lastReplayMs = millis();
      if (replayIdx < N_TEST_WINDOWS) {
        if (inferBusy) return;                    // previous window still running
        memcpy_P(window, TEST_WINDOWS[replayIdx], sizeof window);
        char src[24]; snprintf(src, sizeof src, "REPLAY %d/%d", replayIdx + 1, N_TEST_WINDOWS);
        submitWindow(src, TEST_LABELS[replayIdx], TEST_SUBJECTS[replayIdx], replayIdx);
        replayIdx++;
      } else if (nReplay > 0 && !inferBusy) {
        Serial.printf("SUMMARY,n=%d,f32_acc=%.3f,int8_acc=%.3f,f32_mean_us=%llu,int8_mean_us=%llu\r\n", nReplay,
                      (float)correctF32 / nReplay, (float)correctI8 / nReplay, sumLatF32 / nReplay, sumLatI8 / nReplay);
        Serial.print("END\r\n");
        nReplay = 0;  // print once
      }
    }
    return;
  }

  // --- LIVE: fixed-rate polling with deadline scheduling ---
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) >= 0) {
    if ((int32_t)(now - nextSampleUs) > (int32_t)SAMPLE_PERIOD_US) missedDeadlines++;
    nextSampleUs += SAMPLE_PERIOD_US;               // keep phase: no drift accumulation
    float s[HAR_N_CH];
    if (mpuRead(s)) {
      memcpy(ring[(ringHead + ringCount) % HAR_WINDOW], s, sizeof s);
      if (ringCount < HAR_WINDOW) ringCount++; else ringHead = (ringHead + 1) % HAR_WINDOW;
      sinceLastInfer++; rateCount++;
    }
    if (now - rateWindowStart >= 1000000UL) {       // measured (not nominal) sampling rate
      measuredHz = rateCount * 1e6f / (now - rateWindowStart); rateCount = 0; rateWindowStart = now;
    }
    if (ringCount == HAR_WINDOW && sinceLastInfer >= HAR_HOP) {
      sinceLastInfer = 0;
      if (inferBusy) droppedWindows++;          // keep sampling; skip this window
      else { copyRingToWindow(); submitWindow("LIVE MPU6050", -1, -1); }
    }
  } else {
    delay(1);   // idle until next deadline (light duty-cycling of the CPU)
  }
}
