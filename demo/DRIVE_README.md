# Suggested Google Drive folder layout

```
AIoT_HAR_ESP32_demo/
├── AIoT_HAR_ESP32_demo.mp4      (1280×720, 30 fps, H.264/AAC)
├── README.txt                   (the text below)
├── report.pdf                   (from report/report.pdf)
└── fig_tradeoff.png             (optional: before/after figure for quick reference)
```

Sharing: right-click folder → Share → General access: **Anyone with the link → Viewer** → Copy link.

---

# Drive "Description" field of the .mp4 (short, ~300 chars)

Demo of an AIoT course project: ESP32 + MPU6050 human activity recognition with the model running on the microcontroller. Post-training int8 quantization vs float32: accuracy 90.9→91.2 %, RAM 19.8→7.6 KB, 4× smaller weights. 50 Hz polling, Nyquist-based rate, MQTT publish-subscribe output. See README.txt.

---

# README.txt (place next to the video)

ON-DEVICE HUMAN ACTIVITY RECOGNITION ON ESP32 WITH INT8 QUANTIZATION
Course project – Developing the Artificial Intelligence of Things (2026–2027)
Author: Aakash

WHAT THE VIDEO SHOWS (4:30)
0:00  Problem, objectives, board choice (ESP32: 520 KB SRAM, dual core, two I2C buses, Wi-Fi)
0:30  Sensing pipeline: MPU6050 over I2C; reading mode = polling with a phase-locked deadline
      scheduler; sampling frequency 50 Hz chosen by the Nyquist criterion (gait < 20 Hz), 21 Hz
      on-chip anti-aliasing filter; 128-sample (2.56 s) windows, 50 % overlap; measured rate and
      missed-deadline counters (50.0 Hz, 0 missed, 0 dropped)
1:20  Live operation in Wokwi: OLED output, posture changes with the sensor, injected held-out
      windows for the dynamic classes (clearly labelled INJECT – not live data)
2:00  On-device benchmark: 18 windows from unseen subjects (subject-grouped split, no data leakage)
2:30  Optimization: post-training quantization (PTQ, not QAT), float32 → int8, symmetric
      per-channel weights, 500 calibration windows; why lossless (no outlier weights, layer sensitivity)
3:20  Accuracy–latency–size trade-off and Pareto frontier, incl. the honest simulator latency result
3:55  Communication: MQTT publish–subscribe, one ~150 B message per window, live dashboard
4:15  Limitations and future work

KEY RESULTS (9 unseen test subjects, 2 947 windows)
  accuracy            float32 90.9 %  →  int8 91.2 %   (no loss; 99 % identical predictions)
  weights             29.8 KB         →  7.4 KB        (4×)
  .tflite in flash    36.5 KB         →  18.4 KB       (2×)
  RAM (tensor arena)  19.8 KB         →  7.6 KB        (2.6×)
  sampling            measured 50.0 Hz, 0 missed deadlines, 0 dropped windows under full load
  latency (Wokwi)     float32 111 ms, int8 446 ms – simulator is not cycle-accurate and penalizes
                      the 64-bit requantization of the reference int8 kernels; on physical ESP32
                      with ESP-NN kernels int8 is the fast path
  communication       ≈6 bit/s of labels instead of 4.8 kbit/s raw IMU; Wi-Fi stack +472 KB flash

DESIGN DECISIONS IN ONE LINE EACH
  Reading mode   polling – streaming unnecessary at 14 B/sample; event-based not possible (no INT pin
                 in the simulated sensor) and the device is not power-limited
  Sampling rate  signal-driven (Nyquist), not energy-driven (duty cycling)
  Drift / sync   phase-locked scheduler so jitter never accumulates; rate measured, not assumed
  Concurrency    sampler on core 1; inference, display, network on core 0; IMU and OLED on
                 separate I2C controllers (three iterations to reach 0 missed deadlines)
  Model          1D-CNN, 7 446 parameters, sized for the device first
  Evaluation     subject-grouped 17/4/9 split to avoid leakage; validated on the target
                 (host-vs-device agreement 100 % float32, 94.4 % int8)
  Optimization   full-integer PTQ with the TFLite converter; input scale 0.10107, zero-point −6

REPOSITORY
  code, training scripts, optimized model (har_int8.tflite / model_data.h), Wokwi diagram,
  README with reproduction steps, report:  <GITHUB LINK>

TOOLS
  ESP32 DevKit-C v4 in Wokwi · TensorFlow 2.17 / Keras 3 · TensorFlow Lite Micro · arduino-cli ·
  MQTT (broker.hivemq.com) · UCI HAR dataset (Anguita et al., 2013)

AI-TOOL DISCLOSURE
  Claude Code (Claude Opus 5) was used for pair-programming, debugging and simulation support
  during development; details in the written report.
