# YouTube title

On-Device Human Activity Recognition on ESP32 with Int8 Quantization (AIoT Course Project)

# YouTube description (paste as-is; edit the links)

Course project for "Developing the Artificial Intelligence of Things" (University of Helsinki MOOC, 2026–2027). An ESP32 reads an MPU6050 6-axis IMU, windows the stream, classifies six activities with a 1D-CNN running on the microcontroller itself, and publishes only the label over MQTT. The model is optimized with post-training int8 quantization and compared with the float32 baseline on accuracy, latency and memory.

Repository (code, training scripts, optimized model, Wokwi diagram, report): <GITHUB LINK>
Wokwi project: <WOKWI LINK>

CHAPTERS
0:00 Problem, objectives and board choice (ESP32: 520 KB SRAM, dual core, two I2C buses, Wi-Fi)
0:30 Sensing pipeline – MPU6050 over I2C, polling with a phase-locked deadline scheduler, 50 Hz chosen by the Nyquist criterion (gait < 20 Hz), 21 Hz on-chip anti-aliasing filter, 128-sample / 2.56 s windows with 50 % overlap, measured rate and missed-deadline counters
1:20 Live operation – simulated sensor, OLED output, injected held-out windows for the dynamic classes
2:00 On-device benchmark – 18 windows from unseen subjects (subject-grouped split, no data leakage), both models on identical inputs
2:30 Optimization – post-training quantization (PTQ, not QAT), float32 → int8, symmetric per-channel weights, 500 calibration windows; why it was lossless (no outlier weights, layer sensitivity)
3:20 Accuracy–latency–size trade-off and the Pareto frontier – including the honest simulator latency result
3:55 Communication – MQTT publish–subscribe, one ~150-byte message per window, live dashboard
4:15 Limitations and future work

KEY RESULTS (9 unseen test subjects, 2 947 windows)
• Accuracy: float32 90.9 % → int8 91.2 % (no loss; 99 % of predictions identical)
• Weights: 29.8 KB → 7.4 KB (4×) · .tflite in flash: 36.5 KB → 18.4 KB (2×)
• RAM (tensor arena on ESP32): 19.8 KB → 7.6 KB (2.6×)
• Sampling: measured 50.0 Hz, 0 missed deadlines, 0 dropped windows with inference, display and Wi-Fi all active
• Latency (Wokwi simulation, reference TFLite-Micro kernels): float32 111 ms, int8 446 ms – explained in the video: the simulator is not cycle-accurate and penalizes the 64-bit requantization in the reference int8 kernels; on physical ESP32 with ESP-NN kernels int8 is the fast path
• Communication: label channel ≈ 6 bit/s vs 4.8 kbit/s raw IMU stream; Wi-Fi/TCP stack costs +472 KB flash

SENSING DESIGN DECISIONS
• Reading mode: polling (not streaming – 14 B per sample is trivial; not event-based – the simulated sensor has no INT pin and the device is not power-limited)
• Sampling frequency determined by the signal (Nyquist), not by energy (duty cycling) – the opposite of the environmental-sensing topic
• Drift/synchronization: phase-locked scheduler so jitter never accumulates; rate is measured, not assumed
• Concurrency: sampler on core 1, inference + display + network on core 0; IMU and OLED on separate I2C controllers – three iterations were needed to reach 0 missed deadlines

AI DESIGN DECISIONS
• 1D-CNN, 7 446 parameters, sized for the device first
• Subject-grouped train/val/test split (17/4/9 subjects) to avoid data leakage from overlapping windows of the same person
• Full-integer PTQ with the TFLite converter; input scale 0.10107 / zero-point −6
• Validated on the target: host-vs-device agreement 100 % (float32), 94.4 % (int8)

TOOLS
ESP32 DevKit-C v4 in Wokwi · TensorFlow 2.17 / Keras 3 · TensorFlow Lite Micro (TensorFlowLite_ESP32) · Arduino-CLI · MQTT (broker.hivemq.com) · UCI HAR dataset (Anguita et al., 2013)

AI-tool disclosure: Claude Code (Claude Opus 5) was used for pair-programming, debugging and simulation support during development; details in the written report.

#AIoT #TinyML #ESP32 #Quantization #TensorFlowLite #EdgeAI #Wokwi #MQTT #HumanActivityRecognition #MPU6050

# Tags (YouTube "tags" field)

AIoT, TinyML, ESP32, TensorFlow Lite Micro, int8 quantization, post-training quantization, edge AI, Pareto frontier, human activity recognition, MPU6050, Wokwi, MQTT, publish subscribe, Nyquist, IoT sensing pipeline, University of Helsinki MOOC

# Settings when uploading
- Visibility: Unlisted (link works for the grader; not public)
- Category: Science & Technology
- Language: English
- Thumbnail: screenshot of the Wokwi diagram with the OLED showing a prediction, or fig_tradeoff.png
