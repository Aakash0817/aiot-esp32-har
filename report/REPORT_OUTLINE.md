# Report outline (5 pages) — write the prose yourself; every number below is in `results/summary.md`

Grading: Description 25 % · Justification 25 % · **Contextualisation within course topics 50 %**.
So for every design decision write three sentences: *what* you did, *why* (the alternative you rejected), *which course concept* it applies.

## 1. Prototype (≈0.8 page)
- **Problem**: recognise 6 daily activities (walk / upstairs / downstairs / sit / stand / lay) on a wearable-class MCU without sending raw IMU data to the cloud (privacy, bandwidth, latency, battery). Evaluate against Chapter 1.6 (the course's system-level view of an AIoT device: sensing → processing → communication → energy).
- **Objective**: full sensing pipeline + on-device model + one edge-optimisation technique (PTQ) with before/after evidence.
- **Development process** (be concrete, this is where "iterations" go):
  1. Chose ESP32 (dual core, 520 KB SRAM, FPU) over Pico/Uno → justify with model RAM need (float arena 19.8 KB, int8 7.6 KB) and dual-core.
  2. Dataset UCI HAR (real MPU-class 50 Hz signals, 30 subjects) instead of recording in Wokwi (sliders cannot produce realistic gait).
  3. First model used MaxPooling → **abandoned**: no deterministic GPU gradient kernel, which broke bit-exact reproducibility; switched to AveragePooling (same accuracy).
  4. TF 2.16 converter bug on Keras-3 Conv1D → pinned TF 2.17.1 (reproducibility).
  5. First firmware ran inference inside `loop()` → sampler missed 64 samples per inference → **moved inference to a FreeRTOS task on core 0**.
  6. OLED refresh on the shared I2C bus still caused missed deadlines → **OLED moved to the second I2C controller (Wire1)**; result 50.0 Hz, 0 missed, 0 dropped.
  7. `-Os` → `-O2` compiler flag: 7× lower latency for +14 KB flash (a size-vs-latency trade-off in itself).

## 2. Sensing pipeline (≈1.2 pages)
### Hardware, sensors, communication, energy
- MPU6050: 6-axis MEMS, ±2 g / ±250 °/s, 16-bit, on-chip DLPF set to 21 Hz (**anti-aliasing**: below fs/2 = 25 Hz).
- **Reading mode: polling** with a phase-locked `micros()` deadline scheduler (why not interrupt: Wokwi MPU6050 has no INT pin model; why not streaming/FIFO: 12 B/sample at 50 Hz is trivial on I2C 400 kHz). Mention the course note that "a sensor configured for 100 Hz rarely delivers 100 Hz" → you *measure* the rate (50.0 Hz) and count missed deadlines rather than assume it.
- **Sampling frequency 50 Hz — justified by Nyquist**: human locomotion energy is < ~20 Hz, so fs ≥ 40 Hz; 50 Hz matches the dataset (train/deploy parity). Not energy-driven (contrast with the DHT22 duty-cycling topic).
- **Windowing**: 128 samples = 2.56 s ≥ one full gait cycle; hop 64 (50 % overlap) so a decision every 1.28 s. Ring buffer of 128×6 floats = 3 KB.
- **Signals captured**: gravity+body acceleration (g) and angular rate (rad/s); per-channel standardisation with train-set mean/std compiled into firmware (train/deploy parity).
- **Communication**: MQTT **publish-subscribe** over Wi-Fi, one ~150 B JSON message per 1.28 s window (QoS 0, public broker), consumed by a web dashboard. Justify: pub-sub decouples device from consumers, no polling; only the inference result leaves the device (~6 bit/s of information) instead of the raw IMU stream (50 Hz × 12 B ≈ 4.8 kbit/s) → the bandwidth/privacy/energy argument for edge AI. Costs you measured: +472 KB flash for the network stack; TCP connect blocks ~1 s → networking moved to core 0 after it caused 36 missed deadlines in loop(); inference latency 87 → 110 ms when core 0 also runs the network stack.
- **Energy budget** (estimate, be explicit it is an estimate): MPU6050 ≈ 3.9 mA active; ESP32 ≈ 30–50 mA active at 240 MHz without Wi-Fi; inference duty = latency / 1.28 s (e.g. 87 ms/1280 ms ≈ 7 %) → light-sleep between samples is the next step. Tie to duty cycling concept.
- **Concurrency**: sampler on core 1, inference task on core 0, separate I2C buses → real-time sampling is never blocked (system-level metric: 0 missed deadlines).

### On-device AI
- Model: 1D-CNN (Conv16 → AvgPool4 → Conv32 → AvgPool4 → Conv32 → GAP → Dense32 → Dense6), 7 446 params. Why CNN: learns filters over the time axis = learned feature extraction instead of hand-crafted features; small enough for the arena.
- Evaluation protocol: **subject-grouped split** (17 train / 4 val / 9 test subjects, no subject overlap) → the accuracy is for *unseen people*; explain why a random split would leak.
- **PTQ settings (report verbatim)**: full-integer PTQ via TFLite converter, `TFLITE_BUILTINS_INT8`, int8 symmetric per-channel weights, int8 asymmetric per-tensor activations, int32 bias, int8 input/output; calibration with 500 seeded training windows; input scale 0.10107 / zero-point −6; output scale 1/256 / zp −128. Post-training (no QAT) because the accuracy loss was already ≈ 0.
- **Before / after** (Table from summary.md): accuracy 90.94 → 91.18 % (+0.24 pt, within noise; 99.05 % prediction agreement); weights 29.8 KB → 7.4 KB (4×); .tflite 36.5 → 18.4 KB (2× — explain the flatbuffer/quant-param overhead on a tiny model); arena 19.8 → 7.6 KB (2.6×); on-device 18-window replay 94.4 % vs 88.9 % (one borderline STANDING window, conf 0.57).
- **Why no accuracy loss — outlier / layer sensitivity**: all layers have max|w| ≤ 3.5 σ (table), i.e. no outlier weights, so per-channel int8 steps of 0.003–0.0045 lose nothing; the most sensitive layer is `conv1d_1` (largest max|w|/σ) — cite the weight-histogram figure.
- **Latency — honest discussion**: Wokwi float 87 ms vs int8 354 ms. Explain (i) Wokwi is not cycle-accurate (bench: 170 k MACs = 167 ms), (ii) TFLM *reference* int8 kernels requantise with int64 multiplies, which Wokwi executes 4× slower than int32, (iii) on real ESP32 with ESP-NN kernels int8 conv is the accelerated path. Conclusion on the **Pareto frontier**: int8 dominates on size and RAM at equal accuracy; on latency the gain depends on hardware/kernel support — a key lesson of the "AI on the edge" chapter.
- Host-vs-device agreement 100 % (float) / 94.4 % (int8): integer kernels differ slightly between TFLite and TFLM (rounding in requantisation).

## 3. Self-reflection (≈1 page)
- **Limitations**: SITTING↔STANDING confusion (both static, same gravity vector — needs barometer or longer context); Wokwi sliders cannot exercise dynamic classes live (hence replay from held-out subjects); latency not measurable on real silicon; no communication link; no sleep modes; dataset from a waist-mounted phone → orientation mapping assumed.
- **Future work**: ESP-NN kernels / physical board for real latency; light-sleep duty cycling; MQTT publish of labels; QAT if a smaller 4-bit model is wanted; on-device calibration for orientation.
- **ONE course concept** — pick one and go deep (suggested: *grouped evaluation splits* or *the accuracy–latency–size Pareto frontier*): define it, why it matters for AIoT, exactly how you applied it and what you observed.
- How your understanding of AIoT evolved (your own words).

## 4. Figures to include
`results/fig_tradeoff.png` (before/after, 3 axes) · `results/fig_confusion.png` · `results/fig_weights.png` · `results/fig_training.png` · a screenshot of the Wokwi diagram.

## 5. AI-tool declaration (mandatory, last section — do not shorten)
> **AI tool used:** Claude Code (Anthropic), model Claude Opus 5 (`claude-opus-5`), September 2026.
> **Purpose:** pair-programming and troubleshooting during development: scaffolding the training/quantisation scripts and the ESP32 firmware, debugging toolchain issues (TensorFlow 2.16 converter bug, non-deterministic GPU pooling kernel), running and interpreting Wokwi simulations, and reviewing the structure of this report. All design decisions (topic, board, optimisation technique, evaluation protocol) were made by me, and the text of this report and the demo video are my own.
Adjust the wording to be accurate for how you actually worked.
