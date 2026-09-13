# Demo video — sequence, screen content and narration

Target: 4 min 30 s · 1280×720 · 30 fps · H.264/AAC `.mp4` · landscape.
Rubric: Description & Justification 20 % · Operation & Functionality 30 % · Contextualisation 50 % → say the course terms out loud (they are **bold** below).

## Preparation (before recording)

1. VS Code open on `AIOT/`, Wokwi simulator running (`F1 → Wokwi: Start Simulator`), serial terminal visible with a large font (`Ctrl+=` twice).
2. Browser window with `dashboard/index.html` open and **connect** clicked; status shows *connected*.
3. Arrange: simulator left 60 % of the screen, dashboard right 40 % (or alt-tab between them).
4. Second VS Code tab group with `results/summary.md` and the figures `fig_tradeoff.png`, `fig_weights.png`, `fig_confusion.png` ready.
5. Test the microphone level — record 10 s and play it back.
6. Sequence to rehearse on the sim: drag accel sliders → press SIT → press WALK → press DOWN → press MODE → wait for SUMMARY.

Recorder: OBS Studio (Display Capture + Mic) or SimpleScreenRecorder; output preset "1280×720, 30 fps, mp4".

---

## Scene 1 — Title and problem (0:00 – 0:30)
**Screen:** Wokwi diagram, sim stopped or paused; mouse hovers over each part as it is named.

> "This is an on-device human activity recognition prototype. An ESP32 reads a six-axis MPU6050 inertial sensor over I2C, classifies each two-and-a-half-second window into six activities — walking, upstairs, downstairs, sitting, standing, lying — with a neural network that runs on the microcontroller itself, and publishes only the label over MQTT.
> Why on the device? Raw motion data is 4.8 kilobits per second of personally identifiable data. The course's **data-minimisation** principle says don't transmit raw data if you can avoid it, and the radio, not the CPU, dominates the **energy budget**. So the intelligence stays at the edge.
> I chose the ESP32 because I actually use its features: 520 kilobytes of RAM to hold both the float and the int8 model, **two cores** to separate real-time sampling from inference, two I2C controllers, and Wi-Fi."

## Scene 2 — Sensing pipeline (0:30 – 1:20)
**Screen:** `esp32_har.ino` scrolled to `mpuInit` and the sampler in `loop()`; then the serial terminal.

> "The sensing pipeline is designed, not default. **Reading mode is polling** with a phase-locked deadline scheduler — the next deadline is the previous deadline plus 20 milliseconds, never 'now plus 20', so jitter never accumulates into **drift**. I didn't use **event-based** interrupts because Wokwi's sensor has no INT pin and the device isn't power-limited here; **streaming** is pointless at 14 bytes per sample.
> **Sampling frequency is 50 hertz, chosen by the Nyquist criterion**: human motion has energy below about 20 hertz, so I need at least 40, and 50 matches the training data. The sensor's on-chip low-pass filter at 21 hertz is the **anti-aliasing** filter. This is the opposite of the environmental-sensing topic where the rate is chosen by **duty cycling** for energy.
> **Windowing**: 128 samples, 2.56 seconds — more than a full gait cycle — with 50 % overlap, so a decision every 1.28 seconds.
> And here is the important line —" *(point at `LIVE,rate=50.0Hz,missed=0,dropped=0`)* "— the course says a sensor configured for 100 hertz rarely delivers 100 hertz. So I don't assume; I **measure** the rate and count missed deadlines and dropped windows. It's 50.0, zero, zero — while inference, the display and Wi-Fi are all running."

## Scene 3 — Live operation (1:20 – 2:00)
**Screen:** simulator; click the MPU6050, drag the accelerometer sliders; watch OLED and the dashboard.

> "Live mode. Gravity is on the vertical axis, so the model says a static upright posture." *(drag accel so gravity moves to another axis)* "Rotate the sensor — and it becomes LAYING. The OLED shows the int8 prediction, the confidence, the latency of both models, and the measured rate; the dashboard on the right receives the same result over MQTT.
> Wokwi's sensor is slider-driven, so it can't walk. To demonstrate the dynamic classes I inject **recorded windows from test subjects the model never saw**, through the identical on-device pipeline — the log marks them INJECT so nobody mistakes them for live data."
*(press WALK, then DOWN, then SIT; read the OLED each time)*
> "WALKING, 96 %. WALKING DOWNSTAIRS. SITTING. Same preprocessing, same model, same code path as the live stream."

## Scene 4 — On-device benchmark (2:00 – 2:30)
**Screen:** press the green MODE button; let the 18 REPLAY lines scroll; stop on `SUMMARY`.

> "MODE runs the on-device benchmark: 18 windows, three per class, from the held-out subjects — a **subject-grouped split**, so these are unseen people, no **data leakage** from overlapping windows of the same person. Both models run on every window so they're compared on identical inputs. Summary: float 94 %, int8 89 % on this small set; the one difference is a borderline standing-versus-sitting window."

## Scene 5 — Quantisation: what and why (2:30 – 3:20)
**Screen:** `03_quantize.py` settings block, then `results/summary.md` table, then `fig_weights.png`.

> "The optimisation technique is **post-training quantisation, not quantisation-aware training**: train in float32, then convert to **int8** with the TFLite converter — full integer: weights int8 **symmetric per-channel**, activations int8 per-tensor, biases int32, input and output int8, **calibrated** on 500 training windows. Input scale 0.101, zero-point minus six.
> Results on 2,947 windows from nine unseen subjects: float 90.9 %, int8 91.2 % — **no accuracy loss**, 99 % of predictions identical. Weights four times smaller. RAM — the tensor arena — from 19.8 to 7.6 kilobytes.
> Why lossless? The course explains that **outlier weights** stretch the quantisation bins and crush resolution for the small weights. I checked every layer: the largest weight is under 3.5 sigma everywhere —" *(show histograms)* "— compact, Gaussian-looking distributions, so int8 steps of about 0.004 lose nothing. The most **sensitive layer** is the second convolution; it would be the first to keep in float if there had been a loss. That's why PTQ was enough and QAT wasn't needed."

## Scene 6 — The trade-off and the honest latency result (3:20 – 3:55)
**Screen:** `fig_tradeoff.png`.

> "The **accuracy–latency–size trilemma**. On two axes the int8 model **dominates** at equal accuracy — it is strictly closer to the **Pareto frontier**. On latency, the simulator shows int8 *slower*: 446 versus 111 milliseconds. I benchmarked the simulator: it isn't cycle-accurate, and it runs 64-bit integer math four times slower than 32-bit — and the reference int8 kernels in TensorFlow Lite Micro requantise with 64-bit multiplies. On a physical ESP32 with ESP-NN kernels int8 is the fast path. So the lesson is that **moving toward the frontier for your device** depends on the device's integer execution path, not only on the model."

## Scene 7 — Communication (3:55 – 4:15)
**Screen:** dashboard; then the `MQTT,connected` and `TASK,inference_core=0` lines.

> "Communication is **MQTT publish–subscribe**: one 150-byte JSON message per window, QoS 0, because a lost label is replaced 1.3 seconds later. Pub-sub suits a constrained device — it never waits for a consumer. Six bits per second of labels instead of 4.8 kilobits of raw data. It wasn't free: the network stack costs 472 kilobytes of flash, and the first version, connecting inside the main loop, caused 36 **missed deadlines** — the '**ideal network fallacy**' from chapter 1.6. Networking now runs on core 0, the sampler alone on core 1, and the sensor and display are on separate I2C buses."

## Scene 8 — Limitations and close (4:15 – 4:30)
**Screen:** confusion matrix, then back to the running sim.

> "Limitations: sitting versus standing is confused about one time in five — same gravity vector, a sensing limitation, not a quantisation one; dynamic classes can't be produced live by the simulated sensor; latency needs real silicon; and the broker is public and unencrypted. Everything in the report regenerates from the scripts in the repository. Thanks for watching."

---

## Checklist against the demo spec
- [x] sensor and **reading mode** + why (Scene 2)
- [x] **sampling frequency** and whether Nyquist- or energy-driven (Scene 2)
- [x] noise / drift / synchronisation handling: anti-alias filter, phase-locked scheduler, measured rate (Scene 2)
- [x] device receiving sensor data and producing an output (Scene 3)
- [x] which technique, PTQ vs QAT, precision change, accuracy change, outliers / layer sensitivity (Scene 5)
- [x] accuracy, latency, memory before/after and the Pareto explanation (Scenes 5–6)
- [x] communication method and pattern, why suitable for constrained device (Scene 7)
- [x] 3–5 min, 720p, 30 fps, mp4
