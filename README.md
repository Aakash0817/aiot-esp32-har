# On-device Human Activity Recognition on ESP32 (AIoT course project)

An ESP32 reads a 6-axis MPU6050 IMU at a fixed 50 Hz, windows the stream into
2.56 s frames and classifies the activity (walking, walking upstairs/downstairs,
sitting, standing, laying) **on the device** with a 1D-CNN. The model is
optimised with **post-training int8 quantisation** and compared against the
float32 baseline for accuracy, latency and memory footprint.

* Board: ESP32 DevKit-C v4 (Wokwi simulation) · Sensor: MPU6050 (I2C, polled) · Display: SSD1306 OLED
* Wokwi diagram: [`firmware/esp32_har/diagram.json`](firmware/esp32_har/diagram.json)
  · Wokwi project link: **<paste your wokwi.com project URL here>**
* Optional add-on: results are published over **MQTT (publish-subscribe)**; live web dashboard in [`dashboard/index.html`](dashboard/index.html)
* All reported numbers: [`results/summary.md`](results/summary.md) (auto-generated)
* Report: `report/report.pdf` · Demo video: **<paste YouTube link here>**

## Repository layout

| path | what |
|---|---|
| `data/download_uci_har.sh` | downloads the UCI HAR dataset (30 subjects, 50 Hz accel+gyro) |
| `training/01_prepare_data.py` | raw signals → (N,128,6) windows, subject-grouped train/val/test split, normalisation stats |
| `training/02_train.py` | float32 1D-CNN baseline ("before") |
| `training/03_quantize.py` | post-training int8 quantisation ("after") + host evaluation of both models |
| `training/04_export_c.py` | exports both `.tflite` models, normalisation constants and 18 held-out test windows as C headers |
| `training/05_parse_device_log.py` | parses the ESP32 serial log → on-device accuracy / latency / arena numbers, host-vs-device agreement |
| `training/06_make_figures.py`, `07_summary.py` | figures and the results table |
| `firmware/esp32_har/` | **sensing + on-device inference firmware** (Arduino/ESP32, TFLite Micro), Wokwi diagram, build & run scripts |
| `firmware/esp32_har/model_data.h` | **the optimised model artifact** (`har_int8_tflite[]`) and the float32 baseline, as C arrays |
| `dashboard/index.html` | MQTT-over-WebSocket subscriber: live activity, timeline, latency (open the file in a browser, click *connect*) |
| `firmware/wokwi_timing_bench/` | micro-benchmark used to calibrate Wokwi simulated time |
| `results/` | `har_float32.tflite`, `har_int8.tflite`, all metrics as JSON, figures, `device_log.txt`, `summary.md` |

## Reproducing every number in the report

Tested on Ubuntu, Python 3.12, NVIDIA GPU (CPU works too, just slower). All
scripts are seeded (`SEED=42`) and TensorFlow runs with deterministic ops, so
`results/summary.md` should be regenerated bit-for-bit.

### 1. Host side (dataset → trained model → quantised model)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash data/download_uci_har.sh
cd training
../.venv/bin/python 01_prepare_data.py      # results/dataset.npz, dataset_info.json
../.venv/bin/python 02_train.py             # results/har_float32.keras, train_float32.json
../.venv/bin/python 03_quantize.py          # results/har_float32.tflite, har_int8.tflite, quantization.json
../.venv/bin/python 04_export_c.py          # firmware/esp32_har/{model_data,har_config,test_windows}.h
```

`requirements.txt` pins `tensorflow[and-cuda]==2.17.1` / `keras==3.3.3`.
TF 2.16 has a converter bug with Keras-3 `Conv1D` ("missing attribute 'value'"); 2.17 is required.

### 2. Device side (build firmware, run it in Wokwi, capture the serial log)

Prerequisites: [arduino-cli](https://arduino.github.io/arduino-cli) with the
`esp32:esp32@2.0.17` core and libraries `TensorFlowLite_ESP32@1.0.0`,
`Adafruit SSD1306@2.5.17`, `Adafruit GFX Library@1.12.6`, `Adafruit BusIO@1.17.4`, `PubSubClient@2.8.0`:

```bash
arduino-cli core install esp32:esp32@2.0.17
arduino-cli lib install "TensorFlowLite_ESP32@1.0.0" "Adafruit SSD1306@2.5.17" "Adafruit GFX Library@1.12.6" "Adafruit BusIO@1.17.4" "PubSubClient@2.8.0"
```

```bash
firmware/esp32_har/build.sh                    # compiles with -O2 → firmware/esp32_har/build/
curl -L https://wokwi.com/ci/install.sh | sh   # wokwi-cli (once)
export WOKWI_CLI_TOKEN=...                     # free token from https://wokwi.com/dashboard/ci
firmware/esp32_har/run_wokwi.sh                # runs replay.scenario.yaml → results/device_log.txt
```

The scenario boots the firmware, lets it run ~8 s in LIVE mode (prints the
measured sampling rate), then sends `r` over serial to start REPLAY mode, which
pushes 18 held-out test windows through both models and prints a `SUMMARY` line.

Without the CLI: open `firmware/esp32_har/` in the Wokwi VS Code extension
(`wokwi.toml` points at the built binary) or paste the sources + `diagram.json`
into a wokwi.com project (libraries in `libraries.txt`), press the green MODE
button to switch LIVE ↔ REPLAY, and save the serial monitor output to
`results/device_log.txt`.

### 3. Metrics, figures, summary table

```bash
cd training
../.venv/bin/python 05_parse_device_log.py    # results/device_metrics.json
../.venv/bin/python 06_make_figures.py        # results/fig_*.png
../.venv/bin/python 07_summary.py             # results/summary.md  ← the table used in the report
```

## Firmware behaviour (what you see in the serial monitor)

```
BOOT,fs=50Hz,window=128,hop=64,free_heap=...
SENSOR,mpu6050=ok
MODEL,float32,flash_bytes=36460,arena_used_bytes=19792,arena_size=24576
MODEL,int8,flash_bytes=18352,arena_used_bytes=7556,arena_size=12288
TASK,inference_core=0,sampler_core=1
LIVE,rate=50.0Hz,missed=0,dropped=0,f32=STANDING,...,int8=SITTING,...      # every 64 samples (1.28 s)
INJECT,15,subj=20,label=LAYING,f32=LAYING(ok),...,int8=LAYING(ok),...        # class button pressed
MODE,REPLAY
REPLAY,0,subj=13,label=WALKING,f32=WALKING,f32_conf=0.998,f32_us=...,int8=WALKING,...
SUMMARY,n=18,f32_acc=0.944,int8_acc=0.889,f32_mean_us=...,int8_mean_us=...
```

* **Sampling**: polling the MPU6050 over I2C with a `micros()` deadline scheduler
  (period 20 000 µs, phase-locked so jitter does not accumulate). The achieved
  rate and missed deadlines are measured and printed, not assumed.
* **Windowing**: 128-sample ring buffer, inference every 64 new samples (50 % overlap) — identical to the training windows.
* **Preprocessing parity**: per-channel mean/std from the training subjects are
  compiled into `har_config.h`; the same code path standardises LIVE and REPLAY windows.
* **Concurrency**: sampling runs in `loop()` on core 1; inference runs in a
  FreeRTOS task pinned to core 0. IMU and OLED are on separate I2C controllers
  (`Wire`, `Wire1`) so the ~25 ms display refresh never blocks a sensor read.
* **Both models** run on every window so latency/RAM can be compared on identical inputs.

## Communication add-on (MQTT, optional)

`ENABLE_MQTT 1` (default) in `esp32_har.ino` makes the ESP32 join Wokwi's simulated
`Wokwi-GUEST` Wi-Fi and publish one JSON message (~150 B) per classified window to
`aiot/har/<chip-id>/activity` on the public broker `broker.hivemq.com:1883` (QoS 0).
Open `dashboard/index.html` in any browser and click **connect** (it subscribes to
`aiot/har/+/activity` over WebSockets, port 8884) to see the live activity timeline.

Design points (see report): publish-subscribe decouples the constrained device from any
number of consumers; only the *inference result* leaves the device (~6 bit/s of
information vs ≈4.8 kbit/s of raw IMU data), which is the bandwidth/energy argument for
edge AI; the network stack costs **+472 KB flash** (518 → 991 KB) and the TCP connect
blocks for ~1 s, so all networking runs in the core-0 task — the first version in
`loop()` caused 36 missed sample deadlines. Set `ENABLE_MQTT 0` for a fully offline
build (sampling/inference/replay behaviour and numbers are unchanged except ~20 % lower
inference latency because core 0 is then idle apart from inference).

## Wiring (Wokwi `diagram.json`)

| part | pins |
|---|---|
| MPU6050 | VCC→3V3, GND→GND, SDA→GPIO21, SCL→GPIO22 (I2C bus 0) |
| SSD1306 OLED (0x3C) | VCC→3V3, GND→GND, SDA→GPIO25, SCL→GPIO26 (I2C bus 1) |
| push button "MODE" (green) | GPIO4 (internal pull-up) → GND: runs the 18-window REPLAY benchmark |
| class buttons WALK / UP / DOWN / SIT / STAND / LAY (blue) | GPIO32 / 33 / 27 / 14 / 12 / 13 (pull-up) → GND: each press injects one *recorded* held-out window of that class into the on-device pipeline (`INJECT` lines). Wokwi's slider-driven IMU cannot produce a gait, so this is how dynamic classes are demonstrated interactively. |
| on-board LED | GPIO2, on while inference runs |

## Note on latency numbers

Wokwi is functionally accurate but not cycle-accurate. `firmware/wokwi_timing_bench`
shows 170 k simple float MACs take 167 ms of simulated time (~240 cycles each;
real silicon needs ~10–20) and 64-bit integer math is ~4× slower again. The
int8 reference kernels in TFLite Micro requantise with 64-bit multiplies, so in
Wokwi the int8 model is *slower* than float32, whereas on a physical ESP32 with
Espressif's ESP-NN kernels int8 is the fast path. Flash and arena figures are
independent of the simulator. See the report for the discussion.
