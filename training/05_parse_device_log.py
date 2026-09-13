"""Step 5: turn a captured ESP32 serial log (results/device_log.txt) into results/device_metrics.json.

The log is produced by firmware/esp32_har (Wokwi or real board) after a REPLAY run:
  MODEL,float32,flash_bytes=...,arena_used_bytes=...
  REPLAY,i,subj=..,label=..,f32=..,f32_us=..,int8=..,int8_us=..
  SUMMARY,n=18,f32_acc=..,int8_acc=..,f32_mean_us=..,int8_mean_us=..
"""
import os, re, sys, json, numpy as np
from common import *

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "device_log.txt")
kv = lambda line: {k: re.sub(r"\((ok|WRONG)\)$", "", v) for k, v in (p.split("=", 1) for p in line.split(",") if "=" in p)}
models, replay, live, summary = {}, [], [], None
for line in open(path, errors="ignore"):
    line = line.strip()
    if line.startswith("MODEL,"):   models[line.split(",")[1]] = {k: int(v) for k, v in kv(line).items()}
    elif line.startswith("REPLAY,"): replay.append(kv(line))
    elif line.startswith("LIVE,"):   live.append(kv(line))
    elif line.startswith("SUMMARY,"): summary = kv(line)

def stats(key):
    v = np.array([int(r[key]) for r in replay]); return {"mean_us": float(v.mean()), "std_us": float(v.std()), "max_us": int(v.max())}
out = {"models": models,
       "replay_windows": len(replay),
       "f32_on_device_acc": float(np.mean([r["f32"] == r["label"] for r in replay])) if replay else None,
       "int8_on_device_acc": float(np.mean([r["int8"] == r["label"] for r in replay])) if replay else None,
       "f32_latency": stats("f32_us") if replay else None, "int8_latency": stats("int8_us") if replay else None,
       "speedup_x": (np.mean([int(r["f32_us"]) for r in replay]) / np.mean([int(r["int8_us"]) for r in replay])) if replay else None,
       "live_measured_rate_hz": [float(r["rate"].rstrip("Hz")) for r in live],
       "summary_line": summary}
save_json(out, "device_metrics.json")
print(json.dumps(out, indent=2))

# ---- correctness check: device predictions vs host TFLite interpreter on the same windows ----
if replay:
    import tensorflow as tf
    d = np.load(os.path.join(RESULTS, "dataset.npz")); mean, std = d["mean"], d["std"]
    idx = load_json("replay_windows.json")["replay_test_indices"]
    X = ((d["Xte"][idx] - mean) / std).astype(np.float32)
    def host_preds(name):
        it = tf.lite.Interpreter(model_path=os.path.join(RESULTS, name)); it.allocate_tensors()
        inp, out_ = it.get_input_details()[0], it.get_output_details()[0]; p = []
        for x in X:
            x = x[None]
            if inp["dtype"] == np.int8:
                s, z = inp["quantization"]; x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], x); it.invoke(); p.append(CLASSES[it.get_tensor(out_["index"])[0].argmax()])
        return p

    dev_f32 = [kv_["f32"] for kv_ in replay]; dev_i8 = [kv_["int8"] for kv_ in replay]
    hf, hi = host_preds("har_float32.tflite"), host_preds("har_int8.tflite")
    out["host_device_agreement"] = {"float32": float(np.mean([a == b for a, b in zip(hf, dev_f32)])),
                                    "int8": float(np.mean([a == b for a, b in zip(hi, dev_i8)]))}
    save_json(out, "device_metrics.json")
    print("host/device agreement:", out["host_device_agreement"])
