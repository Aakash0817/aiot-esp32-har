"""Step 3: post-training quantisation (PTQ) with the TFLite converter.

Three artefacts are produced and evaluated on the SAME held-out test subjects:
  har_float32.tflite   - float32 weights + activations (baseline, 'before')
  har_int8.tflite      - full-integer int8 (weights, activations, input, output) ('after')
Settings (reported verbatim in the report):
  * scheme: full-integer PTQ, symmetric per-channel int8 weights, asymmetric per-tensor
    uint/int8 activations (TFLite default), int32 bias, int8 input & output tensors
  * calibration: N_REP representative windows drawn from the TRAIN split with a fixed seed
"""
import os, time, json, numpy as np
from common import *
set_seed()
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

N_REP = 500
d = np.load(os.path.join(RESULTS, "dataset.npz"))
mean, std = d["mean"], d["std"]
norm = lambda X: ((X - mean) / std).astype(np.float32)
Xtr, Xte, yte = norm(d["Xtr"]), norm(d["Xte"]), d["yte"]
model = tf.keras.models.load_model(os.path.join(RESULTS, "har_float32.keras"))

# TF 2.16 cannot convert Keras-3 Conv1D ("missing attribute value"); TF 2.17.1 is required.
make_conv = lambda: tf.lite.TFLiteConverter.from_keras_model(model)

# --- float32 tflite (baseline) ---
conv = make_conv()
tfl_f32 = conv.convert()

# --- full-integer int8 PTQ ---
rng = np.random.default_rng(SEED)
rep_idx = rng.choice(len(Xtr), N_REP, replace=False)
def representative():
    for i in rep_idx:
        yield [Xtr[i:i+1]]
conv = make_conv()
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = representative
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl_i8 = conv.convert()

for name, blob in [("har_float32.tflite", tfl_f32), ("har_int8.tflite", tfl_i8)]:
    open(os.path.join(RESULTS, name), "wb").write(blob)

def evaluate(blob, name):
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds, times = [], []
    for x in Xte:
        x = x[None]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            x = np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        t0 = time.perf_counter(); it.invoke(); times.append(time.perf_counter() - t0)
        preds.append(it.get_tensor(out["index"])[0].argmax())
    preds = np.array(preds)
    return {"model": name, "size_bytes": len(blob),
            "test_accuracy": float((preds == yte).mean()),
            "macro_f1": float(f1_score(yte, preds, average="macro")),
            "host_latency_ms_mean": float(np.mean(times) * 1e3),
            "input_dtype": str(np.dtype(inp["dtype"])), "input_quant": [float(v) for v in inp["quantization"]],
            "output_quant": [float(v) for v in out["quantization"]],
            "confusion_matrix": confusion_matrix(yte, preds).tolist()}, preds

r32, p32 = evaluate(tfl_f32, "float32")
r8, p8 = evaluate(tfl_i8, "int8_ptq")
agreement = float((p32 == p8).mean())

# --- per-layer weight statistics (for the 'outlier weights / layer sensitivity' discussion) ---
layer_stats = []
for l in model.layers:
    if l.weights:
        w = l.weights[0].numpy().ravel()
        layer_stats.append({"layer": l.name, "n": int(w.size), "min": float(w.min()), "max": float(w.max()),
                            "std": float(w.std()), "max_abs_over_std": float(np.abs(w).max() / w.std()),
                            "int8_step": float((w.max() - w.min()) / 255)})

# --- quantised tensor summary from the int8 model ---
it = tf.lite.Interpreter(model_content=tfl_i8); it.allocate_tensors()
tensors = [{"name": t["name"], "dtype": str(np.dtype(t["dtype"])), "shape": [int(s) for s in t["shape"]],
            "scale_n": len(np.atleast_1d(t["quantization_parameters"]["scales"]))}
           for t in it.get_tensor_details() if t["dtype"] in (np.int8, np.int32)]

summary = {"settings": {"scheme": "full-integer PTQ (TFLITE_BUILTINS_INT8)", "weights": "int8 symmetric, per-channel",
                        "activations": "int8 asymmetric, per-tensor", "bias": "int32",
                        "io": "int8", "representative_windows": N_REP, "seed": SEED},
           "float32": r32, "int8_ptq": r8, "prediction_agreement": agreement,
           "size_reduction_x": r32["size_bytes"] / r8["size_bytes"],
           "accuracy_delta": r8["test_accuracy"] - r32["test_accuracy"],
           "layer_weight_stats": layer_stats, "int8_tensors": tensors}
save_json(summary, "quantization.json")
for r in (r32, r8):
    print(f"{r['model']:10s} size={r['size_bytes']:6d} B  acc={r['test_accuracy']:.4f}  f1={r['macro_f1']:.4f}  host_lat={r['host_latency_ms_mean']:.3f} ms")
print(f"size reduction {summary['size_reduction_x']:.2f}x, accuracy delta {summary['accuracy_delta']:+.4f}, agreement {agreement:.4f}")
print(json.dumps(layer_stats, indent=1))
