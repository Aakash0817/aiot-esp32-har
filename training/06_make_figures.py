"""Step 6: figures for the report (results/fig_*.png) from the JSON produced by steps 2-5."""
import os, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
SHORT = ["WALK", "UPSTAIRS", "DOWNSTAIRS", "SIT", "STAND", "LAY"]

q = load_json("quantization.json"); tr = load_json("train_float32.json"); h = load_json("history.json")
dev = load_json("device_metrics.json") if os.path.exists(os.path.join(RESULTS, "device_metrics.json")) else None

# 1. training curves
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val (held-out subjects)"); ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(h["accuracy"]); ax[1].plot(h["val_accuracy"]); ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_training.png"), dpi=150)

# 2. confusion matrices float vs int8
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for a, key, title in [(ax[0], "float32", "float32"), (ax[1], "int8_ptq", "int8 PTQ")]:
    cm = np.array(q[key]["confusion_matrix"]); a.imshow(cm, cmap="Blues")
    a.set_xticks(range(6)); a.set_yticks(range(6)); a.set_xticklabels(SHORT, rotation=45, ha="right"); a.set_yticklabels(SHORT)
    for i in range(6):
        for j in range(6): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_title(f"{title}: acc={q[key]['test_accuracy']*100:.1f}%"); a.set_xlabel("predicted"); a.set_ylabel("true")
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_confusion.png"), dpi=150)

# 3. before/after on the three axes of the trade-off (accuracy, memory, latency)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.8))
names = ["float32\n(before)", "int8 PTQ\n(after)"]
acc = [q["float32"]["test_accuracy"] * 100, q["int8_ptq"]["test_accuracy"] * 100]
ax[0].bar(names, acc, color=["#888", "#e07b39"]); ax[0].set_ylim(80, 100); ax[0].set_title("test accuracy (%)")
for i_, v in enumerate(acc): ax[0].text(i_, v + 0.4, f"{v:.1f}", ha="center")
flash = [q["float32"]["size_bytes"] / 1024, q["int8_ptq"]["size_bytes"] / 1024]
arena = [dev["models"]["float32"]["arena_used_bytes"] / 1024, dev["models"]["int8"]["arena_used_bytes"] / 1024] if dev else [0, 0]
x = np.arange(2); ax[1].bar(x - 0.2, flash, 0.4, label="flash (.tflite)"); ax[1].bar(x + 0.2, arena, 0.4, label="RAM (arena)")
ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_title("memory (KB)"); ax[1].legend(fontsize=8, loc="upper right")
ax[1].set_ylim(0, max(flash) * 1.25)
for i_, (f_, a_) in enumerate(zip(flash, arena)): ax[1].text(i_ - 0.2, f_ + 0.8, f"{f_:.1f}", ha="center", fontsize=8); ax[1].text(i_ + 0.2, a_ + 0.8, f"{a_:.1f}", ha="center", fontsize=8)
if dev and dev.get("f32_latency"):
    lat = [dev["f32_latency"]["mean_us"] / 1000, dev["int8_latency"]["mean_us"] / 1000]
    ax[2].bar(names, lat, color=["#888", "#e07b39"]); ax[2].set_title("latency, Wokwi ESP32 (ms)\nreference kernels, not cycle-accurate", fontsize=9)
    ax[2].set_ylim(0, max(lat) * 1.18)
    for i_, v in enumerate(lat): ax[2].text(i_, v + max(lat) * 0.02, f"{v:.0f}", ha="center")
for a in ax: a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_tradeoff.png"), dpi=150)

# 4. weight distributions per layer (outlier discussion)
import tensorflow as tf
m = tf.keras.models.load_model(os.path.join(RESULTS, "har_float32.keras"))
ws = [(l.name, l.weights[0].numpy().ravel()) for l in m.layers if l.weights]
fig, ax = plt.subplots(1, len(ws), figsize=(2.4 * len(ws), 2.6))
for a, (n, w) in zip(ax, ws): a.hist(w, bins=40); a.set_title(n, fontsize=8); a.set_yticks([])
fig.suptitle("float32 weight distributions (no outliers -> PTQ-friendly)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "fig_weights.png"), dpi=150)
print("figures written to", RESULTS)
