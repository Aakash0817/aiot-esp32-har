"""Step 2: train the float32 baseline 1D-CNN (the 'before optimisation' model).

Architecture is deliberately small so it fits the ESP32 (520 KB SRAM, no tensor
accelerator) even before quantisation; quantisation then buys a further 4x size
reduction and an integer-only inference path.
"""
import os, time, json, numpy as np
from common import *
set_seed()
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
from tensorflow.keras import layers, models

d = np.load(os.path.join(RESULTS, "dataset.npz"))
mean, std = d["mean"], d["std"]
norm = lambda X: (X - mean) / std
Xtr, Xva, Xte = norm(d["Xtr"]), norm(d["Xva"]), norm(d["Xte"])
ytr, yva, yte = d["ytr"], d["yva"], d["yte"]

def build():
    m = models.Sequential([
        layers.Input(shape=(WINDOW, N_CH), name="window"),
        layers.Conv1D(16, 5, padding="same", activation="relu"),
        layers.AveragePooling1D(4),
        layers.Conv1D(32, 5, padding="same", activation="relu"),
        layers.AveragePooling1D(4),
        layers.Conv1D(32, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(len(CLASSES), activation="softmax"),
    ], name="har_cnn")
    return m

model = build()
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5)]
t0 = time.time()
hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=120, batch_size=64,
                 callbacks=cb, verbose=2)
train_s = time.time() - t0

loss, acc = model.evaluate(Xte, yte, verbose=0)
pred = model.predict(Xte, verbose=0).argmax(1)
from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(yte, pred).tolist()
model.save(os.path.join(RESULTS, "har_float32.keras"))
info = {"params": int(model.count_params()), "test_accuracy": float(acc), "test_loss": float(loss),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "epochs_run": len(hist.history["loss"]), "train_time_s": train_s,
        "confusion_matrix": cm, "classes": CLASSES}
save_json(info, "train_float32.json")
save_json({k: [float(x) for x in v] for k, v in hist.history.items()}, "history.json")
print(json.dumps({k: v for k, v in info.items() if k != "confusion_matrix"}, indent=2))
print(np.array(cm))
