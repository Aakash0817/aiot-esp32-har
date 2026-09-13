"""Shared constants and helpers for the HAR training pipeline."""
import os, json, random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "UCI_HAR_Dataset")
RESULTS = os.path.join(ROOT, "results")
FIRMWARE = os.path.join(ROOT, "firmware", "esp32_har")

SEED = 42
FS_HZ = 50          # sampling rate of the dataset AND of the ESP32 firmware
WINDOW = 128        # samples per window (2.56 s @ 50 Hz)
HOP = 64            # 50 % overlap
N_CH = 6            # ax ay az gx gy gz
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]
# Channel order used everywhere (host + firmware): accel in g, gyro in rad/s.
SIGNALS = ["total_acc_x", "total_acc_y", "total_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z"]
# Subjects (out of the 21 train subjects) held out for validation / early stopping
VAL_SUBJECTS = [1, 3, 5, 7]

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2)

def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
