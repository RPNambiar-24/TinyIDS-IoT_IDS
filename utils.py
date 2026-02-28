import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os, pickle

SEED = 42
SEEN_RATIO = 0.7  # 70% attack families for training, 30% unseen

FEATURE_COLS = [
    'duration', 'proto', 'service', 'orig_bytes', 'resp_bytes',
    'conn_state', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes',
    'resp_pkts', 'resp_ip_bytes'
]

LABEL_COL = 'label'

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
