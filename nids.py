"""Shared data loading and model pipelines for the NSL-KDD intrusion detector."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

# NSL-KDD files have no header row, so we attach the official column names
COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
           'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
           'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
           'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
           'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
           'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
           'label', 'difficulty']

CATEGORICAL = ['protocol_type', 'service', 'flag']
FEATURES = COLUMNS[:41]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Explicit dtypes let pandas skip type inference and keep the frame small (float32 vs float64)
DTYPES = {**{c: 'category' for c in CATEGORICAL}, **{c: np.float32 for c in NUMERIC},
          'label': 'category', 'difficulty': np.int8}


def load_kdd(path):
    """Load an NSL-KDD file and return (X, y), where y is 0 = normal, 1 = attack."""
    df = pd.read_csv(path, names=COLUMNS, dtype=DTYPES)
    return df[FEATURES], (df['label'] != 'normal').astype(np.int8), df['label']


def build_random_forest():
    # Trees split on individual values, so ordinal codes are fine for the categoricals.
    # Categories never seen in training (e.g. new services) map to -1 instead of erroring.
    encode = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), CATEGORICAL)],
        remainder='passthrough',
    )
    return make_pipeline(encode, RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))


def build_svm():
    # SVMs are distance-based, so the categoricals are one-hot encoded and the heavily skewed
    # numeric features (src_bytes, dst_bytes, ...) are log-scaled before standardizing.
    encode = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        ('num', make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUMERIC),
    ])
    return make_pipeline(encode, SVC(kernel='rbf', C=10, random_state=42))
