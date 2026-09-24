# Machine Learning Network Intrusion Detection Engine

A Python/scikit-learn engine that classifies network connection records as **normal** or
**malicious**. It trains and compares Random Forest and SVM models on the NSL-KDD dataset and
includes a real-time detector that classifies a live stream of connection records.

## Highlights

- **99.90% (Random Forest) and 99.78% (SVM) accuracy** on a stratified 80/20 split of the
  125,973-record NSL-KDD training set
- **Real-time detector** (`detect.py`) that watches a log file or stdin and flags malicious
  connections as they arrive, at about **10,000 records/s** with a median latency of about
  **75 µs per record**
- **Shared preprocessing pipelines** (`nids.py`) reused by training, evaluation, and live
  detection, plus **micro-batched inference**, which cuts per-record latency by more than 100x
  compared with one-at-a-time prediction
- An honest **generalization test** on the official NSL-KDD test set. It contains 17 attack types
  never seen in training, and accuracy drops to about 77%.

## Dataset

This project uses [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), an improved version of
the classic KDD Cup 99 dataset that removes duplicate records and rebalances difficulty levels.

| File | Rows | Description |
|---|---|---|
| `KDD.txt` | 125,973 | Training connections (`KDDTrain+.txt` in the original release) |
| `KDDTest.txt` | 22,544 | Official test set, including attack types absent from training (`KDDTest+.txt`) |

Each row is one network connection with 41 features: protocol, service, and flag, byte counts,
connection counts, error rates, and host-based traffic stats. It also carries the original
attack-type label and a difficulty score. The label is collapsed into a binary target: `0` =
normal, `1` = attack. The training data is roughly 53% normal and 47% attack.

## Project Structure

```
.
├── nids.py                                   # Column schema, data loading, RF + SVM pipelines
├── train.py                                  # Trains both models, reports metrics, saves to models/
├── detect.py                                 # Real-time detector for streamed connection records
├── lab_-_network_intrusion_detection.ipynb   # EDA, training, model comparison, latency, generalization
├── KDD.txt                                   # Training data
├── KDDTest.txt                               # Official test data
└── requirements.txt
```

## Getting Started

```bash
pip install -r requirements.txt

# Train both models (the SVM takes about a minute) and save them to models/
python train.py

# Replay the official test set through the real-time detector
python detect.py KDDTest.txt

# Or watch a log file that is still being written to, like `tail -f`
python detect.py traffic.log --follow
```

`detect.py` accepts records with 41 features, or with the label and difficulty columns too. When
labels are present, it also reports accuracy against them. Useful options: `--model
models/svm.joblib`, `--batch-size`, `--max-wait` (how long a partial batch can wait before it's
flushed), and `--quiet` (print only the summary).

To explore the analysis, open the notebook with `jupyter lab` and run it top to bottom.

## Approach

1. **EDA**: class balance, distributions of key traffic features (`src_bytes`, `dst_bytes`,
   `count`, error rates), and correlation with the attack label.
2. **Preprocessing**: each model is a scikit-learn `Pipeline`, so the encoders are fit once and
   reused for every evaluation and for live detection.
   - Random Forest: ordinal-encoded categoricals and raw numeric features.
   - SVM: one-hot categoricals, plus log-scaled and standardized numeric features. Byte counts are
     extremely skewed, so the log scaling matters here.
   - Categories never seen in training, such as a new service, are handled instead of crashing.
3. **Modeling**: a `RandomForestClassifier` (100 trees) and an RBF-kernel `SVC` (`C=10`), both
   trained on a stratified 80/20 split of the training file.
4. **Real-time inference**: the detector reads records as they arrive and groups them into small
   micro-batches, up to 256 records or 0.5 s. It then classifies each batch in one vectorized call.
5. **Generalization check**: both models are evaluated on the official test file, whose novel
   attack signatures show whether they learned general attack *patterns* or just memorized known
   attacks.

## Results

| Model | Evaluation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Random Forest | 80/20 split | 99.90% | 99.94% | 99.85% | 99.89% |
| SVM (RBF) | 80/20 split | 99.78% | 99.85% | 99.67% | 99.76% |
| Random Forest | Official test set (17 unseen attack types) | 77.21% | 96.79% | 62.03% | 75.60% |
| SVM (RBF) | Official test set (17 unseen attack types) | 76.99% | 92.13% | 65.14% | 76.32% |

**Inference latency** (per record, measured in the notebook):

| Batch size | Random Forest | SVM |
|---|---|---|
| 1 | ~10.7 ms | ~5.3 ms |
| 32 | ~330 µs | ~620 µs |
| 256 | ~46 µs | ~135 µs |

Latency numbers vary by machine. The Random Forest trains in about 2 s versus about 70 s for the
SVM, and it is faster at the detector's batch size, so it's the detector's default model.

The gap between the two evaluations is the main takeaway. A random split tests the models on attack
signatures they have already seen, so near-perfect accuracy is expected. On the official test set,
precision stays high but recall drops: when the models flag something it's almost always an
attack, but they miss many novel attacks. These models are best treated as baselines for known
attack signatures, not detectors for novel ones.

**Top predictive features:** `src_bytes`, `dst_bytes`, `same_srv_rate`,
`dst_host_same_srv_rate`, and the connection error-rate family (`serror_rate`,
`srv_serror_rate`). Each has a clear security interpretation: scans and floods show up as unusual
traffic volume and bursts of failed or incomplete connections to the same host.

## Limitations and Next Steps

- The detector classifies NSL-KDD-style **connection records**, not raw packets. Running it on
  live traffic would need a feature extractor that builds these 41 features from a packet capture,
  for example with Zeek or a custom scapy aggregator.
- The binary target hides per-category performance (DoS, Probe, R2L, U2R). The rare R2L and U2R
  classes are likely where most misses on the test set come from.
- Possible improvements: class-weighting or resampling for rare attack types, gradient boosting,
  or an anomaly-detection model that doesn't depend on having seen an attack before.
