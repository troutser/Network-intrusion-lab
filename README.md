# Network Intrusion Detection Tool

A Python/scikit-learn tool that classifies network connection records as **normal** or
**malicious** using a Random Forest classifier trained on the NSL-KDD dataset.

## Overview

- Developed a machine learning pipeline to detect malicious network traffic from raw
  connection-level features (protocol, service, byte counts, error rates, etc.)
- Achieves ~99% accuracy on a held-out split of the training distribution, and includes an
  additional generalization test against attack types the model has never seen
- Surfaces the most predictive traffic features (e.g. `src_bytes`, `same_srv_rate`,
  connection error rates) via feature importance analysis

## Dataset

This project uses [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), an improved version of
the classic KDD Cup 99 dataset that removes duplicate records and rebalances difficulty levels.

| File | Rows | Description |
|---|---|---|
| `KDDTrain.txt` | 125,973 | Training connections |
| `KDDTest.txt` | 22,544 | Official holdout set — includes attack types absent from training |

Each row has 41 features (protocol/service/flag, byte counts, connection counts, error rates,
host-based traffic stats) plus the original attack-type label and a difficulty score. The
notebook collapses the label into a binary target: `0` = normal, `1` = attack.

## Project Structure

```
.
├── lab_-_network_intrusion_detection.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
├── KDDTrain.txt                              # Training data
├── KDDTest.txt                               # Official holdout test data
└── README.md
```

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab
jupyter lab
```

Make sure `KDDTrain.txt` and `KDDTest.txt` are in the same directory as the notebook before
running it top to bottom.

## Approach

1. **EDA** — checked class balance, distribution of key traffic features (`src_bytes`,
   `dst_bytes`, error rates), and correlation with the attack label.
2. **Preprocessing** — label-encoded the categorical features (`protocol_type`, `service`,
   `flag`) and standardized the numeric features.
3. **Modeling** — trained a `RandomForestClassifier` (100 estimators) on an 80/20 stratified
   split of the training file.
4. **Generalization check** — separately evaluated the trained model against NSL-KDD's official
   holdout file, which intentionally contains attack signatures never seen during training, to
   measure how well the model generalizes beyond memorized attack patterns.

## Results

| Evaluation | Accuracy |
|---|---|
| 80/20 split of training data (same attack types seen in training) | ~99.9% |
| Official NSL-KDD holdout file (includes unseen attack types) | ~77% |

The model performs well on seen findings and not so well on unseen findings within the holdout file, meaning this model should best be used as a baseline rather than an adaptive model.

**Top predictive features:** `src_bytes`, `dst_bytes`, `same_srv_rate`,
`dst_host_same_srv_rate`, and the connection error-rate family (`serror_rate`,
`srv_serror_rate`) — all features with a clear security interpretation around traffic volume
and failed/incomplete connection attempts.
