"""Train the Random Forest and SVM detectors and save them to models/.

Usage: python train.py [--train KDD.txt] [--test KDDTest.txt]
"""

import argparse
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from nids import build_random_forest, build_svm, load_kdd


def report(name, split, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print(f'  {name:<14} {split:<16} acc {acc:.2%}  precision {prec:.2%}  recall {rec:.2%}  f1 {f1:.2%}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', default='KDD.txt')
    parser.add_argument('--test', default='KDDTest.txt')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()

    X, y, _ = load_kdd(args.train)
    X_holdout, y_holdout, _ = load_kdd(args.test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, build in [('random_forest', build_random_forest), ('svm', build_svm)]:
        model = build()
        start = time.perf_counter()
        model.fit(X_train, y_train)
        print(f'{name}: trained in {time.perf_counter() - start:.1f}s')
        report(name, '80/20 split', y_test, model.predict(X_test))
        report(name, 'official test', y_holdout, model.predict(X_holdout))
        joblib.dump(model, out / f'{name}.joblib')


if __name__ == '__main__':
    main()
