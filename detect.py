"""Classify a live stream of NSL-KDD connection records and flag malicious traffic.

Reads comma-separated records from a file or stdin as they arrive, classifies them in small
batches, and prints an alert for each connection predicted to be an attack. With --follow it keeps
watching the file for new lines, like `tail -f`, so it can sit on a log that's still being written.

Usage:
    python detect.py KDDTest.txt                         # replay a capture
    python detect.py traffic.log --follow                # watch a growing log file
    some_producer | python detect.py -                   # read from stdin
"""

import argparse
import sys
import time

import joblib
import numpy as np
import pandas as pd

from nids import CATEGORICAL, COLUMNS, FEATURES, NUMERIC


def read_lines(stream, follow):
    """Yield lines from stream; with follow=True, wait for new lines instead of stopping at EOF.

    While waiting it yields None, so the batcher gets a chance to flush a partial batch.
    """
    while True:
        line = stream.readline()
        if line:
            yield line
        elif follow:
            time.sleep(0.1)
            yield None
        else:
            return


def batches(lines, batch_size, max_wait):
    """Group lines into batches, flushing early if a batch has waited max_wait seconds."""
    batch, started = [], time.perf_counter()
    for line in lines:
        if line is not None:
            if not batch:
                started = time.perf_counter()
            batch.append(line)
        if len(batch) >= batch_size or (batch and time.perf_counter() - started >= max_wait):
            yield batch
            batch, started = [], time.perf_counter()
    if batch:
        yield batch


def parse(batch):
    """Turn raw CSV lines into a feature frame (and labels, if the records carry them)."""
    rows = [line.rstrip('\n').split(',') for line in batch if line.strip()]
    labeled = len(rows[0]) == len(COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS if labeled else FEATURES)
    df[NUMERIC] = df[NUMERIC].astype(np.float32)
    labels = (df['label'] != 'normal').to_numpy() if labeled else None
    return df[FEATURES], labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', help="file of connection records, or '-' for stdin")
    parser.add_argument('--model', default='models/random_forest.joblib')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-wait', type=float, default=0.5, help='seconds before a partial batch is flushed')
    parser.add_argument('--follow', action='store_true', help='keep watching the file for new records')
    parser.add_argument('--quiet', action='store_true', help='print only the summary, not each alert')
    args = parser.parse_args()

    model = joblib.load(args.model)
    # Spinning up worker threads costs more than it saves on small micro-batches
    if hasattr(model[-1], 'n_jobs'):
        model[-1].n_jobs = 1
    stream = sys.stdin if args.source == '-' else open(args.source)

    seen = alerts = correct = labeled = 0
    latencies = []
    start = time.perf_counter()
    try:
        for batch in batches(read_lines(stream, args.follow), args.batch_size, args.max_wait):
            t0 = time.perf_counter()
            X, y = parse(batch)
            pred = model.predict(X)
            latencies.append((time.perf_counter() - t0) / len(X))

            if not args.quiet:
                for i in np.flatnonzero(pred):
                    row = X.iloc[i]
                    print(f'ALERT record {seen + i}: {row.protocol_type}/{row.service} flag={row.flag} '
                          f'src_bytes={int(row.src_bytes)} serror_rate={row.serror_rate:.2f}')
            seen += len(X)
            alerts += int(pred.sum())
            if y is not None:
                correct += int((pred == y).sum())
                labeled += len(y)
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not sys.stdin:
            stream.close()

    elapsed = time.perf_counter() - start
    per_record_us = np.array(latencies) * 1e6
    print(f'\nClassified {seen:,} connections in {elapsed:.2f}s ({seen / elapsed:,.0f} records/s)', file=sys.stderr)
    print(f'Flagged {alerts:,} as malicious ({alerts / max(seen, 1):.1%})', file=sys.stderr)
    if len(per_record_us):
        print(f'Per-record latency: median {np.median(per_record_us):.1f} us, '
              f'p99 {np.percentile(per_record_us, 99):.1f} us', file=sys.stderr)
    if labeled:
        print(f'Accuracy against labels in the stream: {correct / labeled:.2%}', file=sys.stderr)


if __name__ == '__main__':
    main()
