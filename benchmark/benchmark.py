"""LightGBM benchmark on the Credit Card Fraud Detection dataset (Lab 16).

Usage:
    python3 benchmark.py [--data ~/ml-benchmark/creditcard.csv] [--out benchmark_result.json]

If the CSV is missing, the same dataset is fetched from OpenML (id 1597) as a fallback,
so the benchmark still runs without a Kaggle token.
"""
import argparse
import json
import os
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42


def load_data(path):
    t0 = time.perf_counter()
    if os.path.exists(path):
        df = pd.read_csv(path)
        source = path
    else:
        from sklearn.datasets import fetch_openml
        print(f"[warn] {path} not found -> fetching dataset from OpenML (id 1597)")
        df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
        source = "openml:1597"
    X = df.drop(columns=["Class"]).astype(np.float32)
    y = df["Class"].astype(int)
    return X, y, time.perf_counter() - t0, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    args = parser.parse_args()

    print("=" * 60)
    print("LightGBM CPU Benchmark - Credit Card Fraud Detection")
    print("=" * 60)

    X, y, load_time, source = load_data(args.data)
    print(f"Loaded {len(X):,} rows x {X.shape[1]} features in {load_time:.3f}s "
          f"(fraud ratio {y.mean():.4%})")

    # 70% train / 10% validation (early stopping) / 20% test, stratified
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=0.125, stratify=y_trval, random_state=SEED)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        # Small, regularised trees: larger trees overfit the ~400 fraud rows within a few rounds
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=100,
        reg_lambda=5.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        # Early-stop on AUC only: with 0.17% positives the default binary_logloss
        # worsens after the first trees and stops training at iteration 1.
        metric="auc",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(100)],
    )
    train_time = time.perf_counter() - t0
    best_iter = int(model.best_iteration_ or model.n_estimators)
    print(f"Training done in {train_time:.3f}s, best iteration = {best_iter}")

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
    }

    # Inference latency: 1 row, repeated to get a stable median
    one_row = X_test.iloc[[0]]
    for _ in range(10):  # warm-up
        model.predict_proba(one_row)
    lat = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        lat.append(time.perf_counter() - t0)
    latency_ms = float(np.median(lat) * 1000)
    latency_p95_ms = float(np.percentile(lat, 95) * 1000)

    # Inference throughput: batch of 1000 rows
    batch = X_test.iloc[:1000]
    runs = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict_proba(batch)
        runs.append(time.perf_counter() - t0)
    batch_time = float(np.median(runs))
    throughput = len(batch) / batch_time

    result = {
        "dataset": {"source": source, "rows": int(len(X)), "features": int(X.shape[1]),
                    "train_rows": int(len(X_train)), "val_rows": int(len(X_val)),
                    "test_rows": int(len(X_test))},
        "environment": {"hostname": platform.node(), "python": platform.python_version(),
                        "lightgbm": lgb.__version__, "cpu_count": os.cpu_count()},
        "load_data_time_s": round(load_time, 4),
        "training_time_s": round(train_time, 4),
        "best_iteration": best_iter,
        **{k: round(float(v), 6) for k, v in metrics.items()},
        "inference_latency_1row_ms": round(latency_ms, 4),
        "inference_latency_1row_p95_ms": round(latency_p95_ms, 4),
        "inference_1000rows_time_ms": round(batch_time * 1000, 4),
        "inference_throughput_rows_per_s": round(throughput, 1),
    }

    print("-" * 60)
    rows = [
        ("Thoi gian load data", f"{load_time:.3f} s"),
        ("Thoi gian training", f"{train_time:.3f} s"),
        ("Best iteration", best_iter),
        ("AUC-ROC", f"{metrics['auc_roc']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("F1-Score", f"{metrics['f1_score']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("Inference latency (1 row)", f"{latency_ms:.3f} ms (p95 {latency_p95_ms:.3f} ms)"),
        ("Inference throughput (1000 rows)",
         f"{batch_time * 1000:.2f} ms -> {throughput:,.0f} rows/s"),
    ]
    for name, val in rows:
        print(f"{name:<34}| {val}")
    print("-" * 60)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
