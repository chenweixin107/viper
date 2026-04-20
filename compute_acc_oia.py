import argparse
import json
from pathlib import Path

import pandas as pd


LABEL_KEYS = ["stop", "forward", "left", "right"]


def to_binary(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"Invalid int label: {value}")
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        raise ValueError(f"Invalid float label: {value}")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no"}:
            return 0
        if lowered in {"1", "true", "yes"}:
            return 1
    raise ValueError(f"Cannot convert to binary label: {value!r}")


def extract_json_object(text):
    if isinstance(text, dict):
        return text

    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(text[start:end + 1])


def parse_label_map(text):
    payload = extract_json_object(text)
    if set(payload.keys()) != set(LABEL_KEYS):
        raise ValueError(f"Unexpected keys: {sorted(payload.keys())}")
    return {key: to_binary(payload[key]) for key in LABEL_KEYS}


def safe_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def compute_metrics(rows):
    per_label = {}
    for key in LABEL_KEYS:
        tp = fp = fn = tn = 0
        for pred, gold in rows:
            pred_value = pred[key]
            gold_value = gold[key]
            if pred_value == 1 and gold_value == 1:
                tp += 1
            elif pred_value == 1 and gold_value == 0:
                fp += 1
            elif pred_value == 0 and gold_value == 1:
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        accuracy = 0.0 if total == 0 else (tp + tn) / total
        f1 = safe_f1(tp, fp, fn)
        per_label[key] = {
            "accuracy": accuracy,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    mean_accuracy = sum(metrics["accuracy"] for metrics in per_label.values()) / len(LABEL_KEYS)
    mean_f1 = sum(metrics["f1"] for metrics in per_label.values()) / len(LABEL_KEYS)
    exact_match = 0.0 if not rows else sum(pred == gold for pred, gold in rows) / len(rows)
    return per_label, mean_accuracy, mean_f1, exact_match


def main():
    parser = argparse.ArgumentParser(description="Compute SDD-OIA metrics for ViperGPT outputs.")
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--split", type=str, default="")
    args = parser.parse_args()

    df = pd.read_csv(args.result_path)
    parsed_rows = []
    parse_failures = 0

    for _, row in df.iterrows():
        try:
            pred = parse_label_map(row["result"])
            gold = parse_label_map(row["answer"])
            parsed_rows.append((pred, gold))
        except Exception:
            parse_failures += 1

    total = len(df)
    parsed_total = len(parsed_rows)
    parse_failure_freq = 0.0 if total == 0 else parse_failures / total
    per_label, mean_accuracy, mean_f1, parsed_exact_match = (
        compute_metrics(parsed_rows) if parsed_rows else ({}, 0.0, 0.0, 0.0)
    )
    exact_match = 0.0 if total == 0 else sum(pred == gold for pred, gold in parsed_rows) / total

    if args.split:
        print(f"Split: {args.split}")
    print(f"Result path: {args.result_path}")
    print(f"Parsed rows: {parsed_total}/{total}")
    print(f"parse_failure_freq: {parse_failure_freq:.6f}")
    for key in LABEL_KEYS:
        metrics = per_label.get(key, {"accuracy": 0.0, "f1": 0.0})
        print(f"{key}_accuracy: {metrics['accuracy']:.6f}")
        print(f"{key}_f1: {metrics['f1']:.6f}")
    print(f"label-level mean accuracy: {mean_accuracy:.6f}")
    print(f"label-level mean F1: {mean_f1:.6f}")
    print(f"label-level exact match (parsed rows): {parsed_exact_match:.6f}")
    print(f"label-level exact match: {exact_match:.6f}")


if __name__ == "__main__":
    main()
