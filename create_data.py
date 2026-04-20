import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


DEFAULT_DATASET_ROOT = Path("/data/common/weixinchen/llm_pc/datasets")
DEFAULT_PROMPT_FILE = Path("/home/weixinchen/llm_pc/config/task_prompt_viper.json")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TASK_ALIASES = {
    "KandLogic_3obj_unseen_colors": "KandLogic_3obj",
    "KandLogic_3obj_unseen_shapes": "KandLogic_3obj",
    "MNLogic_XOR_3digit_red": "MNLogic_XOR_3digit",
    "MNLogic_XOR_3digit_rot15": "MNLogic_XOR_3digit",
    "MNLogic_XOR_3digit_rot30": "MNLogic_XOR_3digit",
    "MNLogic_XOR_3digit_rot45": "MNLogic_XOR_3digit",
    "MNMath_Add_3digit_red": "MNMath_Add_3digit",
    "MNMath_Add_3digit_rot15": "MNMath_Add_3digit",
    "MNMath_Add_3digit_rot30": "MNMath_Add_3digit",
    "MNMath_Add_3digit_rot45": "MNMath_Add_3digit",
}

COLUMNS = ["index", "sample_id", "possible_answers", "query_type", "query", "answer", "image_name"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create CSV inputs for ViperGPT.")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task_alias", type=str, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def sort_stems(stems):
    def key_fn(stem):
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    return sorted(stems, key=key_fn)


def resolve_task_alias(data_name, explicit_alias=None):
    if explicit_alias is not None:
        return explicit_alias
    return TASK_ALIASES.get(data_name, data_name)


def list_base_names(split_dir, use_json_labels):
    suffix = ".json" if use_json_labels else ".joblib"
    stems = [path.stem for path in split_dir.glob(f"*{suffix}")]
    if not stems:
        raise FileNotFoundError(f"No label files with suffix '{suffix}' found in {split_dir}")
    return sort_stems(stems)


def load_query(task_name):
    with open(DEFAULT_PROMPT_FILE, "r") as handle:
        queries = json.load(handle)
    try:
        return queries[task_name]
    except KeyError as error:
        raise KeyError(f"Missing Viper prompt for task '{task_name}' in {DEFAULT_PROMPT_FILE}") from error


def load_label(label_path, use_json_labels):
    if use_json_labels:
        with open(label_path, "r") as handle:
            data = json.load(handle)
        return bool(data["label"])

    data = joblib.load(label_path)
    return data["label"]


def build_rows(split_dir, query, use_json_labels):
    rows = []
    for index, base_name in enumerate(list_base_names(split_dir, use_json_labels)):
        image_path = split_dir / f"{base_name}.png"
        label_path = split_dir / f"{base_name}{'.json' if use_json_labels else '.joblib'}"

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        label = load_label(label_path, use_json_labels)
        rows.append([index, index, "", "", query, str(label), str(image_path)])
    return rows


def main():
    args = parse_args()

    dataset_dir = args.dataset_root / args.data_name / args.split
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset split directory does not exist: {dataset_dir}")

    task_name = resolve_task_alias(args.data_name, args.task_alias)
    query = load_query(task_name)
    use_json_labels = "cle4evr" in task_name.lower()
    output_csv = args.output_csv or (DEFAULT_OUTPUT_DIR / f"{args.data_name}.csv")

    rows = build_rows(dataset_dir, query, use_json_labels)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
