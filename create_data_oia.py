import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ROOTS = {
    "SDDOIA": Path("/data/common/weixinchen/llm_pc/datasets/SDDOIA"),
    "BDDOIA": Path("/data/common/weixinchen/llm_pc/datasets/BDDOIA"),
}
DEFAULT_PROMPT_PATH = Path("/home/weixinchen/llm_pc/config/task_prompt_viper.json")
DEFAULT_OUTPUT_DIR = Path("/home/weixinchen/viper/data")
LABEL_KEYS = ["stop", "forward", "left", "right"]
COLUMNS = ["index", "sample_id", "possible_answers", "query_type", "query", "answer", "image_name"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_query(prompt_path: Path, task_name: str) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts[task_name]


def render_answer(label_values) -> str:
    if len(label_values) != len(LABEL_KEYS):
        raise ValueError(f"Expected {len(LABEL_KEYS)} labels, got {len(label_values)}")
    answer = {key: int(value) for key, value in zip(LABEL_KEYS, label_values)}
    return json.dumps(answer, separators=(",", ":"))


def find_image_for_json(json_path: Path) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for {json_path.name}")


def build_rows(split_dir: Path, query: str):
    rows = []
    json_files = sorted(split_dir.glob("*.json"))
    for index, json_path in enumerate(json_files):
        image_path = find_image_for_json(json_path)

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        answer = render_answer(data["label"])
        rows.append([index, index, "", "", query, answer, str(image_path)])
    return rows


def main():
    parser = argparse.ArgumentParser(description="Create OIA CSV files for ViperGPT.")
    parser.add_argument("--task", choices=["SDDOIA", "BDDOIA"], default="SDDOIA")
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dataset_root = args.dataset_root or DEFAULT_DATASET_ROOTS[args.task]
    split_dir = dataset_root / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    query = load_query(args.prompt_path, args.task)
    rows = build_rows(split_dir, query)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.task}_{args.split}.csv"
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
