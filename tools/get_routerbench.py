"""
Utility to download, inspect, and export Martian's RouterBench dataset from Hugging Face.

Repo: https://huggingface.co/datasets/withmartian/routerbench
Files: routerbench_0shot.pkl (95MB), routerbench_5shot.pkl (163MB), routerbench_raw.pkl (1.1GB)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from huggingface_hub import hf_hub_download

MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]


def download_raw_pickle(split: Literal["0shot", "5shot", "raw"] = "0shot", local_dir: str | Path | None = None) -> Path:
    """Download the requested RouterBench pickle file from Hugging Face."""
    filename = f"routerbench_{split}.pkl"
    print(f"[*] Fetching {filename} from withmartian/routerbench on Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id="withmartian/routerbench",
        filename=filename,
        repo_type="dataset",
        local_dir=str(local_dir) if local_dir else None,
    )
    return Path(downloaded_path)


def melt_to_long_format(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Transform wide RouterBench DataFrame (~36.5k rows x 37 cols) into
    long format (~401k rows x 8 cols):
    [prompt, model, response, correct, cost, dataset, oracle_model_to_route_to, sample_id]
    """
    records = []
    for model in MODELS:
        cols_needed = [
            "sample_id",
            "prompt",
            "eval_name",
            model,
            f"{model}|model_response",
            f"{model}|total_cost",
            "oracle_model_to_route_to",
        ]
        # Safety check if all columns present
        available_cols = [c for c in cols_needed if c in df_wide.columns]
        sub = df_wide[available_cols].copy()
        rename_map = {
            "eval_name": "dataset",
            model: "correct",
            f"{model}|model_response": "response",
            f"{model}|total_cost": "cost",
        }
        sub.rename(columns=rename_map, inplace=True)
        sub["model"] = model
        records.append(sub)

    df_long = pd.concat(records, ignore_index=True)
    # Reorder columns intuitively
    standard_order = [
        "prompt",
        "model",
        "response",
        "correct",
        "cost",
        "dataset",
        "oracle_model_to_route_to",
        "sample_id",
    ]
    actual_order = [c for c in standard_order if c in df_long.columns] + [
        c for c in df_long.columns if c not in standard_order
    ]
    return df_long[actual_order]


def load_routerbench(
    split: Literal["0shot", "5shot", "raw"] = "0shot",
    data_format: Literal["long", "wide"] = "long",
    as_hf_dataset: bool = False,
    local_dir: str | Path | None = None,
):
    """
    Load RouterBench as a pandas DataFrame or Hugging Face Dataset.

    Args:
        split: '0shot' (default), '5shot', or 'raw'
        data_format: 'long' (each row is prompt+model response, ~401k rows)
                     or 'wide' (each row is 1 prompt with 11 model columns, ~36.5k rows)
        as_hf_dataset: If True, returns datasets.Dataset instead of pandas DataFrame
        local_dir: Optional custom download folder

    Returns:
        pd.DataFrame or datasets.Dataset
    """
    pickle_path = download_raw_pickle(split=split, local_dir=local_dir)
    with open(pickle_path, "rb") as f:
        df = pickle.load(f)

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if data_format == "long":
        df = melt_to_long_format(df)

    if as_hf_dataset:
        from datasets import Dataset
        return Dataset.from_pandas(df)

    return df


def main():
    parser = argparse.ArgumentParser(description="Download and inspect RouterBench from Hugging Face.")
    parser.add_argument("--split", choices=["0shot", "5shot", "raw"], default="0shot")
    parser.add_argument("--format", choices=["long", "wide"], default="long")
    parser.add_argument("--save-parquet", type=str, default="", help="Path to save output as parquet")
    parser.add_argument("--save-csv", type=str, default="", help="Path to save output as CSV")
    parser.add_argument("--preview", action="store_true", help="Print summary statistics and preview rows")

    args = parser.parse_args()

    df = load_routerbench(split=args.split, data_format=args.format)
    print(f"\n[+] Loaded RouterBench ({args.split}, format={args.format}): {df.shape[0]:,} rows x {df.shape[1]} columns")

    if args.preview or not (args.save_parquet or args.save_csv):
        print(f"Columns: {list(df.columns)}")
        if "dataset" in df.columns:
            print("\nTop 5 source datasets:")
            print(df["dataset"].value_counts().head(5))
        if "oracle_model_to_route_to" in df.columns:
            print("\nOracle model breakdown:")
            print(df["oracle_model_to_route_to"].value_counts())

    if args.save_parquet:
        out_p = Path(args.save_parquet)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_p, index=False)
        print(f"[+] Saved to parquet: {out_p} ({out_p.stat().st_size / (1024*1024):.1f} MB)")

    if args.save_csv:
        out_c = Path(args.save_csv)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_c, index=False)
        print(f"[+] Saved to CSV: {out_c} ({out_c.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
