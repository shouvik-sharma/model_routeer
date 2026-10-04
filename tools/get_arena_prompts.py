#!/usr/bin/env python3
"""
Download and prepare English user prompts from LMSYS Chatbot Arena.

Dataset: lmsys/chatbot_arena_conversations (Gated dataset on Hugging Face)
Requirement: Accept license at https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
             and provide HF_TOKEN via environment variable, `huggingface-cli login`, or --token flag.

Output:
  data/arena/arena_prompts_english.parquet
  data/arena/arena_prompts_english.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import get_token

OUTPUT_DIR = Path("data/arena")
DATASET_NAME = "lmsys/chatbot_arena_conversations"


def parse_conversation(value: Any) -> list:
    """Safely convert a conversation cell into a list of message dictionaries."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def extract_first_user_prompt(conversation: Any) -> str | None:
    """
    Extract the first meaningful user prompt from an OpenAI-style message list.

    Supports common LMSYS message shapes:
      [{"role": "user", "content": "..."}]
      [{"role": "user", "text": "..."}]
      ["user", "..."]
    """
    messages = parse_conversation(conversation)

    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "")).lower()
            content = message.get("content", message.get("text", ""))
            if role in {"user", "human"} and content:
                return str(content).strip()

        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            role, content = message[0], message[1]
            if str(role).lower() in {"user", "human"} and content:
                return str(content).strip()

    return None


def first_available(df: pd.DataFrame, columns: list[str]) -> str | None:
    """Return first existing DataFrame column from a list."""
    return next((col for col in columns if col in df.columns), None)


def download_and_prepare_arena(token: str | None = None) -> tuple[pd.DataFrame, Path, Path]:
    """Download, filter to English, and export Chatbot Arena conversations."""
    hf_token = token or os.environ.get("HF_TOKEN") or get_token()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / "arena_prompts_english.parquet"
    csv_path = OUTPUT_DIR / "arena_prompts_english.csv"

    print(f"[*] Loading '{DATASET_NAME}' from Hugging Face...")
    try:
        ds = load_dataset(DATASET_NAME, split="train", token=hf_token)
    except Exception as e:
        err_msg = str(e)
        if "gated" in err_msg.lower() or "authenticated" in err_msg.lower():
            print("\n" + "!" * 78)
            print("ACCESS ERROR: 'lmsys/chatbot_arena_conversations' is a gated dataset.")
            print("To access this dataset:")
            print("  1. Go to: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations")
            print("  2. Click 'Agree and access repository' to accept their terms.")
            print("  3. Create a free Read Token at: https://huggingface.co/settings/tokens")
            print("  4. Run again with:")
            print("       python -m tools.get_arena_prompts --token YOUR_HF_TOKEN")
            print("     or set $env:HF_TOKEN='YOUR_HF_TOKEN'")
            print("!" * 78 + "\n")
        raise

    df = ds.to_pandas()
    print(f"[+] Downloaded raw rows: {len(df):,}")
    print(f"[+] Columns: {df.columns.tolist()}")

    # Determine conversation columns
    conv_a_col = first_available(df, ["conversation_a", "conversation", "messages"])
    conv_b_col = first_available(df, ["conversation_b"])

    if conv_a_col is None:
        raise ValueError(f"Could not find conversation column in: {df.columns.tolist()}")

    # Extract user prompts
    df["prompt_a"] = df[conv_a_col].map(extract_first_user_prompt)
    if conv_b_col:
        df["prompt_b"] = df[conv_b_col].map(extract_first_user_prompt)
        df["prompt"] = df["prompt_a"].fillna(df["prompt_b"])
    else:
        df["prompt"] = df["prompt_a"]

    # Locate language column
    language_col = first_available(df, ["language", "lang", "detected_language"])
    if language_col:
        print(f"[+] Filtering on language column: '{language_col}'")
        df_en = df[
            (df[language_col].astype(str).str.lower() == "en")
            & df["prompt"].notna()
            & (df["prompt"].str.len() >= 10)
        ].copy()
    else:
        print("[!] No language column found; keeping non-empty prompts (len >= 10)")
        df_en = df[df["prompt"].notna() & (df["prompt"].str.len() >= 10)].copy()

    # Deduplicate prompts
    df_en = df_en.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    print(f"[+] English unique prompts: {len(df_en):,}")

    # Map metadata columns
    model_a_col = first_available(df_en, ["model_a", "model_name_a"])
    model_b_col = first_available(df_en, ["model_b", "model_name_b"])
    winner_col = first_available(df_en, ["winner", "vote", "winner_model_a"])
    moderation_col = first_available(df_en, ["openai_moderation", "openai_moderation_a", "moderation_tag"])
    toxicity_col = first_available(df_en, ["toxic_chat_tag", "toxicity_tag", "toxic_tag"])
    timestamp_col = first_available(df_en, ["timestamp", "tstamp", "date"])

    router_df = pd.DataFrame({
        "prompt_id": range(len(df_en)),
        "prompt": df_en["prompt"],
        "language": "en",

        # Existing Arena preference information
        "model_a": df_en[model_a_col] if model_a_col else None,
        "model_b": df_en[model_b_col] if model_b_col else None,
        "human_preference": df_en[winner_col] if winner_col else None,

        # Safety signals
        "moderation_tag": df_en[moderation_col] if moderation_col else None,
        "toxicity_tag": df_en[toxicity_col] if toxicity_col else None,
        "timestamp": df_en[timestamp_col] if timestamp_col else None,

        # Routing and classification labels
        "task_type": None,
        "task_definition_explicit": None,
        "context_completeness": None,
        "output_contract": None,
        "constraint_set": None,
        "requires_tools": None,
        "requires_local_only": None,
        "modality": "text",
        "risk_level": None,
        "complexity_score": None,
        "ambiguity_score": None,
    })

    router_df.to_parquet(parquet_path, index=False)
    router_df.to_csv(csv_path, index=False)

    print(f"[+] Saved {len(router_df):,} English prompts")
    print(f"    Parquet: {parquet_path}")
    print(f"    CSV:     {csv_path}")

    return router_df, parquet_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare LMSYS Chatbot Arena English prompts.")
    parser.add_argument("--token", type=str, default="", help="Hugging Face User Access Token")
    args = parser.parse_args()

    download_and_prepare_arena(token=args.token or None)


if __name__ == "__main__":
    main()
