#!/usr/bin/env python3
"""
Map Encyclopedic-VQA iNaturalist image IDs to local iNat21 image paths.

Usage:
    python map_evqa_inat21.py \
        --evqa_csv val.csv test.csv \
        --inat_val_json /path/to/iNat21/val.json \
        --inat_root /path/to/iNat21 \
        --output_dir mapped_csvs

Optional fallback:
    python map_evqa_inat21.py \
        --evqa_csv val.csv test.csv \
        --inat_val_json /path/to/iNat21/val.json \
        --inat_train_json /path/to/iNat21/train.json \
        --inat_root /path/to/iNat21 \
        --output_dir mapped_csvs \
        --fallback_to_train

What it does:
- reads E-VQA CSV(s)
- selects rows with dataset_name referring to iNaturalist / iNat21
- parses dataset_image_ids
- resolves each image id to a relative/local image path
- writes:
    * a new CSV with extra columns:
        - resolved_image_paths
        - resolved_image_splits
        - num_resolved_images
    * a JSON report with unresolved IDs

Notes:
- This script is robust to several possible formats of dataset_image_ids:
    * Python-list string: "[2786362, 2787001]"
    * pipe-separated: "2786362|2787001"
    * comma-separated: "2786362,2787001"
    * single id: "2786362"
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evqa_csv",
        nargs="+",
        required=True,
        help="One or more E-VQA CSV files, e.g. val.csv test.csv",
    )
    parser.add_argument(
        "--inat_val_json",
        type=str,
        required=True,
        help="Path to iNat21 val.json",
    )
    parser.add_argument(
        "--inat_train_json",
        type=str,
        default=None,
        help="Optional path to iNat21 train.json",
    )
    parser.add_argument(
        "--inat_root",
        type=str,
        required=True,
        help="Root folder containing extracted iNat21 images, e.g. /data/inat2021",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for mapped CSVs and reports",
    )
    parser.add_argument(
        "--fallback_to_train",
        action="store_true",
        help="If set, unresolved IDs after val lookup are retried in train.json",
    )
    parser.add_argument(
        "--strict_exists_check",
        action="store_true",
        help="If set, verify resolved local paths actually exist on disk",
    )
    return parser.parse_args()


def load_inat_index(json_path: str, split_name: str) -> Dict[int, Dict]:
    """
    Build a mapping:
        image_id -> {
            "split": split_name,
            "file_name": ...,
            "category_id": ...,
            "image_dir_name": ...,
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # image_id -> file_name
    img_map = {}
    for img in data["images"]:
        img_map[int(img["id"])] = {
            "file_name": img["file_name"],
            "split": split_name,
        }

    # image_id -> category_id
    ann_map = {}
    for ann in data["annotations"]:
        ann_map[int(ann["image_id"])] = int(ann["category_id"])

    # category_id -> image_dir_name
    cat_map = {}
    for cat in data["categories"]:
        cat_map[int(cat["id"])] = cat["image_dir_name"]

    out = {}
    for image_id, meta in img_map.items():
        category_id = ann_map.get(image_id)
        image_dir_name = cat_map.get(category_id) if category_id is not None else None
        out[image_id] = {
            "split": split_name,
            "file_name": meta["file_name"],
            "category_id": category_id,
            "image_dir_name": image_dir_name,
        }
    return out


def parse_dataset_image_ids(value) -> List[int]:
    """Parse dataset_image_ids from several possible string formats."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [int(x) for x in value]

    s = str(value).strip()
    if not s:
        return []

    # Try Python literal first, e.g. "[2786362, 2787001]"
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                return [int(str(x).strip()) for x in obj if str(x).strip()]
        except Exception:
            pass

    # Split on common delimiters
    parts = re.split(r"[|,;\s]+", s)
    ids = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # keep only integer-looking tokens
        if re.fullmatch(r"\d+", p):
            ids.append(int(p))
    return ids


def is_inat_row(dataset_name: str) -> bool:
    if dataset_name is None or (isinstance(dataset_name, float) and pd.isna(dataset_name)):
        return False
    s = str(dataset_name).lower()
    return ("inaturalist" in s) or ("inat" in s)


def build_relative_path(split: str, image_dir_name: Optional[str], file_name: str) -> str:
    """
    Reconstruct relative image path.

    Handles both cases:
    - file_name is just the jpg name
    - file_name already contains split/category/... or category/...
    """
    file_name = str(file_name)

    # If file_name already looks like a relative path, preserve it.
    if "/" in file_name or "\\" in file_name:
        normalized = file_name.replace("\\", "/")
        if normalized.startswith(f"{split}/"):
            return normalized
        return f"{split}/{normalized}"

    if image_dir_name:
        return f"{split}/{image_dir_name}/{file_name}"
    return f"{split}/{file_name}"


def resolve_one_id(
    image_id: int,
    val_index: Dict[int, Dict],
    train_index: Optional[Dict[int, Dict]],
    fallback_to_train: bool,
) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Returns:
        rel_path, split, meta
    """
    if image_id in val_index:
        meta = val_index[image_id]
        rel_path = build_relative_path(meta["split"], meta["image_dir_name"], meta["file_name"])
        return rel_path, meta["split"], meta

    if fallback_to_train and train_index is not None and image_id in train_index:
        meta = train_index[image_id]
        rel_path = build_relative_path(meta["split"], meta["image_dir_name"], meta["file_name"])
        return rel_path, meta["split"], meta

    return None, None, None


def process_csv(
    csv_path: Path,
    val_index: Dict[int, Dict],
    train_index: Optional[Dict[int, Dict]],
    inat_root: Path,
    output_dir: Path,
    fallback_to_train: bool,
    strict_exists_check: bool,
) -> None:
    df = pd.read_csv(csv_path)

    resolved_paths_col = []
    resolved_splits_col = []
    unresolved_ids_col = []
    num_resolved_col = []

    unresolved_report = []

    for row_idx, row in df.iterrows():
        dataset_name = row.get("dataset_name", "")
        if not is_inat_row(dataset_name):
            # Keep non-iNat rows empty; user said GLDv2 already handled.
            resolved_paths_col.append("")
            resolved_splits_col.append("")
            unresolved_ids_col.append("")
            num_resolved_col.append(0)
            continue

        image_ids = parse_dataset_image_ids(row.get("dataset_image_ids", ""))
        resolved_paths = []
        resolved_splits = []
        unresolved_ids = []

        for image_id in image_ids:
            rel_path, split, _meta = resolve_one_id(
                image_id=image_id,
                val_index=val_index,
                train_index=train_index,
                fallback_to_train=fallback_to_train,
            )

            if rel_path is None:
                unresolved_ids.append(image_id)
                continue

            full_path = inat_root / rel_path
            if strict_exists_check and not full_path.exists():
                unresolved_ids.append(image_id)
                continue

            resolved_paths.append(str(full_path))
            resolved_splits.append(split)

        resolved_paths_col.append("|".join(resolved_paths))
        resolved_splits_col.append("|".join(resolved_splits))
        unresolved_ids_col.append("|".join(map(str, unresolved_ids)))
        num_resolved_col.append(len(resolved_paths))

        if unresolved_ids:
            unresolved_report.append(
                {
                    "csv_file": str(csv_path),
                    "row_index": int(row_idx),
                    "question": row.get("question", ""),
                    "dataset_name": dataset_name,
                    "dataset_image_ids": row.get("dataset_image_ids", ""),
                    "unresolved_ids": unresolved_ids,
                }
            )

    df["resolved_image_paths"] = resolved_paths_col
    df["resolved_image_splits"] = resolved_splits_col
    df["unresolved_image_ids"] = unresolved_ids_col
    df["num_resolved_images"] = num_resolved_col

    output_csv = output_dir / f"{csv_path.stem}.mapped.csv"
    output_json = output_dir / f"{csv_path.stem}.unresolved.json"

    df.to_csv(output_csv, index=False)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(unresolved_report, f, ensure_ascii=False, indent=2)

    total_inat_rows = sum(is_inat_row(x) for x in df.get("dataset_name", []))
    fully_resolved_rows = (
        df.loc[df["dataset_name"].apply(is_inat_row), "unresolved_image_ids"]
        .fillna("")
        .eq("")
        .sum()
        if "dataset_name" in df.columns
        else 0
    )

    print(f"\nProcessed: {csv_path}")
    print(f"  iNat rows:             {total_inat_rows}")
    print(f"  fully resolved rows:   {fully_resolved_rows}")
    print(f"  output csv:            {output_csv}")
    print(f"  unresolved report:     {output_json}")


def main() -> None:
    args = parse_args()

    inat_root = Path(args.inat_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading iNat21 val index...")
    val_index = load_inat_index(args.inat_val_json, split_name="val")

    train_index = None
    if args.fallback_to_train:
        if not args.inat_train_json:
            raise ValueError("--fallback_to_train requires --inat_train_json")
        print("Loading iNat21 train index...")
        train_index = load_inat_index(args.inat_train_json, split_name="train")

    for csv_file in args.evqa_csv:
        process_csv(
            csv_path=Path(csv_file),
            val_index=val_index,
            train_index=train_index,
            inat_root=inat_root,
            output_dir=output_dir,
            fallback_to_train=args.fallback_to_train,
            strict_exists_check=args.strict_exists_check,
        )


if __name__ == "__main__":
    main()