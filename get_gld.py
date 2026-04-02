#!/usr/bin/env python3
"""
Download only the GLDv2 images needed for Encyclopedic-VQA val/test.

Revised for friendlier HTTP behavior:
- descriptive User-Agent
- polite per-request delay
- redirect-friendly GET requests
- Retry-After handling
- sequential downloading by default

Inputs:
  - EVQA val/test CSV files
  - GLDv2 train.csv from official metadata
Optional:
  - GLDv2 train_clean.csv for validation only

Outputs:
  - needed_gldv2_ids.txt
  - needed_gldv2_manifest.csv
  - downloaded images under:
      <out_dir>/images/<first_char>/<second_char>/<third_char>/<id>.jpg
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests


GLD_ID_RE = re.compile(r"\b[a-f0-9]{16}\b")

DEFAULT_USER_AGENT = (
    "GLDv2SubsetDownload/0.2 "
    "(academic research dataset download; contact: wuyu0023@e.ntu.edu.sg)"
)


def extract_gld_ids(text: str) -> List[str]:
    if text is None:
        return []
    return GLD_ID_RE.findall(str(text).lower())


def looks_like_gld_dataset_name(name: str) -> bool:
    if name is None:
        return False
    name = str(name).lower()
    return any(
        key in name for key in [
            "gld",
            "gldv2",
            "google_landmark",
            "google-landmark",
            "landmark",
        ]
    )


def read_evqa_needed_ids(evqa_csvs: Iterable[Path]) -> Set[str]:
    needed: Set[str] = set()

    for csv_path in evqa_csvs:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required_cols = {"dataset_name", "dataset_image_ids"}
            missing = required_cols - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path} is missing required columns: {sorted(missing)}"
                )

            for row in reader:
                if not looks_like_gld_dataset_name(row.get("dataset_name", "")):
                    continue
                ids = extract_gld_ids(row.get("dataset_image_ids", ""))
                needed.update(ids)

    return needed


def stream_manifest_from_train_csv(
    train_csv: Path,
    needed_ids: Set[str],
    manifest_csv: Path,
) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}

    with train_csv.open("r", encoding="utf-8", newline="") as f_in, \
         manifest_csv.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        required_cols = {"id", "url", "landmark_id"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{train_csv} is missing required columns: {sorted(missing)}"
            )

        writer = csv.DictWriter(f_out, fieldnames=["id", "url", "landmark_id"])
        writer.writeheader()

        for row in reader:
            image_id = row["id"].strip().lower()
            if image_id in needed_ids:
                clean_row = {
                    "id": image_id,
                    "url": row["url"].strip(),
                    "landmark_id": row["landmark_id"].strip(),
                }
                writer.writerow(clean_row)
                found[image_id] = clean_row

    return found


def maybe_validate_with_train_clean(train_clean_csv: Path, needed_ids: Set[str]) -> Set[str]:
    seen: Set[str] = set()

    with train_clean_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"landmark_id", "images"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{train_clean_csv} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            ids = row["images"].strip().split()
            for image_id in ids:
                image_id = image_id.lower()
                if image_id in needed_ids:
                    seen.add(image_id)

    return seen


def gld_output_path(out_dir: Path, image_id: str) -> Path:
    return out_dir / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    value = value.strip()

    if value.isdigit():
        return float(value)

    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delay = (dt - now).total_seconds()
        return max(delay, 0.0)
    except Exception:
        return None


class PoliteDownloader:
    def __init__(
        self,
        user_agent: str,
        min_delay: float = 1.0,
        jitter: float = 0.5,
        timeout: int = 30,
        max_retries: int = 4,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        })
        self.min_delay = min_delay
        self.jitter = jitter
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request_ts = 0.0

    def _sleep_between_requests(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_ts
        target = self.min_delay + random.uniform(0.0, self.jitter)
        if elapsed < target:
            time.sleep(target - elapsed)

    def get(self, url: str, stream: bool = True) -> requests.Response:
        self._sleep_between_requests()
        resp = self.session.get(
            url,
            stream=stream,
            timeout=self.timeout,
            allow_redirects=True,
        )
        self._last_request_ts = time.time()
        return resp


def download_one(
    downloader: PoliteDownloader,
    image_id: str,
    url: str,
    dest: Path,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True

    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, downloader.max_retries + 1):
        try:
            with downloader.get(url, stream=True) as r:
                final_url = str(r.url)

                if r.status_code == 200:
                    with tmp.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                f.write(chunk)
                    tmp.replace(dest)
                    return True

                retry_after = parse_retry_after(r.headers.get("Retry-After"))

                if r.status_code in {429, 500, 502, 503, 504}:
                    wait_s = retry_after if retry_after is not None else min(60.0, 2.0 ** attempt)
                    print(
                        f"[WARN] {image_id}: HTTP {r.status_code}, retrying in {wait_s:.1f}s "
                        f"(attempt {attempt}/{downloader.max_retries})",
                        file=sys.stderr,
                    )
                    time.sleep(wait_s)
                    continue

                if r.status_code == 403:
                    # 403 often means policy/rate/UA restrictions. Retry a little, but not aggressively.
                    wait_s = retry_after if retry_after is not None else 10.0 * attempt
                    print(
                        f"[WARN] {image_id}: HTTP 403 from {final_url} "
                        f"(attempt {attempt}/{downloader.max_retries}); waiting {wait_s:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait_s)
                    continue

                print(
                    f"[FAIL] {image_id}: HTTP {r.status_code} | url={url} | final_url={final_url}",
                    file=sys.stderr,
                )
                return False

        except requests.RequestException as e:
            wait_s = min(60.0, 2.0 ** attempt)
            print(
                f"[WARN] {image_id}: request error {type(e).__name__}: {e} "
                f"(attempt {attempt}/{downloader.max_retries}); waiting {wait_s:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait_s)
        finally:
            if tmp.exists() and (not dest.exists()):
                # keep partial only if you explicitly want resume support; here we remove it
                try:
                    tmp.unlink()
                except OSError:
                    pass

    print(f"[FAIL] {image_id}: exhausted retries | url={url}", file=sys.stderr)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evqa_csv",
        nargs="+",
        required=True,
        help="Path(s) to E-VQA val/test CSV files.",
    )
    parser.add_argument(
        "--gld_train_csv",
        required=True,
        help="Path to official GLDv2 train.csv metadata.",
    )
    parser.add_argument(
        "--gld_train_clean_csv",
        default=None,
        help="Optional path to official GLDv2 train_clean.csv metadata.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for outputs and optional downloads.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download images from URLs in train.csv.",
    )
    parser.add_argument(
        "--user_agent",
        default=DEFAULT_USER_AGENT,
        help="Descriptive User-Agent string with contact info.",
    )
    parser.add_argument(
        "--min_delay",
        type=float,
        default=1.0,
        help="Minimum seconds between requests.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.5,
        help="Additional random delay in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="Maximum retries per image.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evqa_csvs = [Path(p) for p in args.evqa_csv]
    train_csv = Path(args.gld_train_csv)
    train_clean_csv = Path(args.gld_train_clean_csv) if args.gld_train_clean_csv else None

    needed_ids = read_evqa_needed_ids(evqa_csvs)
    print(f"Found {len(needed_ids)} unique GLDv2 ids referenced by E-VQA val/test.")

    ids_txt = out_dir / "needed_gldv2_ids.txt"
    with ids_txt.open("w", encoding="utf-8") as f:
        for image_id in sorted(needed_ids):
            f.write(image_id + "\n")

    manifest_csv = out_dir / "needed_gldv2_manifest.csv"
    found_rows = stream_manifest_from_train_csv(train_csv, needed_ids, manifest_csv)
    print(f"Matched {len(found_rows)} / {len(needed_ids)} ids in GLDv2 train.csv.")

    missing = sorted(needed_ids - set(found_rows))
    if missing:
        missing_txt = out_dir / "missing_in_train_csv.txt"
        with missing_txt.open("w", encoding="utf-8") as f:
            for image_id in missing:
                f.write(image_id + "\n")
        print(f"Wrote {len(missing)} missing ids to {missing_txt}")

    if train_clean_csv is not None:
        seen_in_clean = maybe_validate_with_train_clean(train_clean_csv, set(found_rows))
        not_in_clean = sorted(set(found_rows) - seen_in_clean)
        print(f"Validated {len(seen_in_clean)} ids inside train_clean.csv.")
        if not_in_clean:
            bad_txt = out_dir / "not_found_in_train_clean.txt"
            with bad_txt.open("w", encoding="utf-8") as f:
                for image_id in not_in_clean:
                    f.write(image_id + "\n")
            print(f"Wrote {len(not_in_clean)} ids not found in train_clean.csv to {bad_txt}")

    if args.download:
        downloader = PoliteDownloader(
            user_agent=args.user_agent,
            min_delay=args.min_delay,
            jitter=args.jitter,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

        ok = 0
        fail = 0

        for image_id, row in found_rows.items():
            dest = gld_output_path(out_dir / "images", image_id)
            success = download_one(
                downloader=downloader,
                image_id=image_id,
                url=row["url"],
                dest=dest,
            )
            if success:
                ok += 1
            else:
                fail += 1

        print(f"Download complete: {ok} succeeded, {fail} failed.")
        if fail:
            print(
                "Some source URLs may reject scripted access or may be temporarily unavailable. "
                "Check stderr logs for final URLs and status codes."
            )

    print(f"Done. Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()