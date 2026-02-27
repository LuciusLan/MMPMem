import json
import torch
import os
from glob import glob

#wimg = glob('/data_external/InfoSeek/wiki_image/wikipedia_images_full/*/*.jpg')
os.environ['HF_HOME'] = '/data_external/hf_cache'
os.environ['CUDA_VISIBLE_DEVICES']="2"
#from transformers import AutoModelForCausalLM
import datasets
from datasets import Image as HFImage
from PIL import Image, ImageOps
import hashlib
import numpy as np
import json

from urllib.parse import urlparse, urlunparse, unquote, quote

from datasets import load_dataset, load_from_disk, Dataset, Features, Sequence, Value, Image
from tqdm import tqdm


# ----------------------------
# 1) URL normalization
# ----------------------------
def normalize_url(u: str | None) -> str | None:
    """
    Canonicalize URLs so that minor formatting differences do not prevent matching.
    - strips whitespace
    - converts protocol-relative //... to https://...
    - normalizes scheme to https for http/https
    - lowercases hostname
    - removes fragments (#...)
    - normalizes percent-encoding in the path
    """
    if u is None:
        return None
    u = u.strip()
    if not u:
        return None

    if u.startswith("//"):
        u = "https:" + u

    p = urlparse(u)

    scheme = p.scheme.lower()
    if scheme in ("", "http", "https"):
        scheme = "https"

    netloc = (p.netloc or "").lower()

    # Normalize path percent-encoding (decode then re-encode deterministically)
    path = quote(unquote(p.path), safe="/:@&+$,;=~*'()!-._")

    # Keep query (rare on Wikimedia image URLs, but safe)
    query = p.query

    # Drop params + fragment
    return urlunparse((scheme, netloc, path, "", query, ""))


# ----------------------------
# 2) Load your article JSON (object keyed by article URL)
# ----------------------------
def load_article_object_json(path: str) -> list[dict]:
    """
    Your JSON is a dict:
      { article_url: {title:..., sections:..., image_urls:[...], ...}, ... }

    This converts it into a list of rows with a dedicated article_url field.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = []
    for article_url, entry in obj.items():
        row = dict(entry)  # shallow copy
        row["article_url"] = article_url
        # ensure image_urls exists and is a list
        img_urls = row.get("image_urls", [])
        row["image_urls"] = img_urls if isinstance(img_urls, list) else []
        rows.append(row)
    return rows


# ----------------------------
# 3) Build the join structures
# ----------------------------
def collect_article_image_url_set(article_rows: list[dict]) -> set[str]:
    url_set = set()
    for r in article_rows:
        for u in r.get("image_urls", []):
            nu = normalize_url(u)
            if nu is not None:
                url_set.add(nu)
    return url_set


def build_image_url_to_index(image_ds, article_url_set: set[str]) -> dict[str, int]:
    """
    Map normalized image_url -> row index in the image dataset.
    If duplicates exist, this keeps the first occurrence (deterministic).
    """
    url2idx: dict[str, int] = {}
    for i in tqdm(range(len(image_ds)), desc="Indexing image dataset (url->idx)"):
        u = normalize_url(image_ds[i]["image_urls"])
        if u is None:
            continue
        if u in article_url_set and u not in url2idx:
            url2idx[u] = i
    return url2idx

def build_image_url_to_index_full_column(image_ds, article_url_set):
    urls = image_ds["image_urls"]  # materializes full column as a Python list
    
    url2idx = {}
    for i, u in enumerate(tqdm(urls, desc="Indexing full image_url column")):
        if isinstance(u, list):
            nu_list = [normalize_url(uu) for uu in u]
            for nu in nu_list:
                if nu and nu in article_url_set and nu not in url2idx:
                    url2idx[nu] = i
        else:
            nu = normalize_url(u)
            if nu and nu in article_url_set and nu not in url2idx:
                url2idx[nu] = i
    return url2idx


# ----------------------------
# 4) Partition: matched articles vs mismatched articles
# ----------------------------
def split_articles_with_pixels(article_rows: list[dict], image_ds:Dataset, url2idx: dict[str, int]):
    matched_articles = []
    mismatched_articles = []

    for r in tqdm(article_rows, desc="Matching articles to image pixels"):
        img_urls = r.get("image_urls", [])

        pixels_aligned = []
        captions_aligned = []
        any_matched = False

        for u in img_urls:
            nu = normalize_url(u)
            idx = url2idx.get(nu) if nu is not None else None

            if idx is None:
                pixels_aligned.append(None)
                #captions_aligned.append(None)
            else:
                ex = image_ds[idx]
                # ex["image"] is typically a PIL.Image object (datasets.Image feature)
                pixels_aligned.append(ex["image"])
                #captions_aligned.append(ex.get("caption"))
                any_matched = True

        r_out = dict(r)
        r_out["image_pixels"] = pixels_aligned
        #r_out["image_captions"] = captions_aligned

        if any_matched:
            matched_articles.append(r_out)
        else:
            mismatched_articles.append(r_out)

    return matched_articles, mismatched_articles

def split_articles_with_pixels_sharded(
    article_rows: list[dict],
    image_ds,
    url2idx: dict[str, int],
    out_dir: str,
    shard_size: int = 10_000,
):
    """
    Writes two sharded corpora:
      - matched_articles_XXXXX.parquet: articles with >=1 matched image
      - mismatched_articles_XXXXX.parquet: articles with 0 matched images

    Adds:
      - image_pixels: list aligned to image_urls; entries are dicts (decode=False) or None
      - image_captions: list aligned to image_urls; entries are str or None
    """
    os.makedirs(out_dir, exist_ok=True)
    matched_dir = os.path.join(out_dir, "matched_articles_parquet")
    #mismatched_dir = os.path.join(out_dir, "mismatched_articles_parquet")
    os.makedirs(matched_dir, exist_ok=True)
    #os.makedirs(mismatched_dir, exist_ok=True)

    # Strongly recommended: prevent decoding to PIL during the join
    # if "image" in image_ds.column_names:
    #     try:
    #         image_ds = image_ds.cast_column("image", Image(decode=False))
    #     except Exception:
    #         # If casting fails for any reason, we still proceed.
    #         # The remaining optimizations (batch select + column access) still help.
    #         pass

    n = len(article_rows)
    shard_id = 0

    for start in tqdm(range(0, n, shard_size), desc="Processing article shards"):
        end = min(start + shard_size, n)
        shard = article_rows[start:end]

        # 1) Collect unique normalized URLs in this shard that have an index
        shard_urls = set()
        for r in shard:
            for u in r.get("image_urls", []):
                nu = normalize_url(u)
                if nu is not None and nu in url2idx:
                    shard_urls.add(nu)

        # 2) Prefetch the needed image rows once per shard
        payload_by_url = {}
        if shard_urls:
            indices = [url2idx[u] for u in shard_urls]
            # One gather per shard
            subset = image_ds.select(indices)

            # Column-wise access (fast) from the subset
            sub_urls = subset["image_urls"]
            #sub_imgs = subset["image"] if "image" in subset.column_names else [None] * len(sub_urls)
            sub_imgs = subset["image_pixels"] if "image_pixels" in subset.column_names else [None] * len(sub_urls)
            #sub_caps = subset["caption"] if "caption" in subset.column_names else [None] * len(sub_urls)

            for u, img in tqdm(zip(sub_urls, sub_imgs)):
                if isinstance(u ,list) and isinstance(img ,list):
                    for uu, ind_img in zip(u, img):
                        nu = normalize_url(uu)
                        if nu is not None:
                            payload_by_url[nu] = ind_img
                else:
                    nu = normalize_url(u)
                    if nu is not None:
                        payload_by_url[nu] = img

        # 3) Build rows and immediately write shard outputs
        matched_rows = []
        mismatched_rows = []

        for r in shard:
            img_urls = r.get("image_urls", [])

            pixels_aligned = []
            #captions_aligned = []
            any_matched = False

            for u in img_urls:
                nu = normalize_url(u)
                if nu is None:
                    pixels_aligned.append(None)
                    #captions_aligned.append(None)
                    continue

                payload = payload_by_url.get(nu)
                if payload is None:
                    pixels_aligned.append(None)
                    #captions_aligned.append(None)
                else:
                    #img, cap = payload
                    pixels_aligned.append(payload)
                    #captions_aligned.append(cap)
                    any_matched = True

            r_out = dict(r)
            r_out["image_pixels"] = pixels_aligned
            #r_out["image_captions"] = captions_aligned

            if any_matched:
                matched_rows.append(r_out)
            else:
                mismatched_rows.append(r_out)

        matched_path = os.path.join(matched_dir, f"matched_articles_{shard_id:05d}.parquet")
        #mismatched_path = os.path.join(mismatched_dir, f"mismatched_articles_{shard_id:05d}.parquet")
        base_features = Dataset.from_list([article_rows[0]]).features

        # Extend with your new columns
        features = Features({
            **base_features,
            "image_pixels": Sequence(Image(decode=False)),
            #"image_captions": Sequence(Value("string")),
        })
        if matched_rows:
            Dataset.from_list(matched_rows, features=features).to_parquet(matched_path)
        else:
            # Write an empty shard only if you need strict shard numbering; otherwise skip.
            pass

        if mismatched_rows:
            print(f"Mismatched :{len(mismatched_rows)}")

        shard_id += 1

    print(f"Wrote matched shards to: {matched_dir}")
    #print(f"Wrote mismatched shards to: {mismatched_dir}")

# ----------------------------
# 5) Partition: mismatched images (image rows not in any article image_urls)
# ----------------------------
def mismatched_image_subset(image_ds:Dataset, article_url_set: set[str]):
    # Note: uses a closure capturing article_url_set (OK; single-process recommended for very large sets).
    return image_ds.filter(lambda ex: (normalize_url(ex["image_url"]) not in article_url_set))


# ----------------------------
# 6) End-to-end driver
# ----------------------------
def run(
    article_json_path: str,
    image_dataset_name: str,
    image_split: str = "train",
    out_dir: str = "./join_outputs",
):
    # Load articles
    article_rows = load_article_object_json(article_json_path)

    # Build article URL set
    article_url_set = collect_article_image_url_set(article_rows)
    print(f"Unique article image URLs (normalized): {len(article_url_set):,}")

    # Load image dump dataset
    #image_ds = load_dataset(image_dataset_name, split=image_split)

    image_ds = load_from_disk(image_dataset_name)
    #image_ds = image_ds.cast_column('image', HFImage(decode=True))
    print(f"Image dataset rows: {len(image_ds):,}")

    # Build url->idx for matched candidates
    url2idx = build_image_url_to_index_full_column(image_ds, article_url_set)
    print(f"Matched unique image URLs (url->idx entries): {len(url2idx):,}")

    # Split articles
    #matched_articles, mismatched_articles = split_articles_with_pixels(article_rows, image_ds, url2idx)
    split_articles_with_pixels_sharded(article_rows, image_ds, url2idx, out_dir, 50_000)
    #Dataset.from_list(mismatched_articles) # .save_to_disk(os.path.join(out_dir, "mismatched_articles_no_images"))
    #mismatched_images.save_to_disk(os.path.join(out_dir, "mismatched_images_no_articles"))

    print(f"Saved to: {out_dir}")



if __name__ == "__main__":
    # Example:
    run(
      #article_json_path="/data_external/evqa/encyclopedic_kb_wiki.json",
      article_json_path="/data_external/InfoSeek/wiki_100_dict_v4.json",
      image_dataset_name="/data_external/evqa/image_kb",
      image_split="train",
      out_dir="/data_external/InfoSeek/merged_kb"
    )
    pass
