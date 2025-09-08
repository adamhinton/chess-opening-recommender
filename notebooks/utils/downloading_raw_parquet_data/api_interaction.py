"""
This module contains utility functions for interacting with the Hugging Face API and dataset viewer.
Functions include fetching parquet URLs and filtering them based on specific criteria.
HuggingFace has a repository of .parquet files - that is to say, 1GB datasets of Lichess games.
"""

import urllib.parse
import requests
import re
import urllib
from typing import List


def flatten_parquet_mapping(mapping: dict) -> List[str]:
    """Flatten the Hub /api/datasets/.../parquet mapping to a list of URLs."""
    urls = []
    if not isinstance(mapping, dict):
        return urls
    for subset_val in mapping.values():
        if isinstance(subset_val, dict):
            for split_val in subset_val.values():
                if isinstance(split_val, list):
                    urls.extend(split_val)
    return urls


def get_urls_from_hub_api(repo: str, hf_headers: dict) -> List[str]:
    """Call https://huggingface.co/api/datasets/{repo}/parquet (Hub API)."""
    try:
        api = f"https://huggingface.co/api/datasets/{repo}/parquet"
        r = requests.get(api, headers=hf_headers, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        urls = flatten_parquet_mapping(data)
        return urls
    except Exception:
        return []


def get_urls_from_dataset_viewer(repo: str, hf_headers: dict) -> List[str]:
    """Call dataset-viewer endpoint: https://datasets-server.huggingface.co/parquet?dataset={repo}"""
    try:
        api = "https://datasets-server.huggingface.co/parquet"
        params = {"dataset": repo}
        r = requests.get(api, headers=hf_headers, params=params, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        urls = [
            entry.get("url")
            for entry in data.get("parquet_files", [])
            if entry.get("url")
        ]
        return urls
    except Exception:
        return []


def filter_urls_for_month(urls: List[str], year: str, month_padded: str) -> List[str]:
    """Return only URLs that contain the month/year partition (decoded)."""
    out = []
    for u in urls:
        decoded = urllib.parse.unquote(u)
        if (
            f"year={year}/month={month_padded}" in decoded
            or f"year={year}/month={int(month_padded)}" in decoded
        ):
            out.append(u)

    def shard_index(u):
        m = re.search(r"(\d{1,5})\.parquet$", urllib.parse.unquote(u))
        if m:
            return int(m.group(1))
        m2 = re.search(
            r"train-(\d{1,5})-of-(\d{1,5})\.parquet", urllib.parse.unquote(u)
        )
        if m2:
            return int(m2.group(1))
        return 10**9

    return sorted(out, key=shard_index)
