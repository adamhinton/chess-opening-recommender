"""
This module contains utility functions for downloading parquet files and probing fallback URLs.
It includes logic for handling HTTP requests and saving files locally.
HuggingFace has a repository of .parquet files - that is to say, 1GB datasets of Lichess games.
"""

import requests
from pathlib import Path
from typing import List


def try_head(url: str, hf_headers: dict, timeout: int = 20) -> bool:
    """Quick HEAD-ish check (GET with stream and immediate close) to see if URL exists."""
    try:
        r = requests.get(url, headers=hf_headers, stream=True, timeout=timeout)
        if r.status_code == 200:
            r.raw.read(1)
            r.close()
            return True
        r.close()
        return False
    except Exception:
        return False


def probe_fallback_urls(
    repo: str,
    year: str,
    month: str,
    max_attempts: int,
    patterns: List[str],
    hf_headers: dict,
) -> List[str]:
    """If APIs fail, try probing plausible URL patterns until 404."""
    found = []
    for pattern in patterns:
        if "{total" in pattern:
            for total_guess in range(1, 201):
                consecutive_not_found = 0
                for idx in range(max_attempts):
                    url = pattern.format(
                        repo=repo, year=year, month=month, idx=idx, total=total_guess
                    )
                    if try_head(url, hf_headers):
                        found.append(url)
                        consecutive_not_found = 0
                    else:
                        consecutive_not_found += 1
                        break
                if found:
                    return found
        else:
            for idx in range(max_attempts):
                url = pattern.format(repo=repo, year=year, month=month, idx=idx)
                if try_head(url, hf_headers):
                    found.append(url)
                else:
                    break
            if found:
                return found
    return found


def download_file(
    url: str, dest: Path, hf_headers: dict, chunk_size: int = 1024 * 32
) -> bool:
    """Download url -> dest. Return True on success, False on 404 or error."""
    try:
        r = requests.get(url, headers=hf_headers, stream=True, timeout=60)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        try:
            if r is not None:
                r.close()
        except Exception:
            pass
        print(f"  download error: {e}")
        return False
