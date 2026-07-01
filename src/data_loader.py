"""
data_loader.py
--------------

Streams candidate data from JSONL.GZ files.

Designed for datasets with 100,000+ candidates.

Author: ResumeAI
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Dict, Generator, List

logger = logging.getLogger(__name__)


# ==========================================================
# File Validation
# ==========================================================

def validate_file(file_path: str) -> Path:
    """
    Validate that the input file exists.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return path


# ==========================================================
# Stream JSONL.GZ
# ==========================================================

def stream_candidates(file_path: str):

    path = validate_file(file_path)

    logger.info("Opening dataset...")

    # Detect whether file is actually gzipped
    with open(path, "rb") as test:

        magic = test.read(2)

    if magic == b"\x1f\x8b":
        file = gzip.open(path, "rt", encoding="utf-8")
    else:
        file = open(path, "r", encoding="utf-8")

    with file as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)

            except json.JSONDecodeError:
                logger.warning(
                    f"Invalid JSON at line {line_number}"
                )

# ==========================================================
# Load Small JSON
# ==========================================================

def load_sample_json(file_path: str) -> List[Dict]:
    """
    Loads sample_candidates.json.

    Intended only for testing.
    """

    path = validate_file(file_path)

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)


# ==========================================================
# Candidate Counter
# ==========================================================

def count_candidates(file_path: str) -> int:
    """
    Count candidates without loading them into memory.
    """

    count = 0

    for _ in stream_candidates(file_path):

        count += 1

    return count


# ==========================================================
# Batch Generator
# ==========================================================

def batch_candidates(
    candidate_stream,
    batch_size: int = 128
):
    """
    Yield batches of candidates.

    Useful for embedding generation.
    """

    batch = []

    for candidate in candidate_stream:

        batch.append(candidate)

        if len(batch) >= batch_size:

            yield batch

            batch = []

    if batch:

        yield batch


# ==========================================================
# Dataset Statistics
# ==========================================================

def dataset_statistics(file_path: str):

    logger.info("Scanning dataset...")

    total = 0

    countries = set()

    industries = set()

    for candidate in stream_candidates(file_path):

        total += 1

        profile = candidate.get("profile", {})

        country = profile.get("country")

        industry = profile.get("current_industry")

        if country:
            countries.add(country)

        if industry:
            industries.add(industry)

    print()

    print("=" * 60)

    print("Dataset Statistics")

    print("=" * 60)

    print(f"Candidates : {total}")

    print(f"Countries  : {len(countries)}")

    print(f"Industries : {len(industries)}")

    print("=" * 60)


# ==========================================================
# Preview
# ==========================================================

def preview_dataset(file_path: str, limit: int = 3):

    print()

    print("=" * 60)

    print("Dataset Preview")

    print("=" * 60)

    for i, candidate in enumerate(stream_candidates(file_path), start=1):

        profile = candidate.get("profile", {})

        print()

        print("Candidate :", candidate.get("candidate_id"))

        print("Title     :", profile.get("current_title"))

        print("Experience:", profile.get("years_of_experience"))

        print("Location  :", profile.get("country"))

        if i >= limit:
            break


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    DATASET = "data/candidates.jsonl"

    print()

    preview_dataset(DATASET)

    print()

    print("Counting candidates...")

    total = count_candidates(DATASET)

    print(f"Total Candidates : {total}")

    print()

    dataset_statistics(DATASET)

    print()

    print("Testing batches...")

    stream = stream_candidates(DATASET)

    first_batch = next(batch_candidates(stream, batch_size=5))

    print(f"Batch Size : {len(first_batch)}")