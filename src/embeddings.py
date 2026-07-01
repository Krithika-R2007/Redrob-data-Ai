"""
embeddings.py

Semantic embedding engine for ResumeAI.

Uses:
- Sentence Transformers
- Batch processing
- Streaming-friendly architecture
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pathlib import Path
from docx import Document


logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Wrapper around SentenceTransformer.
    Loads model only once.
    """

    @staticmethod
    def normalize_similarity(score: float) -> float:
        """
        Converts cosine similarity (0-1)
        into a cleaner ATS score (0-100).

        We cap values outside range for safety.
        """

        score = max(0.0, min(1.0, float(score)))

        return round(score * 100, 2)

    def __init__(
        self,
        model_name: str = "models/all-MiniLM-L6-v2",
        batch_size: int = 128,
    ):

        self.batch_size = batch_size

        logger.info("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        logger.info("Model loaded successfully.")

    # --------------------------------------------------------

    def encode(self, text: str):

        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # --------------------------------------------------------

    def encode_batch(
        self,
        texts: List[str],
    ):

        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # --------------------------------------------------------

    @staticmethod
    def similarity(
        jd_embedding,
        candidate_embeddings,
    ):

        scores = cosine_similarity(
            candidate_embeddings,
            jd_embedding.reshape(1, -1),
        )

        return scores.flatten()


# ============================================================
# Job Description
# ============================================================

def load_job_description(path: str) -> str:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".docx":

        document = Document(path)

        paragraphs = [
            p.text.strip()
            for p in document.paragraphs
            if p.text.strip()
        ]

        return "\n".join(paragraphs)

    else:

        return path.read_text(
            encoding="utf-8"
        )

# ============================================================
# Batch helper
# ============================================================

def create_batches(
    items,
    batch_size: int,
):

    batch = []

    for item in items:

        batch.append(item)

        if len(batch) >= batch_size:

            yield batch

            batch = []

    if batch:

        yield batch

# ============================================================
# Similarity Scoring
# ============================================================

def score_candidate_batch(
    engine: EmbeddingEngine,
    jd_embedding,
    processed_candidates,
):
    """
    Computes semantic similarity for one batch of processed candidates.
    Returns:
        (candidate, similarity_score)
    """

    if not processed_candidates:
        return []

    texts = [
        candidate.embedding_text
        for candidate in processed_candidates
    ]

    candidate_embeddings = engine.encode_batch(texts)

    scores = engine.similarity(
        jd_embedding,
        candidate_embeddings,
    )

    scores = [
        engine.normalize_similarity(s)
        for s in scores
    ]

    return list(zip(processed_candidates, scores))


# ============================================================
# Streaming Pipeline
# ============================================================

def embedding_stream(
    engine: EmbeddingEngine,
    processed_candidate_stream,
    job_description: str,
):
    """
    Streaming embedding generator.

    Input:
        processed_candidate_stream

    Output:
        (ProcessedCandidate, semantic_score)
    """

    jd_embedding = engine.encode(job_description)

    for batch in create_batches(
        processed_candidate_stream,
        engine.batch_size,
    ):

        results = score_candidate_batch(
            engine,
            jd_embedding,
            batch,
        )

        for result in results:
            yield result


# ============================================================
# Test Module
# ============================================================

if __name__ == "__main__":

    import logging

    from data_loader import load_sample_json
    from preprocess import preprocess_candidate

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    SAMPLE_FILE = "data/sample_candidates.json"
    JD_FILE = "data/job_description.docx"

    print("\nLoading sample candidates...")

    raw_candidates = load_sample_json(SAMPLE_FILE)

    processed = (
        preprocess_candidate(candidate)
        for candidate in raw_candidates
    )

    jd = load_job_description(JD_FILE)

    engine = EmbeddingEngine()

    print("\nGenerating semantic similarity...\n")

    results = embedding_stream(
        engine=engine,
        processed_candidate_stream=processed,
        job_description=jd,
    )

    print("=" * 80)
    print(f"{'Candidate ID':<18}{'Semantic Score'}")
    print("=" * 80)

    count = 0

    for candidate, score in results:

        print(
            f"{candidate.candidate_id:<18}"
            f"{score:.4f}"
        )

        count += 1

        if count >= 10:
            break

    print("\nFinished.")