"""
scorer.py

Resume scoring engine.

Computes:

1. Semantic similarity
2. Skill matching
3. Experience score
4. Behaviour score
5. Assessment score

Produces one final score out of 100.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Set

from preprocess import ProcessedCandidate


# ==========================================================
# Final Result
# ==========================================================

@dataclass
class CandidateScore:

    candidate: ProcessedCandidate

    semantic_score: float

    skill_score: float

    experience_score: float

    behavior_score: float

    assessment_score: float

    final_score: float


# ==========================================================
# Skill Synonyms
# ==========================================================

SKILL_SYNONYMS = {

    "ml": "machine learning",
    "machine-learning": "machine learning",

    "ai": "artificial intelligence",

    "nlp": "natural language processing",

    "llm": "large language models",

    "llms": "large language models",

    "gcp": "google cloud",

    "aws": "amazon web services",

    "azure cloud": "azure",

    "js": "javascript",

    "ts": "typescript",

    "tf": "tensorflow",

    "cv": "computer vision",

    "spark streaming": "spark",

    "pyspark": "spark",

    "apache spark": "spark",

    "postgres": "postgresql",

    "postgresql": "postgresql",

    "mysql": "sql",

    "mssql": "sql",

    "sqlite": "sql",

    "nosql": "mongodb",

    "mongo": "mongodb",

    "docker containers": "docker",

    "k8s": "kubernetes",

}



# ==========================================================
# Normalize text
# ==========================================================

def normalize(text: str) -> str:

    text = text.lower()

    text = text.replace("/", " ")

    text = text.replace("-", " ")

    text = text.replace("_", " ")

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return SKILL_SYNONYMS.get(text, text)


# ==========================================================
# Extract JD Skills
# ==========================================================

# ==========================================================
# Known Technical Skills
# ==========================================================

KNOWN_SKILLS = {

    # Programming
    "python","java","c","c++","c#","javascript","typescript","go","rust",
    "php","r","scala","kotlin","swift","matlab",

    # Data
    "sql","mysql","postgresql","oracle","sqlite","mongodb","redis",
    "snowflake","databricks","hadoop","spark","pyspark","airflow",
    "kafka","dbt","apache beam","etl","data warehouse",

    # AI / ML
    "machine learning","deep learning","artificial intelligence",
    "natural language processing","computer vision",
    "tensorflow","keras","pytorch","huggingface",
    "llm","large language models","lora",
    "langchain","transformers","opencv","scikit-learn",
    "xgboost","lightgbm","catboost",

    # Cloud
    "aws","amazon web services","azure","google cloud","gcp",

    # DevOps
    "docker","kubernetes","jenkins","git","github","linux",

    # Backend
    "flask","django","fastapi","spring","node.js","express",

    # Frontend
    "react","angular","vue","html","css","tailwind","bootstrap",

    # Misc
    "excel","power bi","tableau","photoshop"
}


# ==========================================================
# Extract JD Skills
# ==========================================================

def extract_jd_skills(job_description: str):

    jd = job_description.lower()

    jd = jd.replace("/", " ")

    jd = jd.replace("-", " ")

    found = set()

    for skill in KNOWN_SKILLS:

        if skill in jd:

            found.add(normalize(skill))

    return found

# ==========================================================
# Skill Matching
# ==========================================================

def compute_skill_score(
    candidate: ProcessedCandidate,
    jd_skills: Set[str],
) -> float:
    """
    Compute a skill matching score out of 25.
    Uses exact, partial and word-overlap matching.
    """

    if not jd_skills:
        return 25.0

    candidate_skills = {
        normalize(skill)
        for skill in candidate.normalized_skill_set
    }

    score = 0.0

    for jd_skill in jd_skills:

        matched = False

        # -------------------------------
        # Exact Match
        # -------------------------------
        if jd_skill in candidate_skills:
            score += 1.0
            continue

        # -------------------------------
        # Partial / Semantic Match
        # -------------------------------
        for skill in candidate_skills:

            # Exact (extra safety)
            if skill == jd_skill:
                score += 1.0
                matched = True
                break

            # One contains the other
            if jd_skill in skill or skill in jd_skill:
                score += 0.80
                matched = True
                break

            # Word overlap
            jd_words = set(jd_skill.split())
            skill_words = set(skill.split())

            overlap = len(jd_words & skill_words)

            if overlap > 0:
                score += 0.60
                matched = True
                break

        # No match → add nothing

    percentage = score / len(jd_skills)

    return round(
        percentage * 25,
        2,
    )

# ==========================================================
# Experience
# ==========================================================

def extract_required_experience(
    job_description: str,
) -> float:

    jd = job_description.lower()

    patterns = [

        r"(\d+)\+?\s*years",

        r"(\d+)\s*-\s*(\d+)\s*years",

        r"minimum\s*(\d+)",

        r"at least\s*(\d+)",

    ]

    for pattern in patterns:

        match = re.search(pattern, jd)

        if not match:
            continue

        if len(match.groups()) == 1:

            return float(match.group(1))

        else:

            low = float(match.group(1))

            high = float(match.group(2))

            return (low + high) / 2

    return 0.0


def compute_experience_score(
    candidate: ProcessedCandidate,
    required_years: float,
) -> float:

    if required_years <= 0:
        return 15.0

    ratio = candidate.years_experience / required_years

    ratio = min(ratio, 1.25)

    return round(
        min(15.0, ratio * 15),
        2,
    )

    # ==========================================================
# Behavior Score (0-10)
# ==========================================================

def compute_behavior_score(
    candidate: ProcessedCandidate,
) -> float:

    behavior = candidate.behavior

    score = 0.0

    if behavior.get("open_to_work", False):
        score += 2.0

    days = behavior.get("days_since_last_active", 365)

    if days <= 7:
        score += 2.0
    elif days <= 30:
        score += 1.5
    elif days <= 90:
        score += 1.0

    response_rate = behavior.get(
        "response_rate",
        0.0,
    )

    score += min(
        response_rate * 2.5,
        2.5,
    )

    interview = behavior.get(
        "interview_completion_rate",
        0.0,
    )

    score += min(
        interview * 2.0,
        2.0,
    )

    github = behavior.get(
        "github_score",
        -1,
    )

    if github >= 75:
        score += 1.0

    elif github >= 40:
        score += 0.5

    if behavior.get("verified_email", False):
        score += 0.25

    if behavior.get("verified_phone", False):
        score += 0.25

    return round(
        min(score, 10.0),
        2,
    )


# ==========================================================
# Assessment Score (0-10)
# ==========================================================

def compute_assessment_score(
    candidate: ProcessedCandidate,
    jd_skills: Set[str],
) -> float:

    assessments = candidate.assessment_scores

    if not assessments:
        return 0.0

    total = 0.0
    matched = 0

    for skill, score in assessments.items():

        skill = normalize(skill)

        for jd_skill in jd_skills:

            if (
                skill == jd_skill
                or skill in jd_skill
                or jd_skill in skill
            ):
                total += score
                matched += 1
                break

    if matched == 0:
        return 0.0

    average = total / matched

    # Convert 0–100 assessment average to 0–10 points
    return round(
        average / 10,
        2,
    )

# ==========================================================
# Semantic Score (0-40)
# ==========================================================

def compute_semantic_score(
    semantic_similarity: float,
) -> float:

    semantic_similarity = max(
        0.0,
        min(
            100.0,
            semantic_similarity,
        ),
    )

    return round(
        semantic_similarity * 0.40,
        2,
    )


# ==========================================================
# Final Score
# ==========================================================

def score_candidate(

    candidate: ProcessedCandidate,

    semantic_similarity: float,

    job_description: str,

) -> CandidateScore:

    jd_skills = extract_jd_skills(
        job_description
    )

    required_experience = extract_required_experience(
        job_description
    )

    semantic_score = compute_semantic_score(
        semantic_similarity
    )

    skill_score = compute_skill_score(
        candidate,
        jd_skills,
    )

    experience_score = compute_experience_score(
        candidate,
        required_experience,
    )

    behavior_score = compute_behavior_score(
        candidate
    )

    assessment_score = compute_assessment_score(
        candidate,
        jd_skills,
    )

    final_score = round(

        semantic_score

        + skill_score

        + experience_score

        + behavior_score

        + assessment_score,

        2,

    )

    return CandidateScore(

        candidate=candidate,

        semantic_score=semantic_score,

        skill_score=skill_score,

        experience_score=experience_score,

        behavior_score=behavior_score,

        assessment_score=assessment_score,

        final_score=final_score,

    )


# ==========================================================
# Batch Scoring
# ==========================================================

def score_candidates(

    embedding_results,

    job_description,

):

    scored = []

    for candidate, semantic_score in embedding_results:

        scored.append(

            score_candidate(

                candidate,

                semantic_score,

                job_description,

            )

        )

    return scored


# ==========================================================
# Ranking
# ==========================================================

def rank_candidates(

    scored_candidates,

):

    return sorted(

        scored_candidates,

        key=lambda x: x.final_score,

        reverse=True,

    )
    # ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    import logging

    from data_loader import load_sample_json
    from preprocess import preprocess_candidate
    from embeddings import (
        EmbeddingEngine,
        embedding_stream,
        load_job_description,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    SAMPLE_FILE = "data/sample_candidates.json"
    JD_FILE = "data/job_description.docx"

    print("\nLoading sample candidates...")

    raw_candidates = load_sample_json(SAMPLE_FILE)

    processed_candidates = (
        preprocess_candidate(candidate)
        for candidate in raw_candidates
    )

    job_description = load_job_description(JD_FILE)

    engine = EmbeddingEngine(
        model_name="models/all-MiniLM-L6-v2"
    )

    embedding_results = embedding_stream(
        engine=engine,
        processed_candidate_stream=processed_candidates,
        job_description=job_description,
    )

    ranked = rank_candidates(

        score_candidates(

            embedding_results,

            job_description,

        )

    )

    print("\n")
    print("=" * 110)
    print(
        f"{'Rank':<5}"
        f"{'Candidate ID':<18}"
        f"{'Final':<10}"
        f"{'Semantic':<10}"
        f"{'Skills':<10}"
        f"{'Exp':<8}"
        f"{'Behavior':<10}"
        f"{'Assess':<10}"
    )
    print("=" * 110)

    for rank, result in enumerate(
        ranked[:10],
        start=1,
    ):

        print(

            f"{rank:<5}"

            f"{result.candidate.candidate_id:<18}"

            f"{result.final_score:<10.2f}"

            f"{result.semantic_score:<10.2f}"

            f"{result.skill_score:<10.2f}"

            f"{result.experience_score:<8.2f}"

            f"{result.behavior_score:<10.2f}"

            f"{result.assessment_score:<10.2f}"

        )

    print("\nTop Candidate")
    print("-" * 40)

    top = ranked[0]

    print("ID          :", top.candidate.candidate_id)
    print("Title       :", top.candidate.current_title)
    print("Company     :", top.candidate.current_company)
    print("Experience  :", top.candidate.years_experience)
    print("Country     :", top.candidate.country)
    print("Final Score :", top.final_score)

    print("\nDone.")