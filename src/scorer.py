from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Set

from preprocess import ProcessedCandidate
from embeddings import EmbeddingEngine


# ==========================================================
# SCORE WEIGHTS
# ==========================================================

SEMANTIC_WEIGHT = 25.0

CAREER_WEIGHT = 25.0

SKILL_WEIGHT = 15.0

TITLE_WEIGHT = 10.0

EXPERIENCE_WEIGHT = 10.0

BEHAVIOR_WEIGHT = 10.0

ASSESSMENT_WEIGHT = 5.0


# ==========================================================
# TITLE RELEVANCE
# ==========================================================

STRONG_TITLES = {

    "machine learning engineer",

    "ml engineer",

    "ai engineer",

    "applied scientist",

    "research scientist",

    "research engineer",

    "search engineer",

    "recommendation engineer",

    "ranking engineer",

    "nlp engineer",

    "computer vision engineer",

    "deep learning engineer",

    "backend engineer",

    "software engineer",

    "data engineer",

    "platform engineer",

    "data scientist",

    "staff engineer",

    "senior software engineer",

    "senior backend engineer",

}


GOOD_TITLES = {

    "analytics engineer",

    "python developer",

    "software developer",

    "full stack engineer",

    "cloud engineer",

    "devops engineer",

    "site reliability engineer",

    "big data engineer",

}


BAD_TITLES = {

    "marketing manager",

    "sales manager",

    "operations manager",

    "customer support",

    "hr",

    "human resources",

    "recruiter",

    "accountant",

    "finance manager",

    "business analyst",

    "civil engineer",

    "mechanical engineer",

    "electrical engineer",

    "architect",

    "teacher",

}


# ==========================================================
# POSITIVE CAREER SIGNALS
# ==========================================================

CAREER_KEYWORDS = {

    "recommendation",

    "recommendation system",

    "retrieval",

    "ranking",

    "semantic search",

    "search engine",

    "vector database",

    "vector db",

    "embedding",

    "embeddings",

    "ann",

    "faiss",

    "milvus",

    "pinecone",

    "weaviate",

    "qdrant",

    "chromadb",

    "rag",

    "llm",

    "transformer",

    "bert",

    "sentence transformer",

    "pytorch",

    "tensorflow",

    "huggingface",

    "feature engineering",

    "feature store",

    "airflow",

    "spark",

    "kafka",

    "real time",

    "real-time",

    "streaming",

    "distributed systems",

    "ml pipeline",

    "production ml",

    "offline evaluation",

    "ab testing",

    "a/b testing",

    "inference",

    "serving",

    "model deployment",

    "semantic retrieval",

    "hybrid search",

}


# ==========================================================
# NEGATIVE CAREER SIGNALS
# ==========================================================

NEGATIVE_KEYWORDS = {

    "marketing",

    "sales",

    "civil",

    "construction",

    "mechanical",

    "electrical",

    "architecture",

    "recruitment",

    "customer support",

    "call center",

    "bank teller",

    "insurance",

    "accounting",

    "taxation",

    "auditing",

}


# ==========================================================
# HELPER
# ==========================================================

def normalize(text: str) -> str:

    text = text.lower()

    text = text.replace("-", " ")

    text = re.sub(r"[^a-z0-9 ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


@dataclass
class CandidateScore:

    candidate: ProcessedCandidate

    semantic: float

    career: float

    skills: float

    title: float

    experience: float

    behavior: float

    assessment: float

    bonus: float

    final_score: float

    # ==========================================================
# JD PARSING
# ==========================================================

SKILL_PATTERNS = {

    "python",
    "sql",
    "spark",
    "pyspark",
    "airflow",
    "kafka",
    "hadoop",

    "pytorch",
    "tensorflow",
    "keras",
    "scikit learn",
    "xgboost",

    "transformers",
    "llm",
    "rag",
    "embeddings",
    "embedding",

    "semantic search",
    "retrieval",
    "ranking",
    "recommendation",
    "recommendation system",

    "vector database",
    "vector db",
    "faiss",
    "milvus",
    "pinecone",
    "weaviate",
    "qdrant",
    "chromadb",

    "aws",
    "azure",
    "gcp",

    "docker",
    "kubernetes",

    "feature engineering",
    "feature store",

    "mlops",
    "model deployment",
    "serving",

    "distributed systems",
    "real time",
    "real-time",

    "nlp",
    "computer vision",

}


TITLE_PATTERNS = {

    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "applied scientist",
    "research scientist",
    "research engineer",
    "backend engineer",
    "software engineer",
    "data engineer",
    "data scientist",
    "search engineer",
    "recommendation engineer",
    "ranking engineer",

}


POSITIVE_PATTERNS = {

    "production",
    "real time",
    "real-time",
    "retrieval",
    "ranking",
    "recommendation",
    "semantic",
    "embedding",
    "vector",
    "feature store",
    "feature engineering",
    "pipeline",
    "spark",
    "kafka",
    "airflow",
    "evaluation",
    "ab testing",
    "a/b testing",
    "distributed",
    "serving",
    "inference",
    "llm",
    "rag",

}


NEGATIVE_PATTERNS = {

    "marketing",
    "sales",
    "finance",
    "hr",
    "civil",
    "construction",
    "mechanical",
    "customer support",
    "operations manager",

}


# ==========================================================
# JD HELPERS
# ==========================================================

def extract_required_skills(
    jd_text: str,
) -> Set[str]:

    jd = normalize(jd_text)

    found = set()

    for skill in SKILL_PATTERNS:

        if skill in jd:

            found.add(skill)

    return found


def extract_required_titles(
    jd_text: str,
) -> Set[str]:

    jd = normalize(jd_text)

    found = set()

    for title in TITLE_PATTERNS:

        if title in jd:

            found.add(title)

    return found


def extract_positive_keywords(
    jd_text: str,
) -> Set[str]:

    jd = normalize(jd_text)

    found = set()

    for word in POSITIVE_PATTERNS:

        if word in jd:

            found.add(word)

    return found


def extract_negative_keywords(
    jd_text: str,
) -> Set[str]:

    jd = normalize(jd_text)

    found = set()

    for word in NEGATIVE_PATTERNS:

        if word in jd:

            found.add(word)

    return found


def extract_required_experience(
    jd_text: str,
) -> float:

    jd = normalize(jd_text)

    patterns = [

        r"(\d+)\+?\s+years",

        r"minimum\s+(\d+)",

        r"at least\s+(\d+)",

        r"(\d+)\s+years of experience",

    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            jd,
        )

        if match:

            return float(match.group(1))

    return 0.0


# ==========================================================
# PARSE JD
# ==========================================================

@dataclass
class JDRequirements:

    skills: Set[str]

    titles: Set[str]

    positive_keywords: Set[str]

    negative_keywords: Set[str]

    required_experience: float


def parse_job_description(
    jd_text: str,
) -> JDRequirements:

    return JDRequirements(

        skills=extract_required_skills(
            jd_text,
        ),

        titles=extract_required_titles(
            jd_text,
        ),

        positive_keywords=extract_positive_keywords(
            jd_text,
        ),

        negative_keywords=extract_negative_keywords(
            jd_text,
        ),

        required_experience=extract_required_experience(
            jd_text,
        ),

    )

# ==========================================================
# TITLE SCORE
# ==========================================================

def compute_title_score(
    candidate: ProcessedCandidate,
    jd: JDRequirements,
) -> float:

    title = normalize(candidate.current_title)

    if title in STRONG_TITLES:
        return TITLE_WEIGHT

    if title in GOOD_TITLES:
        return TITLE_WEIGHT * 0.80

    if title in BAD_TITLES:
        return 0.0

    for strong in STRONG_TITLES:

        if strong in title:

            return TITLE_WEIGHT * 0.90

    for good in GOOD_TITLES:

        if good in title:

            return TITLE_WEIGHT * 0.70

    for bad in BAD_TITLES:

        if bad in title:

            return 0.0

    return TITLE_WEIGHT * 0.30


# ==========================================================
# CAREER SCORE
# ==========================================================

def compute_career_score(
    candidate: ProcessedCandidate,
    jd: JDRequirements,
) -> float:

    text = " ".join([

        candidate.summary,

        candidate.headline,

        candidate.embedding_text,

        " ".join(

            entry.description
            for entry in candidate.career_history

        ),

    ])

    text = normalize(text)

    score = 0.0

    positive_hits = 0

    negative_hits = 0

    for keyword in CAREER_KEYWORDS:

        if keyword in text:

            positive_hits += 1

            score += 1.50

    for keyword in jd.positive_keywords:

        if keyword in text:

            score += 2.00

    for keyword in NEGATIVE_KEYWORDS:

        if keyword in text:

            negative_hits += 1

            score -= 1.00

    for keyword in jd.negative_keywords:

        if keyword in text:

            score -= 2.00

    if positive_hits >= 10:

        score += 3.0

    elif positive_hits >= 6:

        score += 2.0

    elif positive_hits >= 3:

        score += 1.0

    score -= negative_hits * 0.50

    score = max(score, 0.0)

    score = min(score, CAREER_WEIGHT)

    return round(score, 2)


# ==========================================================
# SKILL SCORE
# ==========================================================

def compute_skill_score(
    candidate: ProcessedCandidate,
    jd: JDRequirements,
) -> float:

    if len(jd.skills) == 0:

        return SKILL_WEIGHT * 0.50

    candidate_skills = {

        normalize(skill)

        for skill in candidate.normalized_skill_set

    }

    score = 0.0

    for required in jd.skills:

        if required in candidate_skills:

            score += 1.0

            continue

        matched = False

        for skill in candidate_skills:

            if required in skill:

                score += 0.80

                matched = True

                break

            if skill in required:

                score += 0.80

                matched = True

                break

            overlap = len(

                set(required.split())

                &

                set(skill.split())

            )

            if overlap > 0:

                score += 0.60

                matched = True

                break

        if matched:

            continue

    percentage = score / len(jd.skills)

    percentage = min(percentage, 1.0)

    return round(

        percentage * SKILL_WEIGHT,

        2,

    )


# ==========================================================
# EXPERIENCE SCORE
# ==========================================================

def compute_experience_score(
    candidate: ProcessedCandidate,
    jd: JDRequirements,
) -> float:

    years = candidate.years_experience

    required = jd.required_experience

    if required <= 0:

        required = 5.0

    ratio = years / required

    ratio = min(ratio, 1.50)

    base = min(

        ratio,

        1.0,

    )

    score = EXPERIENCE_WEIGHT * base

    career = compute_career_score(

        candidate,

        jd,

    )

    if career < 5:

        score *= 0.40

    elif career < 10:

        score *= 0.70

    elif career > 20:

        score *= 1.10

    score = min(

        score,

        EXPERIENCE_WEIGHT,

    )

    return round(

        score,

        2,

    )

# ==========================================================
# SEMANTIC SCORE
# ==========================================================

def compute_semantic_score(
    semantic_similarity: float,
) -> float:

    semantic_similarity = max(
        semantic_similarity,
        0.0,
    )

    semantic_similarity = min(
        semantic_similarity,
        1.0,
    )

    return round(
        semantic_similarity * SEMANTIC_WEIGHT,
        2,
    )


# ==========================================================
# BEHAVIOR SCORE
# ==========================================================

def compute_behavior_score(
    candidate: ProcessedCandidate,
) -> float:

    behavior = candidate.behavior

    score = 0.0

    if behavior.get("open_to_work", False):
        score += 2.00

    if behavior.get("verified_email", False):
        score += 0.50

    if behavior.get("verified_phone", False):
        score += 0.50

    github = behavior.get(
        "github_score",
        -1,
    )

    if github > 0:

        score += min(
            github / 20,
            2.0,
        )

    response = behavior.get(
        "response_rate",
        0,
    )

    score += response * 2.0

    interview = behavior.get(
        "interview_completion_rate",
        0,
    )

    score += interview * 2.0

    notice = behavior.get(
        "notice_period_days",
        90,
    )

    if notice <= 30:
        score += 1.5

    elif notice <= 60:
        score += 1.0

    elif notice <= 90:
        score += 0.5

    recent = behavior.get(
        "days_since_last_active",
        365,
    )

    if recent <= 7:
        score += 1.5

    elif recent <= 30:
        score += 1.2

    elif recent <= 90:
        score += 0.8

    elif recent <= 180:
        score += 0.4

    return round(
        min(
            score,
            BEHAVIOR_WEIGHT,
        ),
        2,
    )


# ==========================================================
# ASSESSMENT SCORE
# ==========================================================

def compute_assessment_score(
    candidate: ProcessedCandidate,
    jd: JDRequirements,
) -> float:

    scores = candidate.assessment_scores

    if not scores:

        return 0.0

    total = 0.0

    count = 0

    for skill, value in scores.items():

        total += value

        count += 1

    if count == 0:

        return 0.0

    average = total / count

    average = max(
        average,
        0,
    )

    average = min(
        average,
        100,
    )

    return round(
        average / 100 * ASSESSMENT_WEIGHT,
        2,
    )


# ==========================================================
# BONUS / PENALTY
# ==========================================================

def compute_bonus(
    candidate: ProcessedCandidate,
    semantic_score: float,
    career_score: float,
    title_score: float,
) -> float:

    bonus = 0.0

    title = normalize(
        candidate.current_title,
    )

    text = normalize(
        candidate.embedding_text,
    )

    # Excellent semantic + career

    if semantic_score >= 22:

        if career_score >= 20:

            bonus += 3.0

    # Strong engineering title

    if title in STRONG_TITLES:

        bonus += 1.0

    # Production ML keywords

    production_keywords = [

        "retrieval",

        "ranking",

        "recommendation",

        "embedding",

        "production ml",

        "feature store",

        "semantic search",

        "vector",

    ]

    hits = 0

    for keyword in production_keywords:

        if keyword in text:

            hits += 1

    if hits >= 5:

        bonus += 2.0

    elif hits >= 3:

        bonus += 1.0

    # Penalize obviously irrelevant titles

    if title in BAD_TITLES:

        bonus -= 5.0

    elif career_score < 5:

        bonus -= 3.0

    return round(
        bonus,
        2,
    )

# ==========================================================
# SCORE SINGLE CANDIDATE
# ==========================================================

def score_candidate(
    candidate: ProcessedCandidate,
    jd_text: str,
    semantic_similarity: float,
) -> CandidateScore:

    jd = parse_job_description(
        jd_text,
    )

    semantic = compute_semantic_score(
        semantic_similarity,
    )

    career = compute_career_score(
        candidate,
        jd,
    )

    skills = compute_skill_score(
        candidate,
        jd,
    )

    title = compute_title_score(
        candidate,
        jd,
    )

    experience = compute_experience_score(
        candidate,
        jd,
    )

    behavior = compute_behavior_score(
        candidate,
    )

    assessment = compute_assessment_score(
        candidate,
        jd,
    )

    bonus = compute_bonus(
        candidate,
        semantic,
        career,
        title,
    )

    final_score = (
        semantic
        + career
        + skills
        + title
        + experience
        + behavior
        + assessment
        + bonus
    )

    final_score = max(
        0.0,
        min(
            100.0,
            final_score,
        ),
    )

    return CandidateScore(

        candidate=candidate,

        semantic=round(
            semantic,
            2,
        ),

        career=round(
            career,
            2,
        ),

        skills=round(
            skills,
            2,
        ),

        title=round(
            title,
            2,
        ),

        experience=round(
            experience,
            2,
        ),

        behavior=round(
            behavior,
            2,
        ),

        assessment=round(
            assessment,
            2,
        ),

        bonus=round(
            bonus,
            2,
        ),

        final_score=round(
            final_score,
            2,
        ),

    )


# ==========================================================
# SCORE ALL CANDIDATES
# ==========================================================

def score_candidates(
    candidates,
    semantic_scores,
    jd_text,
):

    ranked = []

    for candidate, similarity in zip(
        candidates,
        semantic_scores,
    ):

        ranked.append(

            score_candidate(
                candidate,
                jd_text,
                similarity,
            )

        )

    ranked.sort(
        key=lambda x: x.final_score,
        reverse=True,
    )

    return ranked


# ==========================================================
# PRETTY PRINT
# ==========================================================

def print_top_candidates(
    ranked,
    top_n=10,
):

    print()

    print("=" * 125)

    print(
        f"{'Rank':<5}"
        f"{'Candidate ID':<18}"
        f"{'Final':<10}"
        f"{'Career':<10}"
        f"{'Semantic':<10}"
        f"{'Skills':<10}"
        f"{'Title':<10}"
        f"{'Exp':<8}"
        f"{'Behavior':<10}"
        f"{'Assess':<10}"
        f"{'Bonus':<8}"
    )

    print("=" * 125)

    for rank, result in enumerate(
        ranked[:top_n],
        start=1,
    ):

        print(

            f"{rank:<5}"

            f"{result.candidate.candidate_id:<18}"

            f"{result.final_score:<10.2f}"

            f"{result.career:<10.2f}"

            f"{result.semantic:<10.2f}"

            f"{result.skills:<10.2f}"

            f"{result.title:<10.2f}"

            f"{result.experience:<8.2f}"

            f"{result.behavior:<10.2f}"

            f"{result.assessment:<10.2f}"

            f"{result.bonus:<8.2f}"

        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()

    print(
        "scorer.py loaded successfully."
    )

    print(
        "This module is intended to be used by main.py"
    )