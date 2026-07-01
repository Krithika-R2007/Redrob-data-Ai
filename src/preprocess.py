"""
preprocess.py
=============

ResumeAI Hackathon

Purpose:
--------
Convert raw candidate JSON into a structured object used by
the embedding engine and scoring engine.

This module DOES NOT:
    - read files
    - write files
    - generate embeddings
    - rank candidates

It ONLY performs feature engineering.

Author: ResumeAI Team
"""

from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------

DATE_FORMAT = "%Y-%m-%d"

TODAY = datetime.today()

SKILL_ALIASES = {

    "python3": "Python",
    "python": "Python",

    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",

    "js": "JavaScript",

    "node": "Node.js",
    "nodejs": "Node.js",

    "ml": "Machine Learning",

    "ai": "Artificial Intelligence",

    "cv": "Computer Vision",

    "nlp": "Natural Language Processing",

    "llm": "Large Language Models",

    "rag": "Retrieval Augmented Generation",

    "aws": "AWS",

    "gcp": "Google Cloud",

    "azure": "Microsoft Azure",

    "tf": "TensorFlow"
}

IMPORTANT_BEHAVIOR_FIELDS = {

    "open_to_work_flag",

    "last_active_date",

    "recruiter_response_rate",

    "avg_response_time_hours",

    "notice_period_days",

    "github_activity_score",

    "interview_completion_rate",

    "verified_email",

    "verified_phone",

    "willing_to_relocate",

    "skill_assessment_scores"
}

# ----------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------

@dataclass
class Skill:

    name: str

    normalized: str

    proficiency: str

    duration_months: int

    endorsements: int


@dataclass
class CareerEntry:

    company: str

    title: str

    start_date: Optional[str]

    end_date: Optional[str]

    duration_months: int

    is_current: bool

    industry: str

    description: str


@dataclass
class ProcessedCandidate:

    candidate_id: str

    current_title: str

    years_experience: float

    headline: str

    summary: str

    current_company: str

    industry: str

    location: str

    country: str

    skills: List[Skill] = field(default_factory=list)

    normalized_skill_set: Set[str] = field(default_factory=set)

    career_history: List[CareerEntry] = field(default_factory=list)

    education_text: str = ""

    certification_text: str = ""

    languages: List[str] = field(default_factory=list)

    behavior: Dict[str, Any] = field(default_factory=dict)

    assessment_scores: Dict[str, float] = field(default_factory=dict)

    embedding_text: str = ""


# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """
    Remove repeated whitespace.

    Keeps punctuation because embedding models
    understand natural language.
    """

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_skill(skill: str) -> str:
    """
    Normalize skill names.

    Examples

    python3

    →

    Python

    NLP

    →

    Natural Language Processing
    """

    if not skill:
        return ""

    key = skill.strip().lower()

    normalized = SKILL_ALIASES.get(key, skill.strip())
    
    return normalized.strip()


def parse_date(date_string: Optional[str]) -> Optional[datetime]:

    if not date_string:
        return None

    try:
        return datetime.strptime(date_string, DATE_FORMAT)

    except Exception:

        return None


def days_since(date_string: Optional[str]) -> int:

    dt = parse_date(date_string)

    if dt is None:
        return 9999

    return (TODAY - dt).days


def safe_float(value: Any) -> float:

    try:

        return float(value)

    except Exception:

        return 0.0


def safe_int(value: Any) -> int:

    try:

        return int(value)

    except Exception:

        return 0


def unique_preserve_order(items: List[str]) -> List[str]:

    seen = set()

    result = []

    for item in items:

        if item not in seen:

            seen.add(item)

            result.append(item)

    return result


# ----------------------------------------------------------
# Profile Extraction
# ----------------------------------------------------------

def extract_profile(candidate: Dict[str, Any]) -> Dict[str, Any]:

    profile = candidate.get("profile", {})

    return {

        "candidate_id": candidate.get("candidate_id", ""),

        "headline": clean_text(profile.get("headline", "")),

        "summary": clean_text(profile.get("summary", "")),

        "current_title": clean_text(profile.get("current_title", "")),

        "years_experience": safe_float(
            profile.get("years_of_experience", 0)
        ),

        "current_company": clean_text(
            profile.get("current_company", "")
        ),

        "industry": clean_text(
            profile.get("current_industry", "")
        ),

        "location": clean_text(
            profile.get("location", "")
        ),

        "country": clean_text(
            profile.get("country", "")
        )

    }
    # ----------------------------------------------------------
# Skill Extraction
# ----------------------------------------------------------

def extract_skills(candidate: Dict[str, Any]) -> tuple[List[Skill], Set[str]]:

    raw_skills = candidate.get("skills", [])

    skill_objects: List[Skill] = []
    normalized_set: Set[str] = set()

    for skill in raw_skills:

        original = clean_text(skill.get("name", ""))

        if not original:
            continue

        normalized = normalize_skill(original)

        normalized_set.add( normalized.lower().replace("-", " ") )

        skill_objects.append(
            Skill(
                name=original,
                normalized=normalized,
                proficiency=clean_text(skill.get("proficiency", "unknown")),
                duration_months=safe_int(skill.get("duration_months", 0)),
                endorsements=safe_int(skill.get("endorsements", 0))
            )
        )

    return skill_objects, normalized_set


# ----------------------------------------------------------
# Career History
# ----------------------------------------------------------

def extract_career_history(candidate: Dict[str, Any]) -> List[CareerEntry]:

    history = candidate.get("career_history", [])

    entries: List[CareerEntry] = []

    history = sorted(
        history,
        key=lambda x: x.get("start_date", ""),
        reverse=True
    )

    for job in history:

        entries.append(

            CareerEntry(

                company=clean_text(job.get("company", "")),

                title=clean_text(job.get("title", "")),

                start_date=job.get("start_date"),

                end_date=job.get("end_date"),

                duration_months=safe_int(
                    job.get("duration_months", 0)
                ),

                is_current=bool(job.get("is_current", False)),

                industry=clean_text(
                    job.get("industry", "")
                ),

                description=clean_text(
                    job.get("description", "")
                )

            )

        )

    return entries


# ----------------------------------------------------------
# Education
# ----------------------------------------------------------

def extract_education(candidate: Dict[str, Any]) -> str:

    education = candidate.get("education", [])

    parts = []

    for edu in education:

        degree = clean_text(edu.get("degree", ""))

        field = clean_text(edu.get("field_of_study", ""))

        institution = clean_text(edu.get("institution", ""))

        tier = clean_text(edu.get("tier", ""))

        grade = clean_text(edu.get("grade", ""))

        sentence = (
            f"{degree} in {field} "
            f"from {institution}. "
            f"{tier}. "
            f"{grade}."
        )

        parts.append(sentence)

    return "\n".join(parts)


# ----------------------------------------------------------
# Certifications
# ----------------------------------------------------------

def extract_certifications(candidate: Dict[str, Any]) -> str:

    certs = candidate.get("certifications", [])

    if not certs:
        return ""

    lines = []

    for cert in certs:

        name = clean_text(cert.get("name", ""))

        issuer = clean_text(cert.get("issuer", ""))

        year = clean_text(str(cert.get("year", "")))

        line = f"{name} ({issuer}, {year})"

        lines.append(line)

    return "\n".join(lines)


# ----------------------------------------------------------
# Languages
# ----------------------------------------------------------

def extract_languages(candidate: Dict[str, Any]) -> List[str]:

    langs = []

    for item in candidate.get("languages", []):

        language = clean_text(item.get("language", ""))

        proficiency = clean_text(item.get("proficiency", ""))

        if language:

            langs.append(
                f"{language} ({proficiency})"
            )

    return langs


# ----------------------------------------------------------
# Behavioral Signals
# ----------------------------------------------------------

def extract_behavior(candidate: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:

    signals = candidate.get("redrob_signals", {})

    behavior = {

        "open_to_work": bool(
            signals.get("open_to_work_flag", False)
        ),

        "days_since_last_active": days_since(
            signals.get("last_active_date")
        ),

        "response_rate": safe_float(
            signals.get("recruiter_response_rate", 0)
        ),

        "response_time_hours": safe_float(
            signals.get("avg_response_time_hours", 999)
        ),

        "notice_period_days": safe_int(
            signals.get("notice_period_days", 180)
        ),

        "github_score": safe_float(
            signals.get("github_activity_score", -1)
        ),

        "interview_completion_rate": safe_float(
            signals.get("interview_completion_rate", 0)
        ),

        "verified_email": bool(
            signals.get("verified_email", False)
        ),

        "verified_phone": bool(
            signals.get("verified_phone", False)
        ),

        "willing_to_relocate": bool(
            signals.get("willing_to_relocate", False)
        )

    }

    assessment_scores = {}

    for skill, score in signals.get("skill_assessment_scores", {}).items():

        assessment_scores[normalize_skill(skill)] = safe_float(score)

    return behavior, assessment_scores
    # ----------------------------------------------------------
# Embedding Text Builder
# ----------------------------------------------------------

def build_embedding_text(
    profile: Dict[str, Any],
    skills: List[Skill],
    career: List[CareerEntry],
    education: str,
    certifications: str,
    languages: List[str]
) -> str:

    lines = []

    # ---------------- Profile ----------------

    if profile["headline"]:
        lines.append(profile["headline"])

    if profile["summary"]:
        lines.append(profile["summary"])

    lines.append(
        f"{profile['current_title']} with "
        f"{profile['years_experience']} years of experience."
    )

    # ---------------- Skills ----------------

    if skills:

        lines.append("\nTechnical Skills:")

        for skill in skills:

            line = (
                f"{skill.name} "
                f"({skill.proficiency}, "
                f"{skill.duration_months} months)"
            )

            lines.append(line)

    # ---------------- Career ----------------

    if career:

        lines.append("\nCareer History:")

        for job in career:

            end = "Present" if job.is_current else (job.end_date or "")

            lines.append(
                f"{job.title} at {job.company} "
                f"({job.start_date} - {end})"
            )

            if job.description:
                lines.append(job.description)

    # ---------------- Education ----------------

    if education:

        lines.append("\nEducation:")

        lines.append(education)

    # ---------------- Certifications ----------------

    if certifications:

        lines.append("\nCertifications:")

        lines.append(certifications)

    # ---------------- Languages ----------------

    if languages:

        lines.append("\nLanguages:")

        lines.extend(languages)

    return "\n".join(lines)


# ----------------------------------------------------------
# Main Preprocessing Function
# ----------------------------------------------------------

def preprocess_candidate(candidate: Dict[str, Any]) -> ProcessedCandidate:

    profile = extract_profile(candidate)

    skills, normalized_skill_set = extract_skills(candidate)

    career = extract_career_history(candidate)

    education = extract_education(candidate)

    certifications = extract_certifications(candidate)

    languages = extract_languages(candidate)

    behavior, assessment_scores = extract_behavior(candidate)

    embedding_text = build_embedding_text(
        profile,
        skills,
        career,
        education,
        certifications,
        languages
    )

    return ProcessedCandidate(

        candidate_id=profile["candidate_id"],

        current_title=profile["current_title"],

        years_experience=profile["years_experience"],

        headline=profile["headline"],

        summary=profile["summary"],

        current_company=profile["current_company"],

        industry=profile["industry"],

        location=profile["location"],

        country=profile["country"],

        skills=skills,

        normalized_skill_set=normalized_skill_set,

        career_history=career,

        education_text=education,

        certification_text=certifications,

        languages=languages,

        behavior=behavior,

        assessment_scores=assessment_scores,

        embedding_text=embedding_text

    )


# ----------------------------------------------------------
# Batch Processing
# ----------------------------------------------------------

def preprocess_candidates(candidates):

    for candidate in candidates:

        yield preprocess_candidate(candidate)


# ----------------------------------------------------------
# Test Block
# ----------------------------------------------------------

if __name__ == "__main__":

    import json

    logger.info("Testing preprocess.py")

    with open(
        "data/sample_candidates.json",
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(f)

    for candidate in candidates[:3]:

        processed = preprocess_candidate(candidate)

        print("=" * 80)

        print("Candidate :", processed.candidate_id)

        print("Title     :", processed.current_title)

        print("Experience:", processed.years_experience)

        print("Skills    :", len(processed.skills))

        print("Skill Set :", sorted(processed.normalized_skill_set)[:10])

        print("Behavior  :", processed.behavior)

        print("Assessment:", processed.assessment_scores)

        print()

        print(processed.embedding_text[:700])

        print()