import os
import sys
import argparse
import pandas as pd
import logging

from tqdm import tqdm

from data_loader import stream_candidates
from preprocess import preprocess_candidates
from embeddings import EmbeddingEngine, load_job_description, embedding_stream
from scorer import (
    score_candidate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_recommendation(score: float) -> str:
    """
    Recommendation based on score thresholds calibrated to this dataset.
    Scores typically range 25–55 out of 100 (due to CPU-based cosine similarity).
      >= 48  → Shortlist
      >= 42  → Strong Consider
      >= 35  → Consider
      <  35  → Reject
    """
    if score >= 48:
        return "Shortlist"
    elif score >= 42:
        return "Strong Consider"
    elif score >= 35:
        return "Consider"
    else:
        return "Reject"


def score_all_candidates(
    embedding_results,
    jd_text,
    total_hint=None,
):

    scored = []

    bar = tqdm(
        embedding_results,
        total=total_hint,
        desc="Scoring candidates",
        unit="cand",
        dynamic_ncols=True,
    )

    for candidate, similarity in bar:

        result = score_candidate(
            candidate,
            jd_text,
            similarity,
        )

        scored.append(result)

    return scored

def export_results(ranked_candidates, top_k, output_dir):
    filtered = ranked_candidates[:top_k] if top_k else ranked_candidates

    csv_rows   = []
    excel_rows = []

    for rank, cs in enumerate(filtered, start=1):
        cand = cs.candidate

        reasoning = (
            f"{cand.current_title} with {cand.years_experience:.1f} yrs; "
            f"{len(cand.skills)} AI core skills; "
            f"response rate {cand.behavior.get('response_rate', 0.0):.2f}."
        )

        csv_rows.append({
            "candidate_id": cand.candidate_id,
            "rank":         rank,
            "score":        round(cs.final_score / 100.0, 4),
            "reasoning":    reasoning,
        })

        excel_rows.append({
    "Rank": rank,
    "Candidate ID": cand.candidate_id,
    "Current Title": cand.current_title,
    "Company": cand.current_company,
    "Country": cand.country,

    "Final Score": cs.final_score,

    "Semantic Score": cs.semantic,

    "Career Score": cs.career,

    "Skill Score": cs.skills,

    "Title Score": cs.title,

    "Experience Score": cs.experience,

    "Behavior Score": cs.behavior,

    "Assessment Score": cs.assessment,

    "Bonus": cs.bonus,

    "Recommendation": get_recommendation(cs.final_score),
})

    csv_path = os.path.join(output_dir, "submission.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Exported CSV  → {csv_path}")

    excel_path = os.path.join(output_dir, "report.xlsx")
    try:
        pd.DataFrame(excel_rows).to_excel(excel_path, index=False)
        logger.info(f"Exported Excel → {excel_path}")
    except PermissionError:
        logger.error("Could not write report.xlsx — please close it in Excel and re-run.")


def generate_statistics(ranked_candidates):
    total = len(ranked_candidates)
    if total == 0:
        logger.warning("No candidates to report statistics for.")
        return

    scores      = [c.final_score for c in ranked_candidates]
    highest     = max(scores)
    lowest      = min(scores)
    average     = sum(scores) / total
    top10       = ranked_candidates[:10]
    top10_avg   = sum(c.final_score for c in top10) / len(top10) if top10 else 0
    shortlisted = sum(1 for c in ranked_candidates if get_recommendation(c.final_score) == "Shortlist")
    strong      = sum(1 for c in ranked_candidates if get_recommendation(c.final_score) == "Strong Consider")
    consider    = sum(1 for c in ranked_candidates if get_recommendation(c.final_score) == "Consider")
    rejected    = sum(1 for c in ranked_candidates if get_recommendation(c.final_score) == "Reject")

    print("\n" + "=" * 50)
    print("STATISTICS REPORT")
    print("=" * 50)
    print(f"Total Candidates   : {total}")
    print(f"Highest Score      : {highest:.2f}")
    print(f"Lowest Score       : {lowest:.2f}")
    print(f"Average Score      : {average:.2f}")
    print(f"Top 10 Average     : {top10_avg:.2f}")
    print(f"Shortlisted        : {shortlisted}")
    print(f"Strong Consider    : {strong}")
    print(f"Consider           : {consider}")
    print(f"Rejected           : {rejected}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Candidate Scoring Pipeline")
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Number of top candidates to export (e.g. 10, 20, 50, 100)",
    )
    parser.add_argument(
        "--data", type=str, default="data/sample_candidates.json",
        help="Path to candidates dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--total", type=int, default=None,
        help="Expected total candidates (for progress bar). E.g. 100000 for full dataset.",
    )
    args = parser.parse_args()

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path  = os.path.join(base_dir, args.data) if not os.path.isabs(args.data) else args.data
    jd_path    = os.path.join(base_dir, "data", "job_description.docx")
    models_dir = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
    output_dir = os.path.join(base_dir, "data")

    # ── 1. Load JD ──────────────────────────────────────────
    logger.info("Loading Job Description...")
    jd_text = load_job_description(jd_path)

    # ── 3. Load embedding engine ─────────────────────────────
    logger.info("Loading Embedding Engine...")
    engine = EmbeddingEngine(model_name=models_dir)

    # ── 4. Set up streaming pipeline ─────────────────────────
    logger.info("Setting up candidate stream...")
    if data_path.endswith(".json"):
        from data_loader import load_sample_json
        candidate_stream = load_sample_json(data_path)
        total_hint = args.total
    else:
        candidate_stream = stream_candidates(data_path)
        total_hint = args.total or 100_000   # reasonable default for progress bar

    logger.info("Preprocessing candidates...")
    processed_stream = preprocess_candidates(candidate_stream)

    logger.info("Generating embeddings...")
    embedding_results = embedding_stream(engine, processed_stream, jd_text)

    # ── 5. Score with progress bar ───────────────────────────
    scored_candidates = score_all_candidates(

    embedding_results,

    jd_text,

    total_hint=total_hint,

)

    # ── 6. Rank ──────────────────────────────────────────────
    logger.info("Ranking candidates...")
    ranked_candidates = sorted(
    scored_candidates,
    key=lambda x: x.final_score,
    reverse=True,
)

    # ── 7. Export ────────────────────────────────────────────
    logger.info(f"Exporting Top {args.top_k} results...")
    export_results(ranked_candidates, args.top_k, output_dir)

    # ── 8. Statistics ────────────────────────────────────────
    logger.info("Generating statistics report...")
    generate_statistics(ranked_candidates)

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
