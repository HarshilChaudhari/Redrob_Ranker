import gradio as gr
import json
import csv
import os
import sys
import io
import time
import math
import tempfile
import numpy as np
import pandas as pd

from features import (
    is_honeypot, compute_taxonomy_score, compute_career_trajectory_raw,
    compute_title_fit_raw, compute_experience_fit, compute_location_raw,
    compute_behavioral_raw, compute_behavioral_modifier, generate_reasoning,
    normalize, TIER_WEIGHT,
)
from rank import load_candidates

SEARCH_CORE = {
    "FAISS", "Pinecone", "Weaviate", "Milvus", "Qdrant", "OpenSearch",
    "Learning to Rank", "Vector Search", "Semantic Search", "BM25",
    "Information Retrieval", "NDCG", "MRR", "MAP",
    "Dense Retrieval", "Hybrid Search", "Cross-Encoder", "Re-ranking",
}


def search_depth_count(skills):
    skill_names = {s.get("name", "") for s in skills}
    return len(skill_names & SEARCH_CORE)


def search_depth_modifier(count):
    return min(1.15, 1.0 + 0.05 * count)


def csv_download_path():
    return os.path.join(tempfile.gettempdir(), "submission.csv")


def run_ranking(jsonl_file):
    if jsonl_file is None:
        return None, "Please upload a candidates.jsonl file first.", None, None

    log_buf = io.StringIO()

    def log(msg):
        print(msg, file=log_buf)

    t0 = time.time()
    np.random.seed(42)

    try:
        log(f"Loading candidates from {jsonl_file.name}...")
        candidates = load_candidates(jsonl_file.name)
    except Exception as e:
        return None, f"Error loading file: {e}", None, None

    N = len(candidates)
    log(f"Loaded {N} candidates")

    is_honeypot_arr = np.zeros(N, dtype=bool)

    signal_collector = {
        "search_appearance_30d": [],
        "saved_by_recruiters_30d": [],
        "connection_count": [],
        "endorsements_received": [],
    }

    t1 = time.time()
    for i, c in enumerate(candidates):
        if is_honeypot(c):
            is_honeypot_arr[i] = True
            continue
        sigs = c.get("redrob_signals", {})
        for k in signal_collector:
            signal_collector[k].append(sigs.get(k, 0))

    behavioral_caps = {}
    for k, vals in signal_collector.items():
        if vals:
            behavioral_caps[k] = max(float(np.percentile(vals, 99)), 1.0)
        else:
            behavioral_caps[k] = 1.0

    valid_count = int(N - np.sum(is_honeypot_arr))
    honeypot_count = N - valid_count
    log(f"Pre-scan: {time.time() - t1:.2f}s")
    log(f"Non-honeypots: {valid_count} | Honeypots: {honeypot_count}")
    log("Behavioral caps (P99 from non-honeypots):")
    for k, v in behavioral_caps.items():
        log(f"  {k}: {v:.0f}")

    skill_tax = np.zeros(N, dtype=np.float32)
    title_fit_raw = np.zeros(N, dtype=np.float32)
    trajectory_raw = np.zeros(N, dtype=np.float32)
    experience_raw = np.zeros(N, dtype=np.float32)
    location_raw = np.zeros(N, dtype=np.float32)
    behavioral_raw_arr = np.zeros(N, dtype=np.float32)
    modifier_raw = np.ones(N, dtype=np.float32)
    n_skills_list = np.zeros(N, dtype=np.int32)
    search_counts = np.zeros(N, dtype=np.int32)

    t2 = time.time()
    for i, c in enumerate(candidates):
        if is_honeypot_arr[i]:
            continue
        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {})
        skills = c.get("skills", [])
        career_hist = c.get("career_history", [])

        skill_tax[i] = compute_taxonomy_score(skills)
        title_fit_raw[i] = compute_title_fit_raw(c)
        trajectory_raw[i] = compute_career_trajectory_raw(career_hist)
        experience_raw[i] = compute_experience_fit(profile.get("years_of_experience", 0) or 0)
        location_raw[i] = compute_location_raw(
            profile.get("location", ""),
            profile.get("country", ""),
            profile.get("willing_to_relocate", False),
        )
        behavioral_raw_arr[i] = compute_behavioral_raw(signals, behavioral_caps)
        modifier_raw[i] = compute_behavioral_modifier(signals)
        n_skills_list[i] = sum(1 for s in skills if s.get("name", "") in TIER_WEIGHT)
        search_counts[i] = search_depth_count(skills)

    log(f"Feature extraction: {time.time() - t2:.2f}s")

    t3 = time.time()
    title_norm = np.array([normalize(x, -0.30, 0.40) for x in title_fit_raw])
    traj_norm = np.array([normalize(x, -0.30, 0.30) for x in trajectory_raw])
    loc_norm = np.array([normalize(x, 0.3, 1.2) for x in location_raw])

    final_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if is_honeypot_arr[i]:
            final_scores[i] = -np.inf
        else:
            sc = search_counts[i]
            sdm = search_depth_modifier(sc)
            loc_raw = location_raw[i]
            if sc >= 3 and loc_raw < 0.8:
                loc_raw = 0.8
            ln = normalize(loc_raw, 0.3, 1.2)
            loc_norm[i] = ln
            base = (
                0.40 * skill_tax[i]
                + 0.12 * title_norm[i]
                + 0.15 * traj_norm[i]
                + 0.15 * experience_raw[i]
                + 0.08 * ln
                + 0.10 * behavioral_raw_arr[i]
            )
            ns = n_skills_list[i]
            if ns < 5:
                penalty = 1.0 - 0.35 * math.exp(-ns / 1.5)
                base *= penalty
            final_scores[i] = base * modifier_raw[i] * sdm

    log(f"Scoring: {time.time() - t3:.2f}s")

    t4 = time.time()
    order = np.argsort(final_scores)[::-1]
    top_indices = order[:100]
    top_scores = final_scores[top_indices]
    log(f"Sort + select: {time.time() - t4:.2f}s")

    rows = []
    csv_rows = []
    for rank_idx, idx in enumerate(top_indices):
        c = candidates[idx]
        cid = c.get("candidate_id", "")
        profile = c.get("profile", {})
        title = profile.get("current_title", "")
        company = profile.get("current_company", "")
        yoe = profile.get("years_of_experience", 0) or 0
        location = profile.get("location", "")
        country = profile.get("country", "")
        ns = int(n_skills_list[idx])
        score = float(top_scores[rank_idx])
        reasoning = generate_reasoning(c, rank_idx + 1, ns)

        skill_names = [s.get("name", "") for s in c.get("skills", []) if s.get("name", "") in TIER_WEIGHT]

        rows.append({
            "Rank": rank_idx + 1,
            "Candidate": cid,
            "Title": title,
            "Company": company,
            "YOE": yoe,
            "Location": f"{location}, {country}" if location else country or "",
            "Skills": ", ".join(skill_names) if skill_names else "—",
            "Score": f"{score:.4f}",
        })
        csv_rows.append([cid, rank_idx + 1, f"{score:.6f}", reasoning])

    total_time = time.time() - t0
    log(f"Total: {total_time:.2f}s")

    csv_path = csv_download_path()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        writer.writerows(csv_rows)

    df = pd.DataFrame(rows)
    summary = {
        "Total candidates": N,
        "Non-honeypots": valid_count,
        "Honeypots detected": honeypot_count,
        "Top score": f"{float(top_scores[0]):.4f}",
        "Bottom score (rank 100)": f"{float(top_scores[99]):.4f}",
        "Runtime": f"{total_time:.2f}s",
    }

    return df, log_buf.getvalue(), summary, gr.DownloadButton(label="Download CSV", value=csv_path, visible=True)


def build_app():
    with gr.Blocks(
        title="Redrob AI Candidate Ranker",
        theme=gr.themes.Soft(),
        css="""
        .app-header { text-align: center; margin-bottom: 1rem; }
        .app-header h1 { margin-bottom: 0.2rem; }
        .app-header p { color: #666; font-size: 0.95rem; }
        """
    ) as app:
        gr.HTML(
            """
            <div class="app-header">
                <h1>Redrob AI Candidate Ranker</h1>
                <p>Taxonomy-based ranking for Senior AI Engineer — Founding Team role</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                jsonl_input = gr.File(
                    label="Upload candidates.jsonl",
                    file_types=[".jsonl"],
                    file_count="single",
                )

                run_btn = gr.Button("Run Ranking", variant="primary", size="lg")

                with gr.Accordion("Scoring Parameters", open=False):
                    gr.Markdown(
                        """
                        **Final Score = raw_base × skill_penalty × behavioral_modifier × search_depth_modifier**

                        | Component | Weight |
                        |---|---|
                        | Skill Taxonomy | 0.40 |
                        | Title Fit | 0.12 |
                        | Career Trajectory | 0.15 |
                        | Experience Fit | 0.15 |
                        | Location Fit | 0.08 |
                        | Behavioral Raw | 0.10 |

                        **Penalties:** <5 taxonomy skills, not open to work, low response rate, stale activity, long notice, low interview completion.
                        **Boost:** Search depth (up to 1.15×).
                        """
                    )

                log_output = gr.Textbox(
                    label="Run Log",
                    lines=12,
                    max_lines=20,
                    interactive=False,
                )

            with gr.Column(scale=2):
                summary_json = gr.JSON(label="Summary", value={})

                results_table = gr.Dataframe(
                    label="Top 100 Results",
                    headers=["Rank", "Candidate", "Title", "Company", "YOE", "Location", "Skills", "Score"],
                    datatype=["number", "str", "str", "str", "number", "str", "str", "str"],
                    col_count=8,
                    wrap=True,
                )

                download_csv = gr.DownloadButton(label="Download CSV", visible=False)

        run_btn.click(
            fn=run_ranking,
            inputs=[jsonl_input],
            outputs=[results_table, log_output, summary_json, download_csv],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
