import json
import csv
import os
import sys
import time
import math
from datetime import datetime
import numpy as np
from features import (
    is_honeypot, compute_taxonomy_score, compute_career_trajectory_raw,
    compute_title_fit_raw, compute_experience_fit, compute_location_raw,
    compute_behavioral_raw, compute_behavioral_modifier, generate_reasoning,
    normalize, TIER_WEIGHT,
    debug_taxonomy, debug_title_fit, debug_trajectory,
    debug_experience_fit, debug_location,
    debug_behavioral_raw, debug_behavioral_modifier,
)


def load_candidates(path):
    candidates = []
    with open(path, "r") as f:
        for line in f:
            candidates.append(json.loads(line))
    return candidates


def main():
    t0 = time.time()
    np.random.seed(42)

    print("Loading candidates...", file=sys.stderr)
    candidates = load_candidates("candidates.jsonl")
    N = len(candidates)
    print(f"Loaded {N} candidates in {time.time() - t0:.2f}s", file=sys.stderr)

    t1 = time.time()
    is_honeypot_arr = np.zeros(N, dtype=bool)

    signal_collector = {
        "search_appearance_30d": [],
        "saved_by_recruiters_30d": [],
        "connection_count": [],
        "endorsements_received": [],
    }

    for i, c in enumerate(candidates):
        if is_honeypot(c):
            is_honeypot_arr[i] = True
            continue
        sigs = c.get("redrob_signals", {})
        for k in signal_collector:
            signal_collector[k].append(sigs.get(k, 0))
        if i % 10000 == 0 and i > 0:
            print(f"  pre-scan {i}/{N}", file=sys.stderr)

    behavioral_caps = {}
    for k, vals in signal_collector.items():
        if vals:
            behavioral_caps[k] = max(float(np.percentile(vals, 99)), 1.0)
        else:
            behavioral_caps[k] = 1.0

    valid_count = int(N - np.sum(is_honeypot_arr))
    print(f"Computed P99 behavioral caps (from {valid_count} non-honeypots):", file=sys.stderr)
    for k, v in behavioral_caps.items():
        print(f"  {k}: {v:.0f}", file=sys.stderr)

    skill_tax = np.zeros(N, dtype=np.float32)
    title_fit_raw = np.zeros(N, dtype=np.float32)
    trajectory_raw = np.zeros(N, dtype=np.float32)
    experience_raw = np.zeros(N, dtype=np.float32)
    location_raw = np.zeros(N, dtype=np.float32)
    behavioral_raw = np.zeros(N, dtype=np.float32)
    modifier_raw = np.ones(N, dtype=np.float32)
    n_skills_list = np.zeros(N, dtype=np.int32)

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

    search_counts = np.zeros(N, dtype=np.int32)

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
        behavioral_raw[i] = compute_behavioral_raw(signals, behavioral_caps)
        modifier_raw[i] = compute_behavioral_modifier(signals)
        n_skills_list[i] = sum(1 for s in skills if s.get("name", "") in TIER_WEIGHT)
        search_counts[i] = search_depth_count(skills)

        if i % 10000 == 0 and i > 0:
            print(f"  features {i}/{N}", file=sys.stderr)

    print(f"Feature extraction: {time.time() - t1:.2f}s", file=sys.stderr)

    t2 = time.time()
    skill_match = skill_tax

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
                0.40 * skill_match[i]
                + 0.12 * title_norm[i]
                + 0.15 * traj_norm[i]
                + 0.15 * experience_raw[i]
                + 0.08 * ln
                + 0.10 * behavioral_raw[i]
            )
            ns = n_skills_list[i]
            if ns < 5:
                penalty = 1.0 - 0.35 * math.exp(-ns / 1.5)
                base *= penalty
            final_scores[i] = base * modifier_raw[i] * sdm

    print(f"Scoring: {time.time() - t2:.2f}s", file=sys.stderr)

    t3 = time.time()
    order = np.argsort(final_scores)[::-1]
    top_indices = order[:100]

    top_scores = final_scores[top_indices]
    top_candidates = [candidates[idx] for idx in top_indices]
    top_n_skills = n_skills_list[top_indices]

    print(f"Sort + select: {time.time() - t3:.2f}s", file=sys.stderr)

    t4 = time.time()
    reasoning_list = []
    for rank_idx, (idx, c, ns) in enumerate(zip(top_indices, top_candidates, top_n_skills)):
        r = generate_reasoning(c, rank_idx + 1, int(ns))
        reasoning_list.append(r)

    print(f"Reasoning generation: {time.time() - t4:.2f}s", file=sys.stderr)

    t_report = time.time()
    os.makedirs("logs", exist_ok=True)
    with open("logs/ranking_report.md", "w") as f:
        f.write("# Ranking Report — Top 100\n\n")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"**Generated:** {now_str}  \n")
        f.write(f"**Reference Date:** 2026-06-07  \n\n")

        f.write("## Scoring Parameters\n\n")
        f.write("| Component | Weight | Normalization |\n")
        f.write("|---|---|---|\n")
        f.write("| Skill Match | 0.40 | taxonomy → [0, 1] |\n")
        f.write("| Title Fit | 0.12 | normalize([-0.30, 0.40]) → [0, 1] |\n")
        f.write("| Career Trajectory | 0.15 | normalize([-0.30, 0.30]) → [0, 1] |\n")
        f.write("| Experience Fit | 0.15 | Gaussian μ=7 σ=2.5, plateau [4.5, 9.5] → [0, 1] |\n")
        f.write("| Location Fit | 0.08 | normalize([0.3, 1.2]) → [0, 1]; bumped to 0.8 raw if ≥3 search skills |\n")
        f.write("| Behavioral Raw | 0.10 | weighted sum of 7 signals → [0, 1] |\n\n")
        f.write("**Final Score:** `raw_base × skill_penalty × behavioral_modifier × search_depth_modifier`  \n")
        f.write("**Skill Penalty:** if n_skills < 5 → `1.0 − 0.35 × exp(−n_skills / 1.5)`  \n")
        f.write("**Behavioral Modifier:** starts at 1.0, subtracts up to 5 penalties, floor 0.3  \n")
        f.write("**Search Depth Modifier:** 1.0 + 0.05 × search_core_skills, capped at 1.15×  \n\n")

        f.write("### Behavioral Caps (P99 from non-honeypots)\n\n")
        f.write(f"| Signal | Cap |\n|---|---|\n")
        for k, v in behavioral_caps.items():
            f.write(f"| {k} | {v:.0f} |\n")

        norm_headers = ["Component", "Raw Value", "Normalized", "Weight", "Contribution"]

        for rank_idx, idx in enumerate(top_indices):
            rank = rank_idx + 1
            c = candidates[idx]
            cid = c.get("candidate_id", "")
            profile = c.get("profile", {})
            signals = c.get("redrob_signals", {})
            skills = c.get("skills", [])
            career_hist = c.get("career_history", [])
            title = profile.get("current_title", "")
            company = profile.get("current_company", "")
            yrs = profile.get("years_of_experience", 0) or 0
            location = profile.get("location", "")
            country = profile.get("country", "")
            ns = int(n_skills_list[idx])
            sc = int(search_counts[idx])
            sdm = search_depth_modifier(sc)

            sm = float(skill_match[idx])
            tn = float(title_norm[idx])
            trj = float(traj_norm[idx])
            ef = float(experience_raw[idx])
            ln = float(loc_norm[idx])
            br = float(behavioral_raw[idx])
            base = 0.40 * sm + 0.12 * tn + 0.15 * trj + 0.15 * ef + 0.08 * ln + 0.10 * br

            if ns < 5:
                sk_penalty = 1.0 - 0.35 * math.exp(-ns / 1.5)
                base_after = base * sk_penalty
            else:
                sk_penalty = 1.0
                base_after = base

            mod = float(modifier_raw[idx])
            final = float(final_scores[idx])

            f.write(f"\n---\n\n")
            f.write(f"### Rank {rank} — {cid}\n\n")
            f.write(f"**Title:** {title}  \n")
            f.write(f"**Company:** {company}  \n")
            f.write(f"**YOE:** {yrs:.1f} | **Location:** {location} | **Country:** {country}  \n\n")

            f.write("#### Component Breakdown\n\n")
            f.write("| " + " | ".join(norm_headers) + " |\n")
            f.write("|---|---|---|---|---|\n")
            components = [
                ("Skill Match", sm, sm, 0.40),
                ("Title Fit (norm)", float(title_fit_raw[idx]), tn, 0.12),
                ("Career Trajectory (norm)", trj, trj, 0.15),
                ("Experience Fit", ef, ef, 0.15),
                ("Location (norm)", ln, ln, 0.08),
                ("Behavioral Raw", br, br, 0.10),
            ]
            for label, raw_val, norm_val, weight in components:
                f.write(f"| {label} | {raw_val:.4f} | {norm_val:.4f} | {weight:.2f} | {norm_val * weight:.4f} |\n")
            f.write(f"\n")

            f.write("| Step | Value |\n")
            f.write("|---|---|\n")
            f.write(f"| Raw Base | {base:.6f} |\n")
            f.write(f"| Skill Count | {ns} {'(≥ 5, no penalty)' if ns >= 5 else '(< 5)'} |\n")
            if ns < 5:
                f.write(f"| Penalty Factor | 1 − 0.35 × exp(−{ns}/{1.5}) = {sk_penalty:.4f} |\n")
            f.write(f"| Base after Penalty | {base_after:.6f} |\n")
            f.write(f"| Behavioral Modifier | {mod:.4f}× |\n")
            f.write(f"| Search Core Skills | {sc} |\n")
            f.write(f"| Search Depth Modifier | {sdm:.4f}× |\n")
            f.write(f"| **Final Score** | **{final:.6f}** |\n\n")

            tax = debug_taxonomy(skills)
            if tax["entries"]:
                f.write("#### Skill Taxonomy Detail\n\n")
                f.write("| Skill | Tier | Proficiency | Duration | Endorsements | Trust Mult | Contribution |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for e in tax["entries"]:
                    f.write(f"| {e['name']} | {e['tier']} (×{'3' if e['tier']=='A' else '2' if e['tier']=='B' else '1'}) | {e['proficiency']} | {e['duration']}mo | {e['endorsements']} | {e['trust_mult']:.3f} | {e['contribution']:.3f} |\n")
                f.write(f"\n**Raw Total:** {tax['raw']:.3f} | **K:** {tax['K']} | **Score:** {tax['raw']} / ({tax['raw']} + {tax['K']}) = **{tax['score']:.4f}**  \n\n")
            else:
                f.write("#### Skill Taxonomy\n\n*No taxonomy-relevant skills.*\n\n")

            f.write("#### Semantic Match Detail\n\n*Semantic component removed — taxonomy-only scoring.*\n\n")

            tfd = debug_title_fit(title)
            f.write("#### Title Fit Detail\n\n")
            f.write(f"- **Current Title:** {title}\n")
            f.write(f"- **Classification:** {tfd['tier']} → raw score = {tfd['score']:+.2f}\n")
            f.write(f"- **Matched:** {tfd['matched'] or '—'}\n")
            f.write(f"- **Normalization:** normalize({tfd['score']:+.2f}, -0.30, 0.40) = **{tn:.4f}**\n\n")

            trd = debug_trajectory(career_hist)
            f.write("#### Career Trajectory Detail\n\n")
            if trd["roles"]:
                f.write("| Company | Title | Bonus | Tier-1 Title |\n")
                f.write("|---|---|---|---|\n")
                for r in trd["roles"]:
                    f.write(f"| {r['company']} | {r['title']} | {r['bonus']:+.2f} | {'✓' if r['is_tier1'] else ''} |\n")
            f.write(f"\n| Component | Value |\n")
            f.write("|---|---|\n")
            f.write(f"| Best Company Score | {trd['best_company_score']:+.2f} |\n")
            f.write(f"| Ever held Tier-1 title | {'Yes' if trd['has_tier1'] else 'No'} |\n")
            f.write(f"| Trajectory Bonus | +{trd['trajectory_bonus']:.2f} |\n")
            f.write(f"| Avg Tenure | {trd['avg_tenure_years']:.1f}yr |\n")
            f.write(f"| Hopper Penalty | -{trd['hopper_penalty']:.2f} |\n")
            pures_str = f"{'Yes' if trd['all_research'] else 'No'} / {'Yes' if trd['has_product'] else 'No'} / {trd['total_exp_years']:.1f}yr"
            f.write(f"| All Research / Has Product / Total Exp | {pures_str} |\n")
            f.write(f"| Pure Research Penalty | -{trd['pure_research_penalty']:.2f} |\n")
            f.write(f"| Headline Boost | +{trd['headline_boost']:.2f} |\n")
            f.write(f"| **Raw Score** | {trd['score']:+.2f} |\n")
            f.write(f"| **Normalized** | normalize({trd['score']:+.2f}, -0.30, 0.30) = **{trj:.4f}** |\n\n")

            exd = debug_experience_fit(yrs)
            f.write("#### Experience Fit Detail\n\n")
            f.write(f"- Years: {yrs:.1f}\n")
            f.write(f"- {exd['reason']}\n")
            f.write(f"- **Score: {exd['score']:.4f}**\n\n")

            locd = debug_location(location, country, profile.get("willing_to_relocate", False))
            f.write("#### Location Detail\n\n")
            f.write(f"- Location: {location}, {country}\n")
            f.write(f"- {locd['location_note']}: base = {locd['base']:.1f}\n")
            f.write(f"- Willing to relocate: {'Yes' if locd['willing_to_relocate'] else 'No'} → +{locd['relo_bonus']:.1f}\n")
            f.write(f"- Raw: {locd['raw']:.1f}\n")
            f.write(f"- **Normalized:** normalize({locd['raw']:.1f}, 0.3, 1.2) = **{ln:.4f}**\n\n")

            brd = debug_behavioral_raw(signals, behavioral_caps)
            f.write("#### Behavioral Raw Detail\n\n")
            f.write("| Signal | Raw | Denominator | Normalized | Weight | Contribution |\n")
            f.write("|---|---|---|---|---|---|\n")
            for sd in brd["signals"]:
                f.write(f"| {sd['name']} | {sd['raw']} | {sd['denom']:.0f} | {sd['norm']:.3f} | {sd['weight']:.2f} | {sd['contrib']:.4f} |\n")
            f.write(f"\n**Total: {brd['total']:.4f}**\n\n")

            bmd = debug_behavioral_modifier(signals)
            f.write("#### Behavioral Modifier Detail\n\n")
            f.write("| Penalty | Amount | Detail |\n")
            f.write("|---|---|---|\n")
            for pd in bmd["penalties"]:
                f.write(f"| {pd['name']} | {pd['penalty']:.4f} | {pd['detail']} |\n")
            f.write(f"\n**Modifier:** 1.0 − Σpenalties = {bmd['modifier']:.4f} (floor 0.3)\n\n")

            f.write("#### Reasoning\n\n")
            f.write(f"{reasoning_list[rank_idx]}\n\n")

    print(f"Report generation: {time.time() - t_report:.2f}s", file=sys.stderr)

    t5 = time.time()
    with open("submission_loc_emb_2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, (idx, r) in enumerate(zip(top_indices, reasoning_list)):
            cid = candidates[idx].get("candidate_id", "")
            writer.writerow([cid, rank_idx + 1, f"{top_scores[rank_idx]:.6f}", r])

    print(f"CSV write: {time.time() - t5:.2f}s", file=sys.stderr)
    print(f"Total: {time.time() - t0:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
