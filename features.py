import math
from datetime import datetime

REFERENCE_DATE = datetime(2026, 6, 7)

TIER_A = {
    "Embeddings", "Information Retrieval", "Vector Search", "FAISS",
    "Pinecone", "RAG", "Learning to Rank", "NDCG", "MRR", "MAP",
    "Recommendation Systems", "Search", "Sentence Transformers", "BM25",
    "Fine-tuning LLMs", "LoRA", "Ranking",
    "Milvus", "Qdrant", "Weaviate", "OpenSearch",
    "Semantic Search", "Dense Retrieval", "Hybrid Search",
    "Cross-Encoder", "Bi-Encoder", "Re-ranking",
}
TIER_B = {
    "Machine Learning", "Deep Learning", "PyTorch", "TensorFlow",
    "XGBoost", "NLP", "Hugging Face Transformers", "Python",
    "MLflow", "MLOps", "scikit-learn", "Feature Engineering",
    "Neural Networks", "Transformer",
}
TIER_C = {
    "LangChain", "LlamaIndex", "Prompt Engineering", "Kubeflow",
    "Elasticsearch", "Weights & Biases", "OpenAI", "Anthropic", "Gemini",
}
ANTI_SKILLS = {
    "Photoshop", "Illustrator", "Accounting", "Sales", "Marketing",
    "Tailwind", "PowerPoint", "Excel", "HTML",
}

PROFICIENCY_WEIGHT = {
    "beginner": 0.3,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}

SKILL_TAXONOMY = TIER_A | TIER_B | TIER_C
TIER_WEIGHT = {}
for s in TIER_A:   TIER_WEIGHT[s] = 3
for s in TIER_B:   TIER_WEIGHT[s] = 2
for s in TIER_C:   TIER_WEIGHT[s] = 1

PRODUCT_COMPANIES = {
    "Swiggy", "Zomato", "Razorpay", "Flipkart", "Ola", "Nykaa",
    "InMobi", "CRED", "Meesho", "Zoho", "Freshworks", "Vedantu",
    "BYJU'S", "Paytm", "PhonePe", "Dream11", "Unacademy",
    "PolicyBazaar", "PharmEasy", "upGrad", "Fractal", "ThoughtWorks",
    "Sprinklr", "Postman", "BrowserStack", "Uber", "Google", "Meta",
    "Amazon", "Microsoft", "Apple", "Netflix", "LinkedIn", "Twitter",
    "MakeMyTrip", "BookMyShow", "Urban Company", "Cure.fit", "Rapido",
    "ShareChat", "Licious", "Porter", "Spinny", "Dunzo",
    "HackerRank", "HackerEarth", "Hasura", "Chargebee",
}

CONSULTING_COMPANIES = {
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Mindtree",
    "Accenture", "Cognizant", "Capgemini", "Mphasis", "LTI",
    "L&T Infotech", "Persistent", "Hexaware", "Coforge", "KPIT",
    "LTIMindtree", "Cyient",
}

PREFERRED_LOCATIONS = {
    "Pune", "Noida", "Delhi", "Mumbai", "Hyderabad", "Bangalore", "Gurgaon",
}

TIER_1_TITLES = [
    "ML Engineer", "Applied ML Engineer", "NLP Engineer", "Search Engineer",
    "Recommendation Systems Engineer", "AI Engineer", "AI Specialist",
]

TITLE_ALIASES = {
    "machine learning engineer": "ML Engineer",
    "lead ml engineer": "ML Engineer",
    "staff ml engineer": "ML Engineer",
    "principal ml engineer": "ML Engineer",
    "senior machine learning engineer": "ML Engineer",
    "sr machine learning engineer": "ML Engineer",
    "sr ml engineer": "ML Engineer",
    "staff machine learning engineer": "ML Engineer",
    "principal machine learning engineer": "ML Engineer",
}
TIER_2_TITLES = [
    "Data Scientist", "Data Engineer", "Software Engineer", "Backend Engineer",
    "Senior Software Engineer", "Full Stack Developer", "Cloud Engineer",
    "Senior Data Engineer", "DevOps Engineer", "Data Analyst",
    "Senior Software Developer", "Platform Engineer", "Infrastructure Engineer",
    "Java Developer", "Frontend Engineer", "Software Developer",
    "Android Developer", "iOS Developer", "Site Reliability Engineer",
    "Systems Engineer", "Security Engineer", "Network Engineer",
]
TIER_3_TITLES = [
    "Project Manager", "Business Analyst", "QA Engineer", "Product Manager",
    "Technical Program Manager", "Engineering Manager",
]
TIER_4_TITLES = [
    "Accountant", "HR Manager", "Sales Executive", "Content Writer",
    "Customer Support", "Graphic Designer", "Marketing Manager",
    "Civil Engineer", "Mechanical Engineer",
]

EDUCATION_TIER_MAP = {
    "tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5, "tier_4": 0.25,
    "unknown": 0.3,
}


def normalize(raw, lo, hi):
    if raw <= lo:
        return 0.0
    if raw >= hi:
        return 1.0
    return (raw - lo) / (hi - lo)


def trust_mult(skill):
    pw = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.3)
    duration = skill.get("duration_months", 0) or 0
    endorsements = skill.get("endorsements", 0) or 0
    duration_factor = min(duration / 24, 1.0)
    endorse_factor = min(math.log2(endorsements + 2) / 5, 1.0)
    return pw * duration_factor * endorse_factor


def compute_taxonomy_score(skills):
    skills_dedup = {}
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        if name not in skills_dedup:
            skills_dedup[name] = skill
        else:
            existing = trust_mult(skills_dedup[name])
            new = trust_mult(skill)
            if new > existing:
                skills_dedup[name] = skill

    raw = 0.0
    for name, skill in skills_dedup.items():
        if name in TIER_WEIGHT:
            tw = TIER_WEIGHT[name]
            tm = trust_mult(skill)
            raw += tm * tw
        elif name in ANTI_SKILLS:
            pass

    K = 8.0
    return raw / (raw + K) if raw > 0 else 0.0


def classify_title(title):
    title_lower = title.lower()
    for tier_titles, is_exact in [(TIER_1_TITLES, False), (TIER_2_TITLES, False),
                                   (TIER_3_TITLES, False), (TIER_4_TITLES, True)]:
        for tt in tier_titles:
            if is_exact:
                if title_lower == tt.lower():
                    return tier_titles[0]
            else:
                if tt.lower() in title_lower:
                    return tt
    normalized = TITLE_ALIASES.get(title_lower)
    if normalized:
        return normalized
    return None


def title_raw_score(title):
    t = classify_title(title)
    if t is None:
        return -0.15
    if t in TIER_1_TITLES:
        return 0.40
    if t in TIER_2_TITLES:
        return 0.25
    if t in TIER_3_TITLES:
        return 0.10
    if t in TIER_4_TITLES:
        return -0.30
    return 0.0


def has_ever_held_tier1(career_history):
    for ch in career_history:
        t = classify_title(ch.get("title", ""))
        if t and t in TIER_1_TITLES:
            return True
    return False


def company_bonus(company):
    if company in PRODUCT_COMPANIES:
        return 0.15
    if company in CONSULTING_COMPANIES:
        return -0.15
    return 0.0


HEADLINE_KEYWORDS = {"search", "ranking", "retrieval", "recommendation"}

def compute_headline_boost(headline):
    if not headline:
        return 0.0
    headline_lower = headline.lower()
    matches = sum(1 for kw in HEADLINE_KEYWORDS if kw in headline_lower)
    if matches >= 2:
        return 0.05
    if matches == 1:
        return 0.025
    return 0.0


def compute_career_trajectory_raw(career_history):
    if not career_history:
        return 0.0
    best_company_score = company_bonus(career_history[0].get("company", ""))
    for ch in career_history[1:]:
        best_company_score = max(best_company_score, company_bonus(ch.get("company", "")))
    trajectory_bonus = 0.10 if has_ever_held_tier1(career_history) else 0.0
    durations = [ch.get("duration_months", 0) or 0 for ch in career_history if ch.get("duration_months")]
    if durations:
        avg_tenure = sum(durations) / len(durations)
        years = avg_tenure / 12
        hopper_penalty = min(max((1.5 - years) / 1.5, 0), 1.0) * 0.10
    else:
        hopper_penalty = 0.0
    all_research = all("Research" in ch.get("title", "") for ch in career_history)
    has_product = any(company_bonus(ch.get("company", "")) > 0 for ch in career_history)
    total_exp_years = sum(ch.get("duration_months", 0) or 0 for ch in career_history) / 12
    pure_research_penalty = 0.15 if (all_research and not has_product and total_exp_years > 2) else 0.0
    headline_boost = compute_headline_boost(career_history[0].get("headline", "") if career_history else "")
    return best_company_score + trajectory_bonus - hopper_penalty - pure_research_penalty + headline_boost


def compute_title_fit_raw(candidate):
    return title_raw_score(candidate.get("profile", {}).get("current_title", ""))


def compute_experience_fit(years):
    if years < 2:
        return 0.0
    if years > 15:
        return 0.1
    if 4.5 <= years <= 9.5:
        return 1.0
    mu, sigma = 7.0, 2.5
    return math.exp(-0.5 * ((years - mu) / sigma) ** 2)


def compute_education_fit(education):
    if not education:
        return 0.0
    best = 0.0
    for entry in education:
        tier = entry.get("tier", "unknown")
        score = EDUCATION_TIER_MAP.get(tier, 0.3)
        deg = entry.get("degree", "")
        if any(deg.startswith(p) for p in {"M.", "Ph.D"}):
            score = min(score + 0.1, 1.0)
        best = max(best, score)
    return best


def compute_location_raw(location, country, willing_to_relocate):
    base = 0.3
    if country == "India":
        base = 0.6
        if any(city in location for city in PREFERRED_LOCATIONS):
            base = 1.0
    relo_bonus = 0.2 if willing_to_relocate else 0.0
    return base + relo_bonus


def compute_behavioral_raw(signals, caps=None):
    c = caps or {}
    return (
        0.20 * (signals.get("profile_completeness_score", 0) / 100)
        + 0.15 * min(signals.get("search_appearance_30d", 0) / c.get("search_appearance_30d", 462), 1.0)
        + 0.15 * min(signals.get("saved_by_recruiters_30d", 0) / c.get("saved_by_recruiters_30d", 27), 1.0)
        + 0.15 * signals.get("interview_completion_rate", 0)
        + 0.15 * min(signals.get("recruiter_response_rate", 0) / 0.5, 1.0)
        + 0.10 * min(signals.get("connection_count", 0) / c.get("connection_count", 894), 1.0)
        + 0.10 * min(signals.get("endorsements_received", 0) / c.get("endorsements_received", 94), 1.0)
    )


def recency_bonus(last_active_date):
    if not last_active_date:
        return 0.0
    try:
        dt = datetime.strptime(last_active_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return 0.0
    days = (REFERENCE_DATE - dt).days
    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.8
    if days <= 90:
        return 0.5
    if days <= 180:
        return 0.2
    return 0.0


def compute_behavioral_modifier(signals):
    modifier = 1.0
    modifier -= 0.25 * (1.0 - (1.0 if signals.get("open_to_work_flag") else 0.0))
    modifier -= 0.15 * (1.0 - min(signals.get("recruiter_response_rate", 0) / 0.5, 1.0))
    modifier -= 0.15 * (1.0 - recency_bonus(signals.get("last_active_date")))
    notice = signals.get("notice_period_days", 0) or 0
    modifier -= 0.10 * max(notice - 30, 0) / 150
    modifier -= 0.10 * (1.0 - signals.get("interview_completion_rate", 0))
    return max(modifier, 0.3)


def is_honeypot(candidate):
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0

    if yoe < 0 or yoe > 50:
        return True

    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months", 0) == 0:
            return True

    career = candidate.get("career_history", [])
    total_career_months = sum((ch.get("duration_months", 0) or 0) for ch in career)
    career_years = total_career_months / 12
    if career_years > 0 and abs(yoe - career_years) > 5:
        return True

    return False


def generate_reasoning(candidate, rank, n_skills):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    history = candidate.get("career_history", [])
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    location = profile.get("location", "")
    country = profile.get("country", "")
    yrs = profile.get("years_of_experience", 0)
    rr = signals.get("recruiter_response_rate", 0)
    notice = signals.get("notice_period_days", 0) or 0
    headline = profile.get("headline", "")

    top_skill = ""
    for s in candidate.get("skills", []):
        name = s.get("name", "")
        if name in TIER_WEIGHT:
            top_skill = name
            break

    prev_role = history[-1].get("title", "") if len(history) >= 1 else title
    current_role = title

    strengths = []
    if title_raw_score(title) >= 0.25:
        strengths.append(f"{title} background")
    if n_skills >= 5:
        strengths.append(f"{n_skills} AI skills")
    if rr >= 0.5:
        strengths.append("responsive")
    if any(co in company for co in PRODUCT_COMPANIES):
        strengths.append("product experience")
    concern = ""
    if notice > 60:
        concern = f"notice {notice}d"
    elif rr < 0.2:
        concern = "low response rate"

    strength_str = strengths[0] if strengths else f"{yrs}yrs experience"

    templates = [
        f"{title} at {company} -- {yrs:.0f}yrs. {n_skills} AI skills. {strength_str}. {concern}".strip(". ") + ".",
        f"{headline.split('|')[0].strip()} -- {yrs:.0f}yrs in {industry}. Top skill: {top_skill}. RR {rr:.0%}.",
        f"From {prev_role} to {current_role}: {yrs:.0f}yrs progression. {n_skills} relevant skills. {location}.",
        f"{company} ({industry}) | {title} | {yrs:.0f}yrs. Key: {top_skill or 'general eng'}. Gap: {concern or 'none'}.",
    ]
    idx = (rank - 1) % len(templates)
    return templates[idx]


def debug_taxonomy(skills):
    skills_dedup = {}
    for skill in skills:
        name = skill.get("name", "")
        if not name:
            continue
        if name not in skills_dedup:
            skills_dedup[name] = skill
        else:
            existing = trust_mult(skills_dedup[name])
            new = trust_mult(skill)
            if new > existing:
                skills_dedup[name] = skill

    entries = []
    raw = 0.0
    for name, skill in skills_dedup.items():
        if name in TIER_WEIGHT:
            tw = TIER_WEIGHT[name]
            tm = trust_mult(skill)
            contrib = tm * tw
            raw += contrib
            tier_label = "A" if name in TIER_A else ("B" if name in TIER_B else "C")
            entries.append({
                "name": name,
                "tier": tier_label,
                "proficiency": skill.get("proficiency", ""),
                "duration": skill.get("duration_months", 0) or 0,
                "endorsements": skill.get("endorsements", 0) or 0,
                "trust_mult": tm,
                "contribution": contrib,
            })

    K = 8.0
    score = raw / (raw + K) if raw > 0 else 0.0
    return {"score": score, "raw": raw, "K": K, "entries": entries}


def debug_title_fit(title):
    t = classify_title(title)
    raw_score = title_raw_score(title)
    if t is None:
        tier = "no match"
    elif t in TIER_1_TITLES:
        tier = "Tier 1"
    elif t in TIER_2_TITLES:
        tier = "Tier 2"
    elif t in TIER_3_TITLES:
        tier = "Tier 3"
    elif t in TIER_4_TITLES:
        tier = "Tier 4"
    else:
        tier = "unknown"
    return {"score": raw_score, "tier": tier, "matched": t}


def debug_trajectory(career_history):
    if not career_history:
        return {"score": 0.0, "roles": [], "best_company_score": 0.0, "has_tier1": False,
                "trajectory_bonus": 0.0, "avg_tenure_years": 0.0, "hopper_penalty": 0.0,
                "all_research": False, "has_product": False, "total_exp_years": 0.0,
                "pure_research_penalty": 0.0}

    role_details = []
    for ch in career_history:
        co = ch.get("company", "")
        title = ch.get("title", "")
        bonus = company_bonus(co)
        ct = classify_title(title)
        is_tier1 = ct is not None and ct in TIER_1_TITLES
        role_details.append({"company": co, "title": title, "bonus": bonus, "is_tier1": is_tier1})

    best_company_score = max(r["bonus"] for r in role_details)
    has_tier1 = any(r["is_tier1"] for r in role_details)
    trajectory_bonus = 0.10 if has_tier1 else 0.0

    durations = [ch.get("duration_months", 0) or 0 for ch in career_history if ch.get("duration_months")]
    if durations:
        avg_tenure = sum(durations) / len(durations)
        years = avg_tenure / 12
        hopper_penalty = min(max((1.5 - years) / 1.5, 0), 1.0) * 0.10
    else:
        years = 0.0
        hopper_penalty = 0.0

    all_research = all("Research" in ch.get("title", "") for ch in career_history)
    has_product = any(company_bonus(ch.get("company", "")) > 0 for ch in career_history)
    total_exp_years = sum(ch.get("duration_months", 0) or 0 for ch in career_history) / 12
    pure_research_penalty = 0.15 if (all_research and not has_product and total_exp_years > 2) else 0.0

    headline_boost = compute_headline_boost(career_history[0].get("headline", "") if career_history else "")
    score = best_company_score + trajectory_bonus - hopper_penalty - pure_research_penalty + headline_boost

    return {
        "score": score,
        "roles": role_details,
        "best_company_score": best_company_score,
        "has_tier1": has_tier1,
        "trajectory_bonus": trajectory_bonus,
        "avg_tenure_years": years,
        "hopper_penalty": hopper_penalty,
        "all_research": all_research,
        "has_product": has_product,
        "total_exp_years": total_exp_years,
        "pure_research_penalty": pure_research_penalty,
        "headline_boost": headline_boost,
    }


def debug_experience_fit(years):
    score = compute_experience_fit(years)
    if years < 2:
        reason = f"{years:.1f}yr < 2yr minimum → 0.0"
    elif years > 15:
        reason = f"{years:.1f}yr > 15yr maximum → 0.1"
    elif 4.5 <= years <= 9.5:
        reason = f"{years:.1f}yr in sweet spot plateau [4.5, 9.5] → 1.0"
    else:
        mu, sigma = 7.0, 2.5
        z = (years - mu) / sigma
        reason = f"Gaussian: μ=7, σ=2.5, z={z:.3f} → exp(-0.5×{z:.3f}²) = {score:.4f}"
    return {"score": score, "years": years, "reason": reason}


def debug_location(location, country, willing_to_relocate):
    base = 0.3
    city_note = "outside India"
    if country == "India":
        base = 0.6
        city_note = "India (non-preferred city)"
        if any(city in location for city in PREFERRED_LOCATIONS):
            base = 1.0
            city_note = "India (preferred city)"
    relo_bonus = 0.2 if willing_to_relocate else 0.0
    raw = base + relo_bonus
    return {
        "raw": raw,
        "base": base,
        "relo_bonus": relo_bonus,
        "location_note": city_note,
        "willing_to_relocate": willing_to_relocate,
    }


def debug_behavioral_raw(signals, caps=None):
    c = caps or {}
    signal_configs = [
        ("profile_completeness", signals.get("profile_completeness_score", 0), 100.0, 0.20, "val / 100"),
        ("search_appearance_30d", signals.get("search_appearance_30d", 0), c.get("search_appearance_30d", 462), 0.15, "min(val / cap, 1)"),
        ("saved_by_recruiters_30d", signals.get("saved_by_recruiters_30d", 0), c.get("saved_by_recruiters_30d", 27), 0.15, "min(val / cap, 1)"),
        ("interview_completion_rate", signals.get("interview_completion_rate", 0), 1.0, 0.15, "direct (0-1)"),
        ("recruiter_response_rate", signals.get("recruiter_response_rate", 0), 0.5, 0.15, "min(val / 0.5, 1)"),
        ("connection_count", signals.get("connection_count", 0), c.get("connection_count", 894), 0.10, "min(val / cap, 1)"),
        ("endorsements_received", signals.get("endorsements_received", 0), c.get("endorsements_received", 94), 0.10, "min(val / cap, 1)"),
    ]
    signals_detail = []
    total = 0.0
    for name, raw_val, denom, weight, note in signal_configs:
        norm = min(raw_val / denom, 1.0) if denom > 0 else 0.0
        contrib = norm * weight
        total += contrib
        signals_detail.append({
            "name": name, "raw": raw_val, "denom": denom, "norm": norm, "weight": weight, "contrib": contrib, "note": note
        })
    return {"total": total, "signals": signals_detail}


def debug_behavioral_modifier(signals):
    penalties = []
    modifier = 1.0

    o2w = 1.0 if signals.get("open_to_work_flag") else 0.0
    p = 0.25 * (1.0 - o2w)
    modifier -= p
    penalties.append({"name": "not open to work", "penalty": p, "detail": f"flag={bool(o2w)}", "formula": "0.25 × (1 − flag)"})

    rr = signals.get("recruiter_response_rate", 0)
    p = 0.15 * (1.0 - min(rr / 0.5, 1.0))
    modifier -= p
    penalties.append({"name": "low recruiter response", "penalty": p, "detail": f"rr={rr:.2f}", "formula": "0.15 × (1 − min(rr/0.5, 1))"})

    rb = recency_bonus(signals.get("last_active_date", ""))
    p = 0.15 * (1.0 - rb)
    modifier -= p
    penalties.append({"name": "stale activity", "penalty": p, "detail": f"recency_bonus={rb:.1f}", "formula": "0.15 × (1 − recency_bonus)"})

    notice = signals.get("notice_period_days", 0) or 0
    p = 0.10 * max(notice - 30, 0) / 150
    modifier -= p
    penalties.append({"name": "long notice", "penalty": p, "detail": f"notice={notice}d", "formula": "0.10 × max(notice−30, 0) / 150"})

    icr = signals.get("interview_completion_rate", 0)
    p = 0.10 * (1.0 - icr)
    modifier -= p
    penalties.append({"name": "low interview completion", "penalty": p, "detail": f"icr={icr:.2f}", "formula": "0.10 × (1 − icr)"})

    modifier = max(modifier, 0.3)
    return {"modifier": modifier, "penalties": penalties}
