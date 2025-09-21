# person2_scoring.py
"""
Person 2: Scoring Engine

Functions:
- evaluate_resume(resume_dict, jd_dict, model=None, config=None) -> dict
- compute_hard_match(...)
- compute_semantic_score(...)
- helper functions for normalization & fuzzy matching
"""

import re
import math
from typing import List, Tuple, Dict, Optional
import numpy as np

# optional fuzzy library; fallback to difflib if not installed
try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# sentence-transformers for semantic embeddings (AI/ML)
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# ---------- Simple canonicalization / normalization ----------
SKILL_SYNONYMS = {
    # map common variants to canonical forms
    "scikit learn": "scikit-learn",
    "scikit": "scikit-learn",
    "scikit_learn": "scikit-learn",
    "python3": "python",
    "py": "python",
    "postgres": "postgresql",
    "postgre": "postgresql",
    "pytorch": "pytorch",
    "tensorflow": "tensorflow",
    "sql": "sql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "gcp",
    "google cloud": "gcp",
    "azure": "azure",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes"
    # extend as needed
}

RE_NON_ALNUM = re.compile(r'[^a-z0-9\s\-\+\.#/]')

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.replace('&', ' and ')
    s = RE_NON_ALNUM.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def canonicalize_skill(skill: str) -> str:
    s = normalize_text(skill)
    # some heuristics
    s = s.replace('(', '').replace(')', '')
    s = s.replace(' - ', ' ')
    s = s.strip()
    # map synonyms
    if s in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[s]
    # try approximate mapping keys
    if s.replace(' ', '') in SKILL_SYNONYMS:
        return SKILL_SYNONYMS[s.replace(' ', '')]
    return s

def canonicalize_list(skills: List[str]) -> List[str]:
    out = []
    seen = set()
    for s in skills:
        if not s: 
            continue
        cand = canonicalize_skill(s)
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out

# ---------- Fuzzy matching helpers ----------
def best_fuzzy_match(item: str, candidates: List[str], score_cutoff: int = 80) -> Tuple[Optional[str], int]:
    """
    Return (best_match, score) if above cutoff, else (None, 0).
    Uses rapidfuzz if available, else difflib.
    """
    if not candidates:
        return None, 0
    if _HAS_RAPIDFUZZ:
        match = rf_process.extractOne(item, candidates, scorer=rf_fuzz.WRatio)
        if match is None:
            return None, 0
        best, score, _ = match
        return best, int(score)
    else:
        # difflib fallback (score 0-100 equivalence: ratio*100)
        best_list = difflib.get_close_matches(item, candidates, n=1, cutoff=score_cutoff/100.0)
        if not best_list:
            return None, 0
        best = best_list[0]
        ratio = int(difflib.SequenceMatcher(None, item, best).ratio() * 100)
        if ratio >= score_cutoff:
            return best, ratio
        return None, 0

# ---------- Hard match ----------
def compute_hard_match(resume_skills: List[str],
                       must_have: List[str],
                       good_to_have: List[str],
                       fuzzy_threshold: int = 85,
                       weights: Tuple[float,float]=(0.8,0.2)
                       ) -> Dict:
    """
    Returns:
      {
        'hard_score': float (0-1),
        'matched': [...],
        'missing': [...],
        'details': { ... }
      }
    weights: weight for (must_have, good_to_have) when both exist
    """
    res_sk = canonicalize_list(resume_skills or [])
    must = canonicalize_list(must_have or [])
    good = canonicalize_list(good_to_have or [])

    matched = []
    matched_from_good = []
    missing = []
    matched_set = set()

    # match must-have exactly first
    for req in must:
        if req in res_sk:
            matched.append(req)
            matched_set.add(req)
        else:
            # try fuzzy match to any resume skill
            best, score = best_fuzzy_match(req, res_sk, score_cutoff=fuzzy_threshold)
            if best:
                matched.append(best)
                matched_set.add(best)
            else:
                missing.append(req)

    # match good-to-have (but give it lower weight)
    for g in good:
        if g in res_sk:
            matched_from_good.append(g)
            matched_set.add(g)
        else:
            best, score = best_fuzzy_match(g, res_sk, score_cutoff=fuzzy_threshold)
            if best:
                matched_from_good.append(best)
                matched_set.add(best)

    # compute scores
    must_score = 1.0
    if must:
        must_score = len([m for m in matched if m in must or m in matched_from_good]) / len(must)
        # note: fuzzy matches mapped to resume names are counted.

    good_score = 0.0
    if good:
        good_score = len(matched_from_good) / len(good)

    # finalize hard score using weights (if must exists)
    if must:
        hard_score = weights[0] * must_score + weights[1] * good_score
    elif good:
        hard_score = good_score
    else:
        # no explicit JD skill list provided — signal via None to prefer semantic-only
        hard_score = None

    return {
        'hard_score': hard_score,          # None or float [0,1]
        'matched': list(matched_set),
        'missing': missing,
        'must_score': must_score if must else None,
        'good_score': good_score if good else None
    }

# ---------- TF-IDF fallback (optional) ----------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_similarity(text1: str, text2: str) -> float:
    texts = [text1 or "", text2 or ""]
    tfv = TfidfVectorizer(ngram_range=(1,2), stop_words='english').fit(texts)
    vecs = tfv.transform(texts)
    sim = cosine_similarity(vecs[0], vecs[1])[0,0]
    return float(sim)

# ---------- Semantic match (AI/ML) ----------
# loads model lazily; use caching to avoid reloading every call
_MODEL_CACHE = {}

def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    model = SentenceTransformer(name)
    _MODEL_CACHE[name] = model
    return model

def compute_semantic_score(resume_text: str, jd_text: str,
                           model_name: str = "all-MiniLM-L6-v2") -> float:
    """
    Returns similarity score in 0..1 (float).
    """
    model = get_embedding_model(model_name)
    # encode as single documents (fast)
    emb_resume = model.encode(resume_text or "", convert_to_tensor=True)
    emb_jd = model.encode(jd_text or "", convert_to_tensor=True)
    sim = util.cos_sim(emb_resume, emb_jd).item()
    # clamp
    sim = max(0.0, min(1.0, float(sim)))
    return sim

# ---------- Final evaluator ----------
def evaluate_resume_detailed(resume_dict: dict,
                             jd_dict: dict,
                             semantic_weight: float = 0.4,
                             hard_match_config: dict = None,
                             model_name: str = "all-MiniLM-L6-v2"
                             ) -> Dict:
    """
    Detailed resume evaluation against JD.
    Returns:
    {
        score: int (0-100),
        verdict: High/Medium/Low,
        hard_match_pct: float,
        semantic_pct: float,
        matched_must_have: [...],
        matched_good_to_have: [...],
        missing_must_have: [...],
        missing_good_to_have: [...],
        semantic_section_scores: {section: float},
        feedback: str,
        details: { ... }
    }
    """
    if hard_match_config is None:
        hard_match_config = {}

    # Extract data
    resume_skills = resume_dict.get('skills', []) or []
    resume_text = resume_dict.get('text', '') or ' '.join(resume_skills)
    resume_sections = resume_dict.get('sections', {})
    
    jd_must = jd_dict.get('must_have', []) or []
    jd_good = jd_dict.get('good_to_have', []) or []
    jd_text = jd_dict.get('raw_text', '') or ' '.join(jd_must + jd_good)

    # ---- Hard match ----
    hard = compute_hard_match(
        resume_skills, jd_must, jd_good,
        fuzzy_threshold=hard_match_config.get('fuzzy_threshold', 85),
        weights=hard_match_config.get('weights', (0.8, 0.2))
    )
    hard_score = hard['hard_score']  # None or float

    # Separate matched must-have / good-to-have
    matched_set = set(hard.get('matched', []))
    matched_must_have = [s for s in matched_set if s in canonicalize_list(jd_must)]
    matched_good_to_have = [s for s in matched_set if s in canonicalize_list(jd_good)]
    missing_must_have = hard.get('missing', [])
    missing_good_to_have = [s for s in canonicalize_list(jd_good) if s not in matched_set]

    # ---- Semantic match ----
    semantic_score = None
    semantic_section_scores = {}
    try:
        semantic_score = compute_semantic_score(resume_text, jd_text, model_name=model_name)
        # per-section semantic
        for sec in ['skills', 'experience', 'projects']:
            sec_text = resume_sections.get(sec, '')
            if sec_text:
                semantic_section_scores[sec] = compute_semantic_score(sec_text, jd_text, model_name=model_name)
    except Exception:
        # fallback to TF-IDF
        semantic_score = tfidf_similarity(resume_text, jd_text)

    # Clamp semantic_score
    semantic_score = max(0.0, min(1.0, float(semantic_score)))
    semantic_pct = round(semantic_score * 100, 2)

    # ---- Combine scores ----
    if hard_score is None:
        # no JD skills -> rely only on semantic
        final_pct = round(semantic_score * 100)
        hard_pct = None
    else:
        hard_pct = round(float(hard_score) * 100, 2)
        w_sem = semantic_weight
        w_hard = 1.0 - w_sem
        final_pct = int(round(w_hard * hard_pct + w_sem * semantic_pct))

    # ---- Verdict ----
    if final_pct > 70:
        verdict = "High"
    elif final_pct >= 40:
        verdict = "Medium"
    else:
        verdict = "Low"

    # ---- Feedback ----
    feedback_parts = []

    if missing_must_have:
        feedback_parts.append(f"Missing must-have skills: {', '.join(missing_must_have)}.")
    if missing_good_to_have:
        feedback_parts.append(f"Missing good-to-have skills: {', '.join(missing_good_to_have)}.")
    if matched_set:
        feedback_parts.append(f"Matched skills: {', '.join(sorted(matched_set))}.")
    
    if semantic_pct < 50:
        feedback_parts.append("Resume appears not strongly aligned with JD context. Highlight relevant projects and keywords.")
    else:
        feedback_parts.append("Resume shows semantic relevance to JD requirements.")
    
    feedback = " ".join(feedback_parts)

    # ---- Return detailed report ----
    return {
        "score": final_pct,
        "verdict": verdict,
        "hard_match_pct": hard_pct,
        "semantic_pct": semantic_pct,
        "matched_must_have": matched_must_have,
        "matched_good_to_have": matched_good_to_have,
        "missing_must_have": missing_must_have,
        "missing_good_to_have": missing_good_to_have,
        "semantic_section_scores": semantic_section_scores,
        "feedback": feedback,
        "details": {
            "hard_details": hard,
            "semantic_score": semantic_score
        }
    }

# ---------- Example usage (comment/uncomment in production) ----------
if __name__ == "__main__":
    # small quick example (replace with parsed outputs)
    resume = {
        "skills": ["Python", "Pandas", "Machine Learning", "SQL"],
        "text": "Experienced in Python, Pandas, built ML models, and used SQL in projects."
    }
    jd = {
        "raw_text": "We need a Python engineer with experience in AWS and SQL. Preferred: Kubernetes",
        "must_have": ["Python", "SQL", "AWS"],
        "good_to_have": ["Kubernetes"]
    }
    res = evaluate_resume_detailed(resume, jd)
    print(res)
