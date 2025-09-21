import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pages.search
import sqlite3
import os

# --- Page Setup ---
st.set_page_config(
    page_title="Resume Relevance Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .report-box {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper function to display candidate report ---
def display_candidate_report(row):
    with st.container():
        st.markdown(f"#### {row['name']}")
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score", f"{row.get('score', 0)}%")
        col2.metric("Verdict", row.get('verdict', 'N/A'))
        col3.metric("Keyword Match", f"{row.get('hard_match_pct', 'N/A')}%")
        col4.metric("Semantic Match", f"{row.get('semantic_pct', 'N/A')}%")

        with st.expander("View Detailed Analysis"):
            if row.get('verdict', '') == 'Error':
                st.error(f"Could not process resume. Feedback: {row.get('feedback', 'No details')}")
            else:
                st.info(f"**AI Feedback:** {row.get('feedback', 'No feedback available.')}")
                matched_must = row.get('matched_must_have', [])
                matched_good = row.get('matched_good_to_have', [])
                missing_must = row.get('missing_must_have', [])
                missing_good = row.get('missing_good_to_have', [])

                st.subheader("✅ Matched Skills")
                if matched_must: st.markdown(", ".join(matched_must))
                if matched_good: st.markdown(", ".join(matched_good))
                if not matched_must and not matched_good: st.write("No direct skill matches found.")

                st.subheader("❌ Missing Skills")
                if missing_must: st.markdown(", ".join(missing_must))
                if missing_good: st.markdown(", ".join(missing_good))
                if not missing_must and not missing_good: st.write("No missing skills detected.")


# --- Main App ---
st.title("📄 Automated Resume Relevance Checker")
st.markdown("Upload a Job Description and resumes to get AI-powered analysis.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Controls")
    jd_file = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])
    resume_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)
    
    st.header("🔍 Filter Results")
    score_threshold = st.slider("Minimum Score", 0, 100, 50)
    verdict_filter = st.multiselect("Filter by Verdict", ["High", "Medium", "Low"], default=["High","Medium"])

analyze_clicked = st.button("🔍 Analyze Resumes")
search_clicked = st.button("🔎 Search Previous Results")

# --- Analyze resumes ---
if analyze_clicked and jd_file and resume_files:
    st.warning("Backend processing logic needs to be added here if using Streamlit-only. Currently placeholder results will be shown.")
    # Example: You can loop over files, call your evaluation functions, and save to session_state
    results = []
    for file in resume_files:
        results.append({
            "name": file.name,
            "score": 75,
            "verdict": "High",
            "hard_match_pct": 70,
            "semantic_pct": 65,
            "matched_must_have": ["Python", "SQL"],
            "matched_good_to_have": ["Streamlit"],
            "missing_must_have": ["Docker"],
            "missing_good_to_have": ["Kubernetes"],
            "feedback": "Good match for skills and experience."
        })
    st.session_state["analysis_results"] = results

# --- Search previous results ---
if search_clicked:
    st.switch_page("pages/search.py")

# --- Display results ---
if "analysis_results" in st.session_state:
    df = pd.DataFrame(st.session_state["analysis_results"])
    df['shortlisted'] = (df["score"] >= score_threshold) & (df["verdict"].isin(verdict_filter))
    shortlisted_df = df[df['shortlisted']].sort_values("score", ascending=False)
    other_df = df[~df['shortlisted']].sort_values("score", ascending=False)

    st.header(f"✅ Shortlisted Candidates ({len(shortlisted_df)})")
    for _, row in shortlisted_df.iterrows(): display_candidate_report(row)

    st.header(f"❌ Other Candidates ({len(other_df)})")
    for _, row in other_df.iterrows(): display_candidate_report(row)
else:
    st.info("👈 Upload JD/resumes or search previous results.")
