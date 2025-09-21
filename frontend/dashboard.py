import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import pages.search

# Import backend modules directly
from backend.parser import parse_resume, parse_jd
from backend.scoring import evaluate_resume_detailed
from backend.gen_ai import generate_ai_insights
from backend.database import insert_result, get_results, search_results

st.set_page_config(
    page_title="Resume Relevance Checker",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (unchanged) ---
st.markdown("""
<style>
    .st-emotion-cache-16txtl3 {padding-top: 2rem;}
    .st-emotion-cache-1y4p8pa {padding-top: 0;}
    .st-emotion-cache-1v0mbdj > img {border-radius: 0.5rem;}
    .st-emotion-cache-1kyxreq {justify-content: center;}
    .st-emotion-cache-1dp5vir {gap: 1rem;}
    .report-box {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper function (unchanged) ---
def display_candidate_report(row):
    # ... (keep existing implementation) ...

# --- Main App ---
    st.title("📄 Automated Resume Relevance Checker")
    st.markdown("Upload a Job Description and one or more resumes to get an AI-powered analysis of each candidate's suitability.")

# --- Sidebar for Uploads and Filters ---
with st.sidebar:
    st.header("⚙️ Controls")
    jd_file = st.file_uploader(
        "1. Upload Job Description",
        type=["txt", "pdf", "docx"],
        key="jd_uploader"
    )
    resume_files = st.file_uploader(
        "2. Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader"
    )
    
    st.header("🔍 Filter Results")
    score_threshold = st.slider(
        "Minimum Score",
        min_value=0, max_value=100, value=50, key="score_slider"
    )
    verdict_filter = st.multiselect(
        "Filter by Verdict",
        options=["High", "Medium", "Low"],
        default=["High", "Medium"], key="verdict_filter"
    )

# --- Main Processing and Display Logic ---
analyze_clicked = st.button("🔍 Analyze Resumes")
search_clicked = st.button("🔎 Search Previous Results")

results = []

if analyze_clicked and jd_file and resume_files:
    with st.spinner('Analyzing resumes... this may take a moment.'):
        for file in resume_files:
            try:
                # Reset file pointers
                jd_file.seek(0)
                file.seek(0)
                
                # Parse files directly
                jd_data = parse_jd(jd_file)
                resume_data = parse_resume(file)
                
                # Evaluate resume
                evaluation = evaluate_resume_detailed(resume_data, jd_data)
                
                # Generate AI insights
                insights = generate_ai_insights(
                    resume_text=resume_data.get('text', ''),
                    jd_text=jd_data.get('raw_text', ''),
                    missing_skills=evaluation.get('missing_must_have', [])
                )
                
                # Combine results
                result = evaluation
                result.update(insights)
                result['name'] = file.name
                
                # Store in database (without URLs)
                insert_result(
                    resume_filename=file.name,
                    jd_job_role=jd_data.get('job_role', 'Unknown'),
                    jd_location=jd_data.get('location', 'Unknown'),
                    relevance_score=result.get('score', 0),
                    resume_url=None,
                    jd_url=None
                )
                
                results.append(result)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
                results.append({
                    "name": file.name,
                    "score": 0,
                    "verdict": "Error",
                    "feedback": f"Processing failed: {str(e)}"
                })
        
        st.session_state["analysis_results"] = results

if search_clicked:
    st.switch_page("pages/search.py")

# --- Display results (unchanged) ---
if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    if results:
        # ... (keep existing display logic) ...
else:
    st.info("👈 **Get started by uploading a Job Description and resumes, or search previous results.**")
    st.image("https://i.imgur.com/tIO5UeA.png", caption="System Architecture Overview")

