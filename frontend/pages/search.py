import streamlit as st
import pandas as pd
from backend.database import get_results, search_results

st.set_page_config(page_title="Search Results", layout="wide")

st.title("🔍 Search Previous Analysis Results")

# Search controls
search_type = st.radio("Search by:", ["Job Role", "Resume Name", "Location"], horizontal=True)
search_query = st.text_input("Enter search term:")

if st.button("Search"):
    if search_query:
        filter_map = {
            "Job Role": "jd_job_role",
            "Resume Name": "resume_filename",
            "Location": "jd_location"
        }
        results = search_results(search_query, filter_map[search_type])
    else:
        results = get_results()
else:
    results = get_results()

if results:
    df = pd.DataFrame(results)
    
    # Display results without file URLs
    for _, row in df.iterrows():
        with st.container():
            st.markdown(f"### {row['resume_filename']}")
            st.markdown(f"**Job Role:** {row['jd_job_role']} | **Location:** {row['jd_location']}")
            st.markdown(f"**Relevance Score:** {row['relevance_score']}%")
            st.markdown("---")
else:
    st.info("No results found. Try a different search term or analyze new resumes.")
