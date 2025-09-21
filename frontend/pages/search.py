# frontend/pages/search.py

import streamlit as st
import pandas as pd
import requests # Use requests to call the API

# Get the backend URL from Streamlit's secrets
API_URL = st.secrets["API_URL"]

st.set_page_config(page_title="Resume Database Search", layout="wide")
st.title("📂 Resume Database Search & Filter")

# --- Load data from the backend API ---
@st.cache_data
def load_data():
    try:
        # Call the new '/api/results' endpoint on your backend
        response = requests.get(f"{API_URL}/api/results")
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data from backend: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- Sidebar filters (this part remains the same) ---
    with st.sidebar:
        st.header("🔍 Filters")
        job_roles = df["jd_job_role"].dropna().unique().tolist()
        selected_roles = st.multiselect("Job Role", job_roles, default=job_roles)
        score_min, score_max = st.slider("Score Range", 0, 100, (50, 100))
        locations = df["jd_location"].dropna().unique().tolist()
        selected_locations = st.multiselect("Location", locations, default=locations)

    # --- Apply filters (this part remains the same) ---
    filtered_df = df[
        (df["jd_job_role"].isin(selected_roles)) &
        (df["relevance_score"].between(score_min, score_max)) &
        (df["jd_location"].isin(selected_locations))
    ]

    st.success(f"Showing {len(filtered_df)} matching candidates")
    st.dataframe(filtered_df)

    st.subheader("📋 Candidate Profiles")
    for _, row in filtered_df.iterrows():
        # --- IMPORTANT: Build the resume URL using the API_URL ---
        resume_url = f"{API_URL}{row['resume_url']}"
        st.markdown(f"""
        #### Candidate ID: {row['id']}
        - **Job Role**: {row['jd_job_role']}
        - **Score**: {row['relevance_score']}%
        - **Location**: {row['jd_location']}
        - [📂 Open Resume]({resume_url})
        - ⏰ Added on: {row['timestamp']}
        ---
        """)
else:
    st.info("No resumes found in the database yet, or there was an error connecting to the backend.")
