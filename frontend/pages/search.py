import streamlit as st
import pandas as pd
import sqlite3

import os

# Get absolute path to the backend folder
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', 'resume_analysis_results.db'))  # correct DB file

st.set_page_config(page_title="Resume Database Search", layout="wide")
st.title("📂 Resume Database Search & Filter")

# --- Load data from SQLite ---
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM analysis_results", conn) # correct table
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # --- Sidebar filters ---
    with st.sidebar:
        st.header("🔍 Filters")

        # Job Role filter
        job_roles = df["jd_job_role"].dropna().unique().tolist()
        selected_roles = st.multiselect("Job Role", job_roles, default=job_roles)

        # Score range filter (use relevance_score column)
        score_min, score_max = st.slider("Score Range", 0, 100, (50, 100))

        # Location filter
        locations = df["jd_location"].dropna().unique().tolist()
        selected_locations = st.multiselect("Location", locations, default=locations)

        st.markdown("---")

        # --- Styled Back Button in Sidebar ---
        st.markdown("""
            <style>
            .back-button {
                display: inline-block;
                width: 100%;
                padding: 0.6em 1.2em;
                margin-top: 0.5em;
                font-size: 1.1em;
                font-weight: 600;
                text-align: center;
                color: white;
                background-color: #1565C0; /* blue */
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
            }
            .back-button:hover {
                background-color: #0D47A1;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(
            '<a href="/" target="_self" class="back-button">⬅ Back to Dashboard</a>',
            unsafe_allow_html=True
        )

    # --- Apply filters ---
    filtered_df = df[
        (df["jd_job_role"].isin(selected_roles)) &
        (df["relevance_score"].between(score_min, score_max)) &
        (df["jd_location"].isin(selected_locations))
    ]

    st.success(f"Showing {len(filtered_df)} matching candidates")

    # Display table
    st.dataframe(filtered_df)
    # Candidate cards with resume links
    st.subheader("📋 Candidate Profiles")
    for _, row in filtered_df.iterrows():
        # Prepend the backend URL to the stored path
        resume_url = f"https://resumerelevancebackend.onrender.com{row['resume_url']}"
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
    st.info("No resumes found in the database yet.")


