# backend/database.py

import os
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Float, DateTime, func

# Get the database connection URL from an environment variable.
# This makes the code portable between local development and production on Render.
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

# Create a database engine. SQLAlchemy will manage connections for you.
engine = create_engine(DATABASE_URL)

# Define the table structure using SQLAlchemy's metadata
metadata = MetaData()
analysis_results_table = Table('analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('resume_filename', String, nullable=False),
    Column('jd_job_role', String),
    Column('jd_location', String),
    Column('relevance_score', Float),
    Column('resume_url', String),
    Column('jd_url', String),
    Column('timestamp', DateTime, server_default=func.now())
)

def create_tables():
    """
    Creates the necessary tables in the database if they don't already exist.
    """
    inspector = inspect(engine)
    if not inspector.has_table(analysis_results_table.name):
        print("Creating 'analysis_results' table...")
        metadata.create_all(engine)
        print("Table created.")
    else:
        print("Table 'analysis_results' already exists.")

def insert_result(resume_filename, jd_job_role, jd_location, relevance_score, resume_url, jd_url):
    """
    Inserts a new analysis result into the database.
    """
    with engine.connect() as conn:
        stmt = analysis_results_table.insert().values(
            resume_filename=resume_filename,
            jd_job_role=jd_job_role,
            jd_location=jd_location,
            relevance_score=relevance_score,
            resume_url=resume_url,
            jd_url=jd_url
        )
        conn.execute(stmt)
        conn.commit()
    print(f"Successfully inserted analysis result for {resume_filename} with score {relevance_score}")

def get_results():
    """
    Fetches all stored analysis results from the database.
    """
    with engine.connect() as conn:
        stmt = text("SELECT * FROM analysis_results ORDER BY timestamp DESC")
        result = conn.execute(stmt)
        rows = result.mappings().all() # .mappings() gets dict-like rows
        return [dict(row) for row in rows]

def search_results(query, filter_by='job_role'):
    """
    Searches results based on a query and a filter column.
    """
    # Basic validation to prevent SQL injection on the column name
    allowed_filters = ['jd_job_role', 'resume_filename', 'jd_location']
    if filter_by not in allowed_filters:
        raise ValueError(f"Invalid filter column: {filter_by}")

    with engine.connect() as conn:
        # Using text() for the query to safely bind parameters
        stmt = text(f"SELECT * FROM analysis_results WHERE {filter_by} LIKE :query ORDER BY timestamp DESC")
        result = conn.execute(stmt, {"query": f"%{query}%"})
        rows = result.mappings().all()
        return [dict(row) for row in rows]