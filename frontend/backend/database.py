import os
from sqlalchemy import create_engine, text, inspect, MetaData, Table, Column, Integer, String, Float, DateTime, func

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set.")

engine = create_engine(DATABASE_URL)

# Define table structure
metadata = MetaData()
analysis_results_table = Table('analysis_results', metadata,
    Column('id', Integer, primary_key=True),
    Column('resume_filename', String, nullable=False),
    Column('jd_job_role', String),
    Column('jd_location', String),
    Column('relevance_score', Float),
    Column('timestamp', DateTime, server_default=func.now())
)

def create_tables():
    inspector = inspect(engine)
    if not inspector.has_table(analysis_results_table.name):
        print("Creating 'analysis_results' table...")
        metadata.create_all(engine)
        print("Table created.")
    else:
        print("Table 'analysis_results' already exists.")

def insert_result(resume_filename, jd_job_role, jd_location, relevance_score):
    with engine.connect() as conn:
        stmt = analysis_results_table.insert().values(
            resume_filename=resume_filename,
            jd_job_role=jd_job_role,
            jd_location=jd_location,
            relevance_score=relevance_score
        )
        conn.execute(stmt)
        conn.commit()
    print(f"Inserted result for {resume_filename} with score {relevance_score}")

def get_results():
    with engine.connect() as conn:
        stmt = text("SELECT * FROM analysis_results ORDER BY timestamp DESC")
        result = conn.execute(stmt)
        return [dict(row) for row in result.mappings().all()]

def search_results(query, filter_by='job_role'):
    allowed_filters = ['jd_job_role', 'resume_filename', 'jd_location']
    if filter_by not in allowed_filters:
        raise ValueError(f"Invalid filter column: {filter_by}")

    with engine.connect() as conn:
        stmt = text(f"SELECT * FROM analysis_results WHERE {filter_by} LIKE :query ORDER BY timestamp DESC")
        result = conn.execute(stmt, {"query": f"%{query}%"})
        return [dict(row) for row in result.mappings().all()]
