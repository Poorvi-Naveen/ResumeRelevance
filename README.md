# Automated Resume Relevance Checker

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)](https://streamlit.io/)  
[![Flask](https://img.shields.io/badge/Backend-Flask-black?logo=flask)](https://flask.palletsprojects.com/)  
[![Database](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)](https://www.sqlite.org/)  
[![AI](https://img.shields.io/badge/AI-Google%20Gemini-green?logo=google)](https://ai.google.dev/)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

---

An **AI-powered recruitment co-pilot** designed to automate, analyze, and enrich the resume evaluation process.

---

## The Problem
At organizations like **Innomatics Research Labs**, the manual process of sifting through thousands of resumes for a handful of job openings is a major bottleneck. Recruiters spend countless hours matching resumes to job descriptions (JDs), leading to:

- **Slow Turnaround**: Delays in shortlisting candidates for hiring companies.  
- **Inconsistency**: Subjective judgments from different evaluators.  
- **High Workload**: Placement staff are buried in administrative tasks instead of focusing on high-value activities like student mentorship and interview preparation.  

---

## Our Solution: Automated Resume Relevance Checker
**Automated Resume Relevance Checker** is a web-based platform that transforms resume evaluation from a manual chore into an intelligent, automated workflow.  

It serves as an **AI Co-pilot** for placement teams, providing not just scores, but **actionable, generative insights** to streamline the entire hiring pipeline.

Our system leverages a **hybrid AI approach**, combining traditional keyword and semantic matching with the power of **Large Language Models (LLMs)** to provide a holistic analysis of each candidate.

---

## Key Features

### Automated Hybrid Scoring
Calculates a precise **Relevance Score (0–100)** by combining:
- **Keyword Matching**: Identifies essential skills, tools, and qualifications.  
- **Semantic Matching**: Understands the contextual relevance of a candidate's experience and projects.  

### GenAI Candidate Co-pilot *(USP)*
Goes beyond scoring to provide **generative insights**:
- **Personalized Feedback**: Custom, encouraging feedback for each student, highlighting strengths and offering suggestions for resume improvement.  
- **Targeted Interview Questions**: Generates technical and behavioral questions based on the intersection of the resume and JD, saving recruiters valuable preparation time.  

### Interactive Dashboard
A user-friendly interface built with **Streamlit** that allows recruiters to:
- Upload a JD and hundreds of resumes in a single batch.  
- Filter and sort candidates by score, verdict, and name.  
- Visualize candidate comparisons with interactive charts.  

### Persistent Storage & File Access
- Results stored in a cloud **PostgreSQL database**.  
- Uploaded resumes and JDs saved for direct access and review.  

---

## Application Preview

### Main Dashboard & Comparison Chart
![Main Dashboard](<img width="1311" height="843" alt="image" src="https://github.com/user-attachments/assets/c853cb7f-99a0-4c2b-924f-9c9467d11abf" />
)

### Detailed Candidate Analysis with GenAI Insights 
![Detailed Analysis1](<img width="959" height="416" alt="image" src="https://github.com/user-attachments/assets/09eb5cc5-fddc-4610-a318-ca6481fbf375" />)
![Detailed Analysis2](<img width="1919" height="845" alt="image" src="https://github.com/user-attachments/assets/03c53f8f-8843-463b-a4b2-2a7f9172b5e9" />)

### Search section with resume evaluation history
![Detailed Analysis](<img width="1919" height="838" alt="image" src="https://github.com/user-attachments/assets/f1cf1bff-5f3e-4ac4-adc2-5aff094101be" />)

---

## Tech Stack

| Category       | Technology                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Backend**    | Python, Flask, PostgreSQL                                                       |
| **Frontend**   | Streamlit, Pandas, Plotly                                                   |
| **Core AI/ML** | PyMuPDF, python-docx (for parsing), Scikit-learn, Sentence-Transformers     |
| **Generative AI** | LangChain, Google Gemini API (gemini-2.0-flash)                          |

---

## System Architecture
<img width="3000" height="2000" alt="image" src="https://github.com/user-attachments/assets/5351b24b-b4e1-4536-9f7a-0a5fa456e3b9" />


- **Frontend (Streamlit)**: User interface  
- **Backend (Flask)**: File parsing, scoring, GenAI model calls, database interaction  
- **Database (PostgreSQL)**: Persistent storage for resumes, JDs, and evaluation results  

---

## Setup and Installation

### 1. Prerequisites
- Python **3.9+**  
- A virtual environment tool (like `venv`)  

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/ResumeRelevance.git
cd ResumeRelevance
```
### 3. Set Up Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
### 5. Configure Google AI API Key
- Obtain your free API key from Google AI Studio.
- In the root directory, create a file named .env.
- Add your API key:
```bash
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## Running the Application
- Automated Resume Relevance Checker requires two terminal sessions: one for backend, one for frontend.
Terminal 1: Start the Backend Server
```bash
# Activate environment if not already
python app.py
```
Runs Flask server on http://127.0.0.1:5000.
- Terminal 2: Start the Frontend Dashboard
```bash
# Open new terminal & activate environment
streamlit run dashboard.py
```

## License
This project is licensed under the MIT License.

## Acknowledgements
- Google AI Studio for providing Gemini API access
- Streamlit for interactive dashboards
- LangChain for orchestration of LLMs

## Contributions
My team mates !
-- Parnika Deepak Bhat (parnika0905@gmail.com)
-- Revathi R (revathimaniraj12@gmail.com)
