import os
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph to build the stateful graph
from langgraph.graph import StateGraph, END

# You'll need to set your Google AI API key as an environment variable
# For a hackathon, you can also hardcode it here temporarily if needed
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

# --- 1. Define the State for our Graph ---
# This is the "memory" that flows through our application
class EvaluationState(TypedDict):
    resume_text: str
    jd_text: str
    resume_skills: List[str]
    jd_must_have: List[str]
    jd_good_to_have: List[str]
    
    # Scores will be populated as we go
    hard_match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    
    # The final generative output
    ai_feedback: str
    interview_questions: List[str]
    final_score: int
    final_verdict: str


# --- 2. Define the Nodes (Steps) in our Graph ---

def calculate_keyword_score(state: EvaluationState):
    """
    Node to calculate the traditional keyword-based score.
    This would contain the logic from your current scoring.py's `compute_hard_match`.
    """
    print("--- Running Keyword Score Node ---")
    # (For brevity, this is a simplified version of your scoring logic)
    resume_s = set(s.lower() for s in state['resume_skills'])
    jd_s = set(s.lower() for s in state['jd_must_have'])
    
    matched = list(resume_s.intersection(jd_s))
    missing = list(jd_s.difference(resume_s))
    
    score = 0
    if jd_s:
        score = round((len(matched) / len(jd_s)) * 100)

    state['hard_match_score'] = score
    state['matched_skills'] = matched
    state['missing_skills'] = missing
    
    return state

def generate_ai_insights(state: EvaluationState):
    """
    Node to call the LLM for generative feedback and interview questions.
    This is the core GenAI feature.
    """
    print("--- Running AI Insights Node ---")
    # Initialize the LLM - using Google's Gemini here
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert HR recruitment assistant. Your goal is to analyze a resume against a job description and provide structured, insightful feedback."),
        ("human", """
        Here is a resume and a job description. Please perform the following tasks:
        1.  Provide a 3-4 sentence personalized feedback paragraph for the candidate, highlighting their strengths and areas for improvement relevant to this specific job.
        2.  Generate 3 targeted interview questions (1 technical, 1 behavioral, 1 general) that a recruiter could ask this candidate based on their resume and the job requirements.

        **Job Description:**
        {jd}

        **Candidate's Resume:**
        {resume}

        Please provide the output in a structured format with clear headings for 'Feedback' and 'Interview Questions'.
        """)
    ])
    
    chain = prompt_template | llm
    
    llm_response = chain.invoke({
        "jd": state['jd_text'],
        "resume": state['resume_text']
    })
    
    # --- Simple parsing of the LLM's text response ---
    # In a real app, you'd use structured output (JSON) for more reliability
    response_text = llm_response.content
    
    try:
        feedback_section = response_text.split("Interview Questions:")[0].replace("Feedback:", "").strip()
        questions_section = response_text.split("Interview Questions:")[1].strip()
        questions = [q.strip() for q in questions_section.split('\n') if q.strip()]
    except IndexError:
        feedback_section = "Could not generate AI feedback."
        questions = ["Could not generate interview questions."]

    state['ai_feedback'] = feedback_section
    state['interview_questions'] = questions

    return state

def finalize_evaluation(state: EvaluationState):
    """
    Node to combine the scores and produce a final verdict.
    """
    print("--- Running Finalization Node ---")
    # A simple weighted average. You can make this more complex.
    # For now, let's just use the hard match score.
    final_score = int(state['hard_match_score'])
    
    verdict = "Low"
    if final_score >= 70:
        verdict = "High"
    elif final_score >= 40:
        verdict = "Medium"
        
    state['final_score'] = final_score
    state['final_verdict'] = verdict
    
    return state


# --- 3. Wire up the Nodes into a Graph ---
workflow = StateGraph(EvaluationState)

workflow.add_node("keyword_scorer", calculate_keyword_score)
workflow.add_node("ai_analyzer", generate_ai_insights)
workflow.add_node("finalizer", finalize_evaluation)

# Define the sequence of execution
workflow.add_edge("keyword_scorer", "ai_analyzer")
workflow.add_edge("ai_analyzer", "finalizer")

# Set the entry point and compile the graph
workflow.set_entry_point("keyword_scorer")
app = workflow.compile()


# This function will be called by your API
def run_evaluation_graph(resume_data, jd_data):
    initial_state = {
        "resume_text": resume_data['text'],
        "jd_text": jd_data['raw_text'],
        "resume_skills": resume_data['skills'],
        "jd_must_have": jd_data['must_have'],
        "jd_good_to_have": jd_data.get('good_to_have', [])
    }
    
    # The END state is the final output after all nodes have run
    final_state = app.invoke(initial_state)
    
    # Reformat for frontend
    return {
        "score": final_state['final_score'],
        "verdict": final_state['final_verdict'],
        "hard_match_pct": final_state['hard_match_score'],
        "semantic_pct": None, # You can integrate your semantic model here if desired
        "matched_must_have": final_state['matched_skills'],
        "missing_must_have": final_state['missing_skills'],
        "feedback": final_state['ai_feedback'],
        "interview_questions": final_state['interview_questions'],
        # Add other fields as needed
    }

