import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please create a .env file with your key.")

def generate_ai_insights(resume_text: str, jd_text: str, missing_skills: list) -> dict:
    """
    Uses a GenAI model to generate both feedback and interview questions.
    Returns a dictionary containing the generated content.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR recruitment assistant for Innomatics Research Labs. Your goal is to analyze a resume against a job description and provide structured, insightful, and actionable content for both the student and the recruiter."),
            ("human", """
            Based on the provided Job Description and Resume, please perform the following two tasks:

            1.  **Generate Feedback:** Write a 3-4 sentence personalized feedback paragraph for the candidate. The tone should be encouraging. Highlight their strengths, mention the most critical missing skills for this role, and offer a concrete suggestion for improvement.
            2.  **Generate Interview Questions:** Create 3 targeted interview questions a recruiter could ask. Make them specific to the skills mentioned in both the resume and the JD.

            **Job Description:**
            {jd}

            **Candidate's Resume:**
            {resume}
            
            **Key Missing Skills to focus on:**
            {missing}

            Please structure your response clearly with "### FEEDBACK ###" and "### QUESTIONS ###" separators. List each question on a new line.
            """)
        ])
        
        chain = prompt_template | llm
        
        response = chain.invoke({
            "jd": jd_text,
            "resume": resume_text,
            "missing": ", ".join(missing_skills) if missing_skills else "None"
        })
        
        content = response.content
        
        # Parse the structured response from the LLM
        feedback = content.split("### QUESTIONS ###")[0].replace("### FEEDBACK ###", "").strip()
        questions_raw = content.split("### QUESTIONS ###")[1].strip()
        questions = [q.strip() for q in questions_raw.split('\n') if q.strip()]

        return {
            "feedback": feedback,
            "interview_questions": questions
        }

    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            print("Gemini API quota exceeded.")
            return {
                "feedback": "AI feedback could not be generated due to quota limits. Please try again tomorrow or upgrade your API quota.",
                "interview_questions": []
            }
        print(f"Error during GenAI insight generation: {e}")
        return {
            "feedback": "AI feedback could not be generated due to a server error.",
            "interview_questions": []
        }

