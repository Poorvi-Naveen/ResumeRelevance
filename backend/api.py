# backend/api.py
from dotenv import load_dotenv
load_dotenv()
from .database import create_tables, insert_result, get_results
import os
import uuid
import traceback
import gc # <--- CHANGE 1: IMPORT THE GARBAGE COLLECTOR
from flask import Flask, Blueprint, request, jsonify, send_from_directory, current_app
from .parser import parse_resume, parse_jd
from .scoring import evaluate_resume_detailed
from .database import create_tables, insert_result
from .gen_ai import generate_ai_insights

# 1. Create the main Flask app object
app = Flask(__name__)

# 2. Set up app configurations
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 3. Call the function to create database tables on startup
create_tables()

# 4. Define the Blueprint for your API routes
api_bp = Blueprint('api', __name__)

# --- Define API routes on the Blueprint ---
@api_bp.route('/analyze', methods=['POST'])

def analyze():
    if 'jd' not in request.files or 'resume' not in request.files:
        return jsonify({"error": "Missing 'jd' or 'resume' file in the request"}), 400

    jd_file = request.files.get('jd')
    resume_file = request.files.get('resume')

    if not jd_file or not resume_file:
        return jsonify({"error": "One of the uploaded files is empty"}), 400

    try:
        resume_extension = os.path.splitext(resume_file.filename)[1]
        jd_extension = os.path.splitext(jd_file.filename)[1]
        resume_unique_filename = str(uuid.uuid4()) + resume_extension
        jd_unique_filename = str(uuid.uuid4()) + jd_extension

        resume_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], resume_unique_filename)
        jd_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], jd_unique_filename)
        
        resume_file.seek(0)
        resume_file.save(resume_filepath)
        jd_file.seek(0)
        jd_file.save(jd_filepath)
        
        resume_url = f"/uploads/{resume_unique_filename}"
        jd_url = f"/uploads/{jd_unique_filename}"

        resume_file.seek(0)
        jd_file.seek(0)
        
        resume_data = parse_resume(resume_file)
        jd_data = parse_jd(jd_file)

        # --- CHANGE 2: CLEAN UP MEMORY AFTER PARSING ---
        # We no longer need the raw file objects in memory, so we release them.
        del resume_file
        del jd_file
        gc.collect()
        # ---------------------------------------------

        result = evaluate_resume_detailed(resume_data, jd_data)
        genai_insights = generate_ai_insights(
            resume_text=resume_data.get('text', ''),
            jd_text=jd_data.get('raw_text', ''),
            missing_skills=result.get('missing_must_have', [])
        )

        result.update(genai_insights)
        result['resume_url'] = resume_url
        result['jd_url'] = jd_url
        
        resume_filename = getattr(resume_data, 'filename', 'Unknown')
        job_role = jd_data.get('job_role', 'Unknown')
        location = jd_data.get('location', 'Unknown')
        score = result.get('score', 0)

        insert_result(
            resume_filename=resume_filename, 
            jd_job_role=job_role, 
            jd_location=location, 
            relevance_score=score, 
            resume_url=resume_url, 
            jd_url=jd_url
        )
        
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred during analysis: {e}"
        print(traceback.format_exc())
        return jsonify({
            "error": error_message, "score": 0, "verdict": "Error",
            "feedback": "Could not process files due to an internal error."
        }), 500
# Add this new route inside backend/api.py, near your other routes

@api_bp.route('/results', methods=['GET'])
def get_all_results():
    try:
        # This calls the existing get_results() function from your database.py
        results = get_results() 
        return jsonify(results)
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Failed to fetch results from the database."}), 500
    
# 5. Register the Blueprint with the app
app.register_blueprint(api_bp, url_prefix='/api')

# 6. Add the other routes that were in your old app.py
@app.route('/')
def index():
    return "<h1>Resume Checker Backend API is running</h1>"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- This part is for local development only ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

