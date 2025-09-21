# backend/api.py
from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import traceback
from flask import Flask, Blueprint, request, jsonify, send_from_directory, current_app

# import functions
from .parser import parse_resume, parse_jd
from .scoring import evaluate_resume_detailed
from .database import create_tables, insert_result
from .gen_ai import generate_ai_insights

# import embedding model once
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # loaded once, ~120MB

# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
create_tables()

api_bp = Blueprint('api', __name__)

@api_bp.route('/analyze', methods=['POST'])
def analyze():
    if 'jd' not in request.files or 'resume' not in request.files:
        return jsonify({"error": "Missing 'jd' or 'resume' file in the request"}), 400

    jd_file = request.files.get('jd')
    resume_file = request.files.get('resume')

    if not jd_file or not resume_file:
        return jsonify({"error": "One of the uploaded files is empty"}), 400

    try:
        # --- save with unique names ---
        resume_extension = os.path.splitext(resume_file.filename)[1]
        jd_extension = os.path.splitext(jd_file.filename)[1]
        resume_unique_filename = str(uuid.uuid4()) + resume_extension
        jd_unique_filename = str(uuid.uuid4()) + jd_extension

        resume_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], resume_unique_filename)
        jd_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], jd_unique_filename)

        resume_file.seek(0); resume_file.save(resume_filepath)
        jd_file.seek(0); jd_file.save(jd_filepath)

        resume_url = f"/uploads/{resume_unique_filename}"
        jd_url = f"/uploads/{jd_unique_filename}"

        resume_data = parse_resume(resume_filepath)
        jd_data = parse_jd(jd_filepath)

        result = evaluate_resume_detailed(resume_data, jd_data, model=embedding_model)

        genai_insights = generate_ai_insights(
            resume_text=resume_data.get('text', ''),
            jd_text=jd_data.get('raw_text', ''),
            missing_skills=result.get('missing_must_have', [])
        )

        result.update(genai_insights)
        result['resume_url'] = resume_url
        result['jd_url'] = jd_url

        resume_filename = getattr(resume_file, 'filename', 'Unknown')
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

# Register API blueprint
app.register_blueprint(api_bp, url_prefix='/api')

@app.route('/')
def index():
    return "<h1>Resume Checker Backend API is running</h1>"

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
