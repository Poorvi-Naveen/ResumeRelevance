
"""
Person 1: Resume & JD parsing utilities
File: person1_resume_parsing.py

Functions provided (public API):
- parse_resume(file_path) -> dict
- parse_jd(file_path) -> dict

This module implements robust extraction from PDF / DOCX / TXT using PyMuPDF and docx2txt,
section segmentation by heuristics, and lightweight skill/education/experience parsing.

Run this file directly to run a quick self-test using built-in sample text.

Note: For scanned PDFs (images-only) you'll need OCR (pytesseract + Tesseract). This file
detects very short extracted text and will warn if OCR might be required.

"""

import fitz  # PyMuPDF
import docx2txt
import re
import os
import unicodedata
import string
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Lightweight stopwords (keeps dependency light) ----
STOPWORDS = set([
    'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'with', 'for', 'to', 'from',
    'by', 'as', 'at', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
    'this', 'that', 'these', 'those', 'experience', 'years', 'year', 'worked'
])

# common heading synonyms mapped to normalized section keys
SECTION_SYNONYMS = {
    'skills': ['skills', 'technical skills', 'skillset', 'technical skillset', 'technical summary', 'technologies', 'technical expertise'],
    'education': ['education', 'academic qualifications', 'qualifications'],
    'experience': ['experience', 'work experience', 'professional experience', 'employment history'],
    'projects': ['projects', 'key projects', 'personal projects', 'project experience'],
    'certifications': ['certifications', 'licenses', 'certificates'],
    'summary': ['summary', 'profile', 'professional summary', 'objective'],
    'contact': ['contact', 'contact info', 'personal info', 'address', 'phone'],
}

# Regex patterns for common entities
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
LINKEDIN_RE = re.compile(r'linkedin\.com/in/[\w\d_-]+')
GITHUB_RE = re.compile(r'github\.com/[\w\d_-]+')
DEGREE_RE = re.compile(r'(Bsc|B\.?S\.?|Msc|M\.?S\.?|Ph\.?D\.?|B\.?Tech|B\.?E|M\.?Tech|M\.?E)\.?,?\s+in\s+([\w\s]+)', re.I)

# Fallback tech keywords for keyword matching
TECH_KEYWORDS = set([
    'python', 'java', 'c++', 'c#', 'javascript', 'html', 'css', 'sql', 'mysql',
    'postgresql', 'mongodb', 'react', 'angular', 'vue', 'nodejs', 'express',
    'flask', 'django', 'spring', 'aws', 'gcp', 'azure', 'docker', 'kubernetes',
    'terraform', 'ansible', 'jenkins', 'git', 'github', 'gitlab', 'agile',
    'scrum', 'data science', 'machine learning', 'deep learning', 'nlp',
    'computer vision', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'tableau', 'power bi', 'spark', 'hadoop', 'kafka', 'api', 'rest', 'graphql',
    'devops', 'ci/cd', 'linux', 'unix', 'shell scripting', 'cloud computing'
])

def _clean_text(text: str) -> str:
    """Removes weird characters and normalizes whitespace."""
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _extract_text_from_pdf(file_stream) -> str:
    """Extracts text from a PDF file stream."""
    text = ""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
    return _clean_text(text)

def _extract_text_from_docx(file_stream) -> str:
    """Extracts text from a DOCX file stream."""
    try:
        text = docx2txt.process(file_stream)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""
    return _clean_text(text)

def _find_sections(text: str) -> Dict[str, str]:
    """Tries to segment resume into sections based on headings."""
    sections = defaultdict(str)
    # Simple regex to find headings (e.g., all caps, followed by newline)
    section_titles = '|'.join(syn for syns in SECTION_SYNONYMS.values() for syn in syns)
    # This regex looks for common headers, case-insensitive, with optional punctuation.
    pattern = re.compile(f'({section_titles})[\\s\\n\\t]*(?:\\n|$)')
    
    matches = list(pattern.finditer(text))
    if not matches:
        return {'raw': text} # Fallback to raw text if no sections are found
        
    start_indices = [m.start() for m in matches]
    end_indices = start_indices[1:] + [len(text)]
    
    for i, m in enumerate(matches):
        title = m.group(1).lower().strip()
        # Find the normalized key
        normalized_key = next((key for key, synonyms in SECTION_SYNONYMS.items() if title in synonyms), title)
        
        content_start = m.end()
        content_end = end_indices[i]
        content = text[content_start:content_end].strip()
        
        sections[normalized_key] += " " + content
    
    sections = {k: _clean_text(v) for k, v in sections.items()}
    sections['raw'] = text # Always keep a copy of the raw text
    return sections

def _extract_skills_from_text(text: str) -> List[str]:
    """Extracts skills using a basic keyword matching method."""
    found_skills = set()
    for keyword in TECH_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.I):
            found_skills.add(keyword)
    return list(found_skills)

def _fallback_skill_search(text: str) -> List[str]:
    """A general search over the whole text for tech keywords."""
    found = set()
    words = text.lower().split()
    for keyword in TECH_KEYWORDS:
        if keyword in words:
            found.add(keyword)
    return list(found)

def parse_resume(file_stream) -> Dict[str, any]:
    """Parses a resume file and returns a structured dictionary."""
    filename = file_stream.filename
    text = ""
    if filename.endswith('.pdf'):
        text = _extract_text_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        text = _extract_text_from_docx(file_stream)
    elif filename.endswith('.txt'):
        text = _clean_text(file_stream.read().decode('utf-8'))
    else:
        raise ValueError("Unsupported file type")

    if not text:
        logger.warning(f"Could not extract text from {filename}. File might be a scanned image.")
        return {}

    # Find sections first
    sections = _find_sections(text)
    
    # Extract info from raw text if sections are not clear
    skills = _extract_skills_from_text(sections.get('skills', sections['raw']))
    qualifications = []
    
    # Fallback to search raw text if skills section is empty
    if not skills:
        skills = _extract_skills_from_text(sections['raw'])

    # Find email, phone, and links from the raw text
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    linkedin = LINKEDIN_RE.search(text)
    github = GITHUB_RE.search(text)

    # Extract degrees
    for m in DEGREE_RE.finditer(text):
        qualifications.append(m.group(0))

    return {
        'filename': filename,
        'text': text,
        'sections': sections,
        'skills': skills,
        'qualifications': qualifications,
        'contact': {
            'email': email.group(0) if email else None,
            'phone': phone.group(0) if phone else None,
            'linkedin': linkedin.group(0) if linkedin else None,
            'github': github.group(0) if github else None,
        }
    }

def parse_jd(file_stream) -> Dict[str, any]:
    """Parses a job description file and returns a structured dictionary."""
    filename = file_stream.filename
    text = ""
    if filename.endswith('.pdf'):
        text = _extract_text_from_pdf(file_stream)
    elif filename.endswith('.docx'):
        text = _extract_text_from_docx(file_stream)
    elif filename.endswith('.txt'):
        text = _clean_text(file_stream.read().decode('utf-8'))
    else:
        raise ValueError("Unsupported file type")
    
    if not text:
        logger.warning(f"Could not extract text from {filename}. File might be a scanned image.")
        return {'raw_text': '', 'must_have': [], 'good_to_have': [], 'job_role': 'Unknown', 'location': 'Unknown'}

    # Use a dictionary to store extracted sections
    sections = defaultdict(str)
    
    # ---- 1. Try to extract Job Role using more specific patterns ----
    job_role_patterns = [
        re.compile(r'^\s*\d+\.\s*(.*?)(?:\n|$)', re.MULTILINE | re.I), # e.g., "1. Data Scientist"
        re.compile(r'Job Title[:\s]*([^\n]+)', re.I), # e.g., "Job Title: Data Scientist"
        re.compile(r'Role Overview[:\s]*([^\n]+)', re.I), # e.g., "Role Overview: Data Scientist"
        re.compile(r'Role:\s*([^\n]+)', re.I), # e.g., "Role: Data Scientist"
        re.compile(r'Position:\s*([^\n]+)', re.I), # e.g., "Position: Data Scientist"
    ]
    job_role = 'Unknown'
    for pattern in job_role_patterns:
        match = pattern.search(text)
        if match:
            job_role = _clean_text(match.group(1)).strip()
            # If the role contains "interns" or other plural, make it singular
            if job_role.lower().endswith('interns'):
                job_role = job_role[:-1]
            break

    # ---- 2. Try to extract Location using more specific patterns ----
    location_patterns = [
        re.compile(r'Location:\s*([^\n(]+)(?:\s*\(Onsite\))?', re.I), # e.g., "Location: Pune (Onsite)"
        re.compile(r'Location:\s*([^\n]+)', re.I), # e.g., "Location: Pune"
        re.compile(r'based in\s*([^\n]+)', re.I) # e.g., "based in Pune"
    ]
    location = 'Unknown'
    for pattern in location_patterns:
        match = pattern.search(text)
        if match:
            location = _clean_text(match.group(1)).strip()
            break

    # ---- 3. Extract skills using a hybrid approach ----
    must_have, good_to_have = [], []
    
    # Check for bulleted lists under "Requirements" or "Skills"
    sections_text = {}
    sections_re = {
        'must_have_skills': re.compile(r'(?:requirements|must have|required skills|key responsibilities)[:\s\n]+([\s\S]{1,500})(?:\n\n|$)', re.I),
        'good_to_have_skills': re.compile(r'(?:preferred|nice to have|good to have|skills)[:\s\n]+([\s\S]{1,500})(?:\n\n|$)', re.I)
    }

    for key, pattern in sections_re.items():
        match = pattern.search(text)
        if match:
            # Clean up the text by splitting on bullet points and then cleaning up the words
            section_content = match.group(1)
            bullet_points = re.split(r'[\n\-\*]', section_content)
            skills = []
            for item in bullet_points:
                item = _clean_text(item).lower()
                # Find keywords in each item
                for keyword in TECH_KEYWORDS:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', item):
                        skills.append(keyword)
            sections_text[key] = list(set(skills)) # Use set to remove duplicates

    must_have = sections_text.get('must_have_skills', [])
    good_to_have = sections_text.get('good_to_have_skills', [])

    # General fallback: scan the entire text for tech keywords
    if not must_have and not good_to_have:
        all_found = _fallback_skill_search(text)
        # If skills were not found in sections, split them between must-have and good-to-have
        # This is a simple heuristic; in a real-world app, you'd use a more sophisticated model.
        must_have = all_found
        
    return {
        'raw_text': text,
        'job_role': job_role,
        'location': location,
        'must_have': list(set(must_have)), # Ensure no duplicates
        'good_to_have': list(set(good_to_have))
    }

# Quick self-test (remove in production)
if __name__ == '__main__':
    print("--- Running self-test for JD parsing ---")
    
    sample_jd_text = """
    Detailed Job Descriptions for Walk-In Drive

    1. Data Science Interns

    Internship Duration: 6 months, followed by permanent employment based on
    performance

    Location: Pune (Onsite)

    2. Data Engineer Intern

    Internship Duration: 6 months

    Requirements:
    Formal training on Python & Spark
    No prior experience required; 2022 or earlier graduates preferred
    
    Job Responsibilities:
    Build scalable streaming data pipelines
    Write complex SQL queries to transform source data
    
    Skills:
    Exceptional programming skills in Python, Spark, Kafka, Pyspark, and C++
    Strong SQL and complex query writing skills

    """
    
    class MockFile:
        def __init__(self, content):
            self.content = content.encode('utf-8')
            self.filename = "mock.pdf"
        
        def read(self):
            return self.content
            
    mock_file = MockFile(sample_jd_text)
    
    jd_data = parse_jd(mock_file)
    print("Extracted JD Data:")
    print(f"  Job Role: {jd_data.get('job_role')}")
    print(f"  Location: {jd_data.get('location')}")
    print(f"  Must Have Skills: {jd_data.get('must_have')}")
    print(f"  Good to Have Skills: {jd_data.get('good_to_have')}")