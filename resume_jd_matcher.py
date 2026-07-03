"""
Resume ↔ Job Description Matcher — built with Cohere

What this does:
  1. Takes your resume text and a job description.
  2. Uses Cohere's `embed` model to turn both into vector embeddings.
  3. Computes semantic similarity to score how well they match.
  4. Uses Cohere's `chat` (Command) model to explain WHY, and suggest
     which of your real skills to foreground for that specific JD.

Setup (takes ~5 minutes):
  1. Get a free API key: https://dashboard.cohere.com/api-keys
  2. pip install cohere
  3. Set your key as an environment variable:
       export CO_API_KEY="your-key-here"      (Mac/Linux)
       setx CO_API_KEY "your-key-here"         (Windows)
  4. Run: python resume_jd_matcher.py

This is a real, working tool — a good one to link in applications
that ask "have you built anything with Cohere?"
"""

import os
import cohere
import numpy as np

co = cohere.Client(os.environ.get("CO_API_KEY", ""))

RESUME_TEXT = """
Final-year B.Tech CSE (AI/ML) student. Built and deployed a full-stack
diabetes risk prediction app (Python, Flask, scikit-learn, React) with a
REST API in production. Built a secure authentication system (Node.js,
Express, JWT, bcrypt, RBAC, MongoDB) also live in production. Strong in
Python, JavaScript, SQL, C, Git, REST API design, and applied NLP
(SpaCy, NLTK). Currently building skills in Transformers, distributed
training, and large-scale ML.
"""

JOB_DESCRIPTION = """
Machine Learning Intern — proficiency in Python and ML frameworks
(TensorFlow, JAX). Experience with large-scale distributed training.
Familiarity with autoregressive sequence models such as Transformers.
Demonstrated passion for applied NLP models and products.
"""


def embed_text(text: str) -> np.ndarray:
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document",
    )
    return np.array(response.embeddings[0])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def explain_match(resume: str, jd: str, score: float) -> str:
    prompt = f"""You are a career coach. Given this resume and job description,
in 4-5 sentences: (1) explain the genuine overlap in plain language,
(2) name the real gaps honestly, (3) suggest which existing resume
points to lead with for this JD. Do not invent skills the resume
doesn't support.

Resume:
{resume}

Job Description:
{jd}

Similarity score: {score:.2f} (0=no overlap, 1=perfect overlap)
"""
    response = co.chat(model="command-a-03-2025", message=prompt)
    return response.text


if __name__ == "__main__":
    resume_vec = embed_text(RESUME_TEXT)
    jd_vec = embed_text(JOB_DESCRIPTION)

    score = cosine_similarity(resume_vec, jd_vec)
    print(f"\nSemantic match score: {score:.3f}\n")

    explanation = explain_match(RESUME_TEXT, JOB_DESCRIPTION, score)
    print("Cohere's analysis:\n")
    print(explanation)
