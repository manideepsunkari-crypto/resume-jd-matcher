# Resume ↔ Job Description Matcher

A small tool that uses [Cohere](https://cohere.com)'s API to score how well a resume semantically matches a job description, then explains the match in plain language.

## What it does

1. Takes a resume and a job description as input text.
2. Uses Cohere's **`embed-english-v3.0`** model to convert both into vector embeddings.
3. Computes **cosine similarity** between the two vectors to produce a semantic match score (0 = no overlap, 1 = perfect overlap).
4. Sends both texts to Cohere's **`command-a-03-2025`** chat model, which explains the genuine overlap, honestly flags real gaps, and suggests which existing resume points to lead with for that specific job.

Unlike simple keyword matching, this compares the *meaning* of the resume and JD, not just whether the same words appear in both — so a resume that says "built REST APIs" can still match a JD asking for "backend service development" even without exact keyword overlap.

## Example output

```
Semantic match score: 0.501

Cohere's analysis:
[Model-generated explanation of overlap, gaps, and which
resume points to emphasize for this specific job]
```

## Setup

1. Get a free API key from the [Cohere Dashboard](https://dashboard.cohere.com/api-keys).
2. Install the dependencies:
   ```
   pip install cohere numpy
   ```
3. Set your API key as an environment variable:
   ```bash
   # Mac/Linux
   export CO_API_KEY="your-key-here"

   # Windows (PowerShell)
   $env:CO_API_KEY="your-key-here"

   # Windows (Command Prompt)
   set CO_API_KEY=your-key-here
   ```
4. Run it:
   ```
   python resume_jd_matcher.py
   ```

To test it against your own resume and a different job description, edit the `RESUME_TEXT` and `JOB_DESCRIPTION` variables at the top of the script.

## Why I built this

I wanted hands-on experience with Cohere's embedding and generation endpoints rather than just reading about them, and a resume/JD matcher felt like a genuinely useful tool to have while applying to internships — including this one.

## Tech used

- Cohere `embed-english-v3.0` — semantic embeddings
- Cohere `command-a-03-2025` — natural-language explanation
- Python, NumPy — cosine similarity calculation

## Notes

- Requires a Cohere API key (free tier works fine for this use case).
- Do not commit your API key to this repo — the script reads it from an environment variable (`CO_API_KEY`), which keeps it out of your source code.
