from matcher import analyze_resume


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = """
We are looking for a Python Developer with experience
in Machine Learning and Natural Language Processing.

The candidate should have knowledge of:

Python
Machine Learning
NLP
Scikit-learn
Pandas
NumPy
SQL
Git

Experience with FastAPI and Docker is preferred.
"""


# ============================================================
# SAMPLE RESUME
# ============================================================

resume_text = """
Computer Science student with experience in Python,
Machine Learning and Artificial Intelligence.

Built multiple projects using Scikit-learn, Pandas
and NumPy.

Developed REST APIs using FastAPI and worked with
SQL databases.

Experienced with Git and GitHub.
"""


# ============================================================
# ANALYSIS
# ============================================================

result = analyze_resume(
    job_description,
    resume_text
)


print("\nRESUME ANALYSIS")
print("=" * 50)

print(
    f"\nSemantic Similarity: "
    f"{result['semantic_score']}%"
)

print(
    f"Skill Match: "
    f"{result['skill_score']}%"
)

print(
    f"\nFINAL MATCH SCORE: "
    f"{result['final_score']}%"
)


print("\nMatched Skills:")

for skill in result["matched_skills"]:

    print(
        f"✓ {skill}"
    )


print("\nMissing Skills:")

for skill in result["missing_skills"]:

    print(
        f"✗ {skill}"
    )