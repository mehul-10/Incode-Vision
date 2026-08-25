import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# MODEL (cached — loaded once, reused across calls)
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def load_model():
    """
    Load the pre-trained Sentence Transformer model.
    Cached at module level so repeated calls don't reload it.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_similarity(
    job_description,
    resume_text
):
    """
    Calculate semantic similarity between a job description
    and a resume.
    """

    model = load_model()

    embeddings = model.encode(
        [
            job_description,
            resume_text
        ]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    score = float(similarity * 100)

    return round(score, 2)


# ============================================================
# SKILL DATABASE
#
# Organized by category. Each canonical skill maps to a list
# of aliases/synonyms that should also be recognized in text.
# The canonical name (first checked) is what gets reported.
# ============================================================

SKILLS_DB = {

    "Languages": {
        "python": ["python"],
        "java": ["java"],
        "c++": ["c++", "cpp"],
        "c": ["c"],
        "c#": ["c#", "csharp", "c sharp"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript", "ts"],
        "go": ["golang", "go"],
        "rust": ["rust"],
        "r": ["r programming", "r language"],
    },

    "Web": {
        "html": ["html", "html5"],
        "css": ["css", "css3"],
        "react": ["react", "react.js", "reactjs"],
        "angular": ["angular", "angular.js", "angularjs"],
        "vue": ["vue", "vue.js", "vuejs"],
        "node.js": ["node.js", "nodejs", "node"],
        "express": ["express", "express.js", "expressjs"],
        "next.js": ["next.js", "nextjs"],
        "rest api": ["rest api", "restful api", "rest"],
        "graphql": ["graphql"],
    },

    "Databases": {
        "sql": ["sql"],
        "mysql": ["mysql"],
        "postgresql": ["postgresql", "postgres"],
        "mongodb": ["mongodb", "mongo"],
        "redis": ["redis"],
        "sqlite": ["sqlite"],
    },

    "AI / ML / Data": {
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning", "dl"],
        "artificial intelligence": ["artificial intelligence", "ai"],
        "natural language processing": ["natural language processing", "nlp"],
        "data science": ["data science"],
        "computer vision": ["computer vision", "cv"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch", "torch"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "keras": ["keras"],
    },

    "Frameworks / Backend": {
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi", "fast api"],
        "spring": ["spring", "spring boot"],
    },

    "Cloud / DevOps": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "google cloud": ["google cloud", "gcp"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "ci/cd": ["ci/cd", "ci cd", "continuous integration"],
        "terraform": ["terraform"],
        "linux": ["linux"],
    },

    "Tools": {
        "git": ["git"],
        "github": ["github"],
        "jira": ["jira"],
        "postman": ["postman"],
    },
}


def _build_alias_index():
    """
    Flatten SKILLS_DB into a list of
    (canonical_skill, category, alias, compiled_regex_pattern)
    tuples, sorted so longer/multi-word aliases are checked
    before shorter ones (avoids short aliases masking longer
    matches).
    """

    entries = []

    for category, skills in SKILLS_DB.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                # Word-boundary-safe pattern: alias must not be
                # immediately preceded/followed by an alnum char.
                # Using lookaround (not \b) so it works correctly
                # for aliases containing symbols like "c++" or "c#".
                # Excludes alnum AND +/# so that e.g. "c" doesn't
                # falsely match inside "c++" or "c#" (and vice versa).
                pattern = re.compile(
                    r"(?<![a-zA-Z0-9+#])"
                    + re.escape(alias)
                    + r"(?![a-zA-Z0-9+#])",
                    re.IGNORECASE
                )
                entries.append((canonical, category, alias, pattern))

    # Longest alias first, so "machine learning" is tested
    # before something that could partially overlap it.
    entries.sort(key=lambda e: len(e[2]), reverse=True)

    return entries


_ALIAS_INDEX = _build_alias_index()

# Kept for backwards compatibility with any code importing SKILLS
SKILLS = sorted({
    canonical
    for skills in SKILLS_DB.values()
    for canonical in skills
})


# ============================================================
# SKILL EXTRACTION
# ============================================================

def extract_skills(text):
    """
    Extract known skills from text using word-boundary-safe
    matching (avoids false positives like "ai" inside "email",
    or "c" inside "container").

    Returns a sorted list of canonical skill names.
    """

    found_skills = set()

    for canonical, _category, _alias, pattern in _ALIAS_INDEX:
        if canonical in found_skills:
            continue

        if pattern.search(text):
            found_skills.add(canonical)

    return sorted(found_skills)


def extract_skills_by_category(text):
    """
    Same as extract_skills, but grouped by category.
    Useful for dashboard-style breakdowns in the UI.
    """

    found = set(extract_skills(text))

    return categorize_skills(found)


def categorize_skills(skill_list):
    """
    Group an arbitrary list/set of canonical skill names by
    their category in SKILLS_DB. Useful for grouping the
    "matched" or "missing" skill lists for dashboard display,
    not just raw extracted text.
    """

    skill_set = set(skill_list)

    grouped = {}

    for category, skills in SKILLS_DB.items():
        matched = sorted(s for s in skills if s in skill_set)

        if matched:
            grouped[category] = matched

    return grouped


# ============================================================
# SKILL MATCHING
# ============================================================

def analyze_skills(
    job_description,
    resume_text
):
    """
    Compare skills required in the job description
    with skills found in the resume.
    """

    required_skills = set(
        extract_skills(
            job_description
        )
    )

    candidate_skills = set(
        extract_skills(
            resume_text
        )
    )

    matched_skills = required_skills.intersection(
        candidate_skills
    )

    missing_skills = required_skills.difference(
        candidate_skills
    )

    if required_skills:

        skill_match_percentage = (
            len(matched_skills)
            /
            len(required_skills)
        ) * 100

    else:

        skill_match_percentage = 0

    return {

        "required_skills":
            sorted(required_skills),

        "candidate_skills":
            sorted(candidate_skills),

        "matched_skills":
            sorted(matched_skills),

        "missing_skills":
            sorted(missing_skills),

        "missing_skills_by_category":
            categorize_skills(missing_skills),

        "matched_skills_by_category":
            categorize_skills(matched_skills),

        "skill_match_percentage":
            round(
                skill_match_percentage,
                2
            )
    }


# ============================================================
# FINAL RESUME ANALYSIS
# ============================================================

def analyze_resume(
    job_description,
    resume_text,
    semantic_weight=0.7,
    skill_weight=0.3
):
    """
    Perform complete resume analysis.

    semantic_weight / skill_weight let callers tune how much
    each component contributes to the final score (must sum
    to 1.0 for the score to stay on a 0-100 scale).
    """

    semantic_score = calculate_similarity(
        job_description,
        resume_text
    )

    skill_analysis = analyze_skills(
        job_description,
        resume_text
    )

    skill_score = (
        skill_analysis[
            "skill_match_percentage"
        ]
    )

    final_score = (
        semantic_score * semantic_weight
        +
        skill_score * skill_weight
    )

    return {

        "semantic_score":
            semantic_score,

        "skill_score":
            skill_score,

        "final_score":
            round(
                final_score,
                2
            ),

        "required_skills":
            skill_analysis[
                "required_skills"
            ],

        "candidate_skills":
            skill_analysis[
                "candidate_skills"
            ],

        "matched_skills":
            skill_analysis[
                "matched_skills"
            ],

        "missing_skills":
            skill_analysis[
                "missing_skills"
            ],

        "missing_skills_by_category":
            skill_analysis[
                "missing_skills_by_category"
            ],

        "matched_skills_by_category":
            skill_analysis[
                "matched_skills_by_category"
            ],

        "candidate_skills_by_category":
            extract_skills_by_category(
                resume_text
            ),
    }