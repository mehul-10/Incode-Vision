import html
import re

import streamlit as st
import pandas as pd

from resume_parser import extract_resume_text
from matcher import analyze_resume


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ResumeAI | Smart Candidate Matching",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

ACCENT = "#2563eb"
ACCENT_SOFT = "#eff6ff"
INK = "#111827"
MUTED = "#64748b"
BORDER = "#e5e7eb"
SUCCESS = "#16a34a"
SUCCESS_SOFT = "#f0fdf4"
BG = "#f8fafc"


# ============================================================
# HTML RENDER HELPER
#
# st.markdown(unsafe_allow_html=True) misrenders multi-line,
# indented HTML strings — CommonMark treats 4+ leading spaces
# as an INDENTED CODE BLOCK, so the raw tags get shown as text
# instead of being parsed as HTML. Collapsing every string to
# a single line (no leading whitespace, no embedded newlines)
# avoids that entirely.
# ============================================================

def render(template):
    single_line = re.sub(r"\s*\n\s*", " ", template).strip()
    st.markdown(single_line, unsafe_allow_html=True)


def esc(text):
    """Escape user-controlled text before injecting into HTML."""
    return html.escape(str(text))


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


# ============================================================
# CUSTOM CSS
#
# Scoped to layout/typography/cards only — NOT to <button>
# elements. Overriding button styles globally previously bled
# into Streamlit's internal widgets (file uploader's browse/
# remove buttons), turning their icons invisible. The accent
# color on our own "Analyze" button comes from
# .streamlit/config.toml + type="primary" instead, which only
# touches buttons we explicitly mark as primary.
# ============================================================

render(
    f"""
    <style>

    .stApp {{
        background: {BG};
    }}

    .block-container {{
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    header {{
        background: transparent;
    }}

    .navbar {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 16px 24px;
        margin-bottom: 45px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}

    .logo {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {INK};
    }}

    .logo span {{
        color: {ACCENT};
    }}

    .status {{
        font-size: 0.85rem;
        color: {SUCCESS};
        background: {SUCCESS_SOFT};
        padding: 7px 12px;
        border-radius: 20px;
        border: 1px solid #bbf7d0;
    }}

    .hero {{
        text-align: center;
        max-width: 750px;
        margin: auto;
        margin-bottom: 45px;
    }}

    .hero h1 {{
        font-size: 3rem;
        font-weight: 750;
        color: {INK};
        letter-spacing: -1.5px;
        margin-bottom: 12px;
    }}

    .hero h1 span {{
        color: {ACCENT};
    }}

    .hero p {{
        color: {MUTED};
        font-size: 1.05rem;
        line-height: 1.7;
    }}

    .section-label {{
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: {MUTED};
        margin-bottom: 10px;
    }}

    div[data-testid="stFileUploader"] section {{
        border-radius: 14px;
        border: 1.5px dashed {BORDER};
        background: white;
    }}

    .stTextArea textarea {{
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}

    .results-header {{
        margin-top: 55px;
        margin-bottom: 25px;
    }}

    .results-header h2 {{
        color: {INK};
        margin-bottom: 2px;
    }}

    .results-header p {{
        color: {MUTED};
        margin-top: 0;
    }}

    .candidate-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 16px;
    }}

    .top-card {{
        border: 1.5px solid {ACCENT};
        box-shadow: 0 10px 26px rgba(37, 99, 235, 0.14);
    }}

    .rank-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 30px;
        height: 30px;
        border-radius: 9px;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 10px;
    }}

    .top-badge {{
        background: {ACCENT};
        color: white;
    }}

    .top-pill {{
        display: inline-block;
        margin-left: 10px;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        vertical-align: middle;
    }}

    .candidate-name {{
        font-size: 1.1rem;
        font-weight: 650;
        color: {INK};
        display: inline-flex;
        align-items: center;
    }}

    .score {{
        font-size: 2rem;
        font-weight: 750;
        color: {ACCENT};
        line-height: 1;
    }}

    .score-label {{
        color: {MUTED};
        font-size: 0.8rem;
    }}

    .progress-bg {{
        width: 100%;
        background: {BORDER};
        border-radius: 20px;
        height: 8px;
        margin-top: 12px;
        overflow: hidden;
    }}

    .progress-fill {{
        background: {ACCENT};
        height: 100%;
        border-radius: 20px;
    }}

    .skill-tag {{
        display: inline-block;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        border-radius: 20px;
        padding: 5px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.8rem;
    }}

    .missing-tag {{
        display: inline-block;
        background: {BG};
        color: {MUTED};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 5px 10px;
        margin: 3px 4px 3px 0;
        font-size: 0.8rem;
    }}

    .skill-category-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }}

    .footer {{
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 60px;
        padding-top: 25px;
        border-top: 1px solid {BORDER};
    }}

    </style>
    """
)


# ============================================================
# SKILL RENDER HELPERS
# ============================================================

def render_skill_tags(skills, css_class):
    if not skills:
        return ""
    return "".join(
        f'<span class="{css_class}">{esc(skill)}</span>'
        for skill in skills
    )


def render_skills_by_category(grouped, css_class, empty_message):
    if not grouped:
        st.caption(empty_message)
        return

    for category, skills in grouped.items():
        render(
            f'<div class="skill-category-label">{esc(category)}</div>'
            f'{render_skill_tags(skills, css_class)}'
        )


# ============================================================
# NAVBAR
# ============================================================

render(
    """
    <div class="navbar">
        <div class="logo">Resume<span>AI</span></div>
        <div class="status">Matching Ready</div>
    </div>
    """
)


# ============================================================
# HERO
# ============================================================

render(
    """
    <div class="hero">
        <h1>Find the right candidate, <span>faster.</span></h1>
        <p>
            Upload resumes and compare them against a job
            description using semantic matching and skill
            analysis.
        </p>
    </div>
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

col1, col2 = st.columns(2, gap="large")

with col1:
    render('<div class="section-label">01 — JOB DESCRIPTION</div>')

    job_description = st.text_area(
        "Paste the job description",
        placeholder=(
            "Describe the role, responsibilities, "
            "required skills and qualifications..."
        ),
        height=300,
        label_visibility="collapsed"
    )

with col2:
    render('<div class="section-label">02 — CANDIDATE RESUMES</div>')

    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.caption("Upload one or more PDF or DOCX resumes.")


# ============================================================
# ACTIONS
# ============================================================

st.write("")

action_col1, action_col2, _ = st.columns([1.1, 1, 3])

with action_col1:
    analyze_button = st.button(
        "Analyze Candidates →",
        type="primary",
        use_container_width=True
    )

with action_col2:
    clear_button = False

    if "results" in st.session_state:
        clear_button = st.button(
            "Clear Results",
            use_container_width=True
        )

    if clear_button:
        del st.session_state["results"]
        st.rerun()


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not job_description.strip():
        st.warning("Please enter a job description.")

    elif not uploaded_files:
        st.warning("Please upload at least one resume.")

    else:
        results = []
        failed_files = []

        progress_bar = st.progress(0)
        status_text = st.empty()
        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):

            try:
                status_text.write(f"Analyzing {uploaded_file.name}...")

                filename, resume_text = extract_resume_text(uploaded_file)

                analysis = analyze_resume(job_description, resume_text)
                analysis["filename"] = filename
                analysis["final_score"] = clamp(analysis["final_score"])
                analysis["semantic_score"] = clamp(analysis["semantic_score"])
                analysis["skill_score"] = clamp(analysis["skill_score"])

                results.append(analysis)

            except Exception as error:
                failed_files.append((uploaded_file.name, str(error)))

            progress_bar.progress((index + 1) / total_files)

        status_text.empty()
        progress_bar.empty()

        for filename, message in failed_files:
            st.error(f"Could not analyze **{filename}**: {message}")

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        if results:
            st.session_state["results"] = results
        else:
            st.session_state.pop("results", None)
            st.info("No resumes could be analyzed. Please check the files and try again.")


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "results" in st.session_state:

    results = st.session_state["results"]

    render(
        """
        <div class="results-header">
            <h2>Candidate Rankings</h2>
            <p>Candidates ranked by semantic similarity and skill alignment.</p>
        </div>
        """
    )

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    table_data = [
        {
            "Rank": rank,
            "Candidate": result["filename"],
            "Match Score": f"{result['final_score']}%",
            "Semantic": f"{result['semantic_score']}%",
            "Skills": f"{result['skill_score']}%",
        }
        for rank, result in enumerate(results, start=1)
    ]

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    # --------------------------------------------------------
    # CANDIDATE CARDS
    # --------------------------------------------------------

    for rank, result in enumerate(results, start=1):

        is_top = rank == 1
        card_class = "candidate-card top-card" if is_top else "candidate-card"
        badge_class = "rank-badge top-badge" if is_top else "rank-badge"
        score = clamp(result["final_score"])
        top_pill = '<span class="top-pill">Top Match</span>' if is_top else ""

        render(
            f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="candidate-name">
                            <span class="{badge_class}">{rank}</span>
                            {esc(result["filename"])}
                            {top_pill}
                        </div>
                        <div class="score-label" style="margin-top:4px;">Overall Match</div>
                    </div>
                    <div class="score">{score}%</div>
                </div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width:{score}%;"></div>
                </div>
            </div>
            """
        )

        with st.expander(f"View analysis for {result['filename']}"):

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Overall Match", f"{clamp(result['final_score'])}%")
            metric2.metric("Semantic Match", f"{clamp(result['semantic_score'])}%")
            metric3.metric("Skill Match", f"{clamp(result['skill_score'])}%")

            st.write("")

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.markdown("**Matched Skills**")
                render_skills_by_category(
                    result.get("matched_skills_by_category", {}),
                    css_class="skill-tag",
                    empty_message="No matching skills identified."
                )

            with detail_col2:
                st.markdown("**Missing Skills**")
                missing_by_category = result.get("missing_skills_by_category", {})

                if not missing_by_category:
                    st.success("Excellent — no required skills were missing.")
                else:
                    render_skills_by_category(
                        missing_by_category,
                        css_class="missing-tag",
                        empty_message="None"
                    )


# ============================================================
# FOOTER
# ============================================================

render(
    """
    <div class="footer">
        ResumeAI · Candidate screening assistant<br>
        Built with Python, Streamlit and Sentence Transformers<br>
        Built by <b>Mehul Gupta</b>
    </div>
    """
)