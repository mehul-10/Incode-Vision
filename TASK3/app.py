import streamlit as st
from PIL import Image, UnidentifiedImageError

from caption_generator import generate_caption


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VisionAI — Image Caption Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background: #f8fafc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       NAVBAR
    ======================================================== */

    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;

        background: white;

        border: 1px solid #e5e7eb;

        border-radius: 18px;

        padding: 14px 22px;

        margin-bottom: 55px;

        box-shadow:
            0 4px 18px rgba(15, 23, 42, 0.04);
    }


    .brand {
        font-size: 1.25rem;

        font-weight: 800;

        color: #111827;

        letter-spacing: -0.5px;
    }


    .brand span {
        color: #6366f1;
    }


    .nav-badge {

        font-size: 0.78rem;

        font-weight: 600;

        color: #166534;

        background: #f0fdf4;

        border: 1px solid #bbf7d0;

        padding: 7px 13px;

        border-radius: 50px;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {

        text-align: center;

        max-width: 780px;

        margin: 0 auto 55px auto;
    }


    .hero-badge {

        display: inline-block;

        padding: 7px 14px;

        border-radius: 50px;

        background: #eef2ff;

        color: #4f46e5;

        border: 1px solid #c7d2fe;

        font-size: 0.76rem;

        font-weight: 700;

        letter-spacing: 0.7px;

        margin-bottom: 18px;
    }


    .hero h1 {

        font-size: 3.6rem;

        line-height: 1.05;

        font-weight: 800;

        letter-spacing: -2px;

        color: #111827;

        margin-bottom: 18px;
    }


    .hero h1 span {

        color: #6366f1;
    }


    .hero p {

        font-size: 1.08rem;

        line-height: 1.7;

        color: #64748b;

        max-width: 650px;

        margin: auto;
    }


    /* ========================================================
       SECTION LABEL
    ======================================================== */

    .section-label {

        font-size: 0.73rem;

        font-weight: 750;

        color: #94a3b8;

        letter-spacing: 1.5px;

        margin-bottom: 10px;
    }


    /* ========================================================
       UPLOAD CARD
    ======================================================== */

    .upload-card {

        background: white;

        border: 1px solid #e5e7eb;

        border-radius: 22px;

        padding: 28px;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.035);

        min-height: 240px;
    }


    .upload-title {

        font-size: 1.15rem;

        font-weight: 700;

        color: #111827;

        margin-bottom: 6px;
    }


    .upload-description {

        font-size: 0.88rem;

        color: #64748b;

        line-height: 1.6;

        margin-bottom: 20px;
    }


    /* ========================================================
       IMAGE CARD
    ======================================================== */

    .image-card {

        background: white;

        border: 1px solid #e5e7eb;

        border-radius: 22px;

        padding: 14px;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.035);
    }


    /* ========================================================
       CAPTION CARD
    ======================================================== */

    .caption-card {

        background: white;

        border: 1px solid #e5e7eb;

        border-radius: 22px;

        padding: 30px;

        margin-top: 22px;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.035);
    }


    .caption-label {

        font-size: 0.72rem;

        font-weight: 750;

        letter-spacing: 1.4px;

        color: #94a3b8;

        margin-bottom: 10px;
    }


    .caption-text {

        font-size: 1.25rem;

        line-height: 1.65;

        color: #1e293b;

        font-weight: 550;
    }


    /* ========================================================
       MODEL INFO
    ======================================================== */

    .model-card {

        background: #ffffff;

        border: 1px solid #e5e7eb;

        border-radius: 22px;

        padding: 25px;

        height: 100%;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.035);
    }


    .model-name {

        font-size: 1.2rem;

        font-weight: 750;

        color: #111827;

        margin-bottom: 8px;
    }


    .model-description {

        font-size: 0.88rem;

        color: #64748b;

        line-height: 1.65;

        margin-bottom: 20px;
    }


    .tech-pill {

        display: inline-block;

        background: #f8fafc;

        border: 1px solid #e2e8f0;

        color: #475569;

        padding: 6px 10px;

        border-radius: 50px;

        font-size: 0.75rem;

        margin: 3px;
    }


    /* ========================================================
       PIPELINE
    ======================================================== */

    .pipeline {

        display: flex;

        justify-content: space-between;

        align-items: center;

        gap: 10px;

        margin-top: 15px;
    }


    .pipeline-step {

        flex: 1;

        text-align: center;

        background: #f8fafc;

        border: 1px solid #e5e7eb;

        border-radius: 14px;

        padding: 14px 8px;

        font-size: 0.76rem;

        color: #475569;
    }


    .pipeline-number {

        display: block;

        font-weight: 800;

        color: #6366f1;

        margin-bottom: 5px;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {

        text-align: center;

        color: #94a3b8;

        font-size: 0.8rem;

        margin-top: 70px;

        padding-top: 25px;

        border-top: 1px solid #e5e7eb;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 768px) {

        .hero h1 {

            font-size: 2.7rem;

        }

        .pipeline {

            flex-direction: column;

        }

        .pipeline-step {

            width: 100%;

        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVBAR
# ============================================================

st.markdown(
    """
    <div class="navbar">

        <div class="brand">
            Vision<span>AI</span>
        </div>

        <div class="nav-badge">
            ● AI Model Ready
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            VISION × LANGUAGE AI
        </div>

        <h1>
            Give your images<br>
            <span>a voice.</span>
        </h1>

        <p>
            Upload an image and let a vision-language model
            understand its contents and turn what it sees
            into a natural-language description.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN CONTENT
# ============================================================

left, right = st.columns(
    [1.45, 1],
    gap="large"
)


# ============================================================
# LEFT — UPLOAD
# ============================================================

with left:

    st.markdown(
        """
        <div class="section-label">
            IMAGE INPUT
        </div>

        <div class="upload-card">

            <div class="upload-title">
                Upload your image
            </div>

            <div class="upload-description">
                Choose a JPG, JPEG, or PNG image.
                The AI will analyze the visual content
                and generate a description.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# RIGHT — MODEL INFORMATION
# ============================================================

with right:

    st.markdown(
        """
        <div class="section-label">
            AI ENGINE
        </div>

        <div class="model-card">

            <div class="model-name">
                BLIP
            </div>

            <div class="model-description">
                Salesforce's BLIP vision-language model
                combines visual understanding with language
                generation to create image captions.
            </div>

            <span class="tech-pill">
                Transformers
            </span>

            <span class="tech-pill">
                PyTorch
            </span>

            <span class="tech-pill">
                BLIP
            </span>

            <span class="tech-pill">
                Pillow
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except UnidentifiedImageError:

        st.error(
            "The uploaded file is not a valid image."
        )

        st.stop()

    except Exception as error:

        st.error(
            f"Unable to read the image: {error}"
        )

        st.stop()


    st.write("")


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    st.markdown(
        """
        <div class="section-label">
            PREVIEW
        </div>
        """,
        unsafe_allow_html=True
    )


    image_col, action_col = st.columns(
        [2.2, 1],
        gap="large"
    )


    with image_col:

        st.markdown(
            '<div class="image-card">',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with action_col:

        st.markdown(
            """
            <div style="
                background:white;
                border:1px solid #e5e7eb;
                border-radius:22px;
                padding:24px;
            ">

                <div style="
                    color:#94a3b8;
                    font-size:0.72rem;
                    font-weight:750;
                    letter-spacing:1.3px;
                ">
                    READY TO ANALYZE
                </div>

                <div style="
                    color:#111827;
                    font-size:1.05rem;
                    font-weight:700;
                    margin-top:8px;
                ">
                    Your image is ready.
                </div>

                <div style="
                    color:#64748b;
                    font-size:0.83rem;
                    line-height:1.6;
                    margin-top:8px;
                    margin-bottom:20px;
                ">
                    Click below to generate an AI-powered
                    description.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        generate = st.button(
            "✦ Generate Caption",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # GENERATE CAPTION
    # ========================================================

    if generate:

        try:

            with st.spinner(
                "Understanding your image..."
            ):

                caption = generate_caption(
                    image
                )


            if caption:

                st.markdown(
                    """
                    <div class="caption-card">

                        <div class="caption-label">
                            AI GENERATED DESCRIPTION
                        </div>

                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="caption-text">
                        “{caption}”
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.warning(
                    "The model did not generate a caption."
                )


        except Exception as error:

            st.error(
                "Something went wrong while generating "
                "the caption."
            )

            st.exception(error)


# ============================================================
# HOW IT WORKS
# ============================================================

st.write("")

st.markdown(
    """
    <div class="section-label">
        HOW IT WORKS
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="pipeline">

        <div class="pipeline-step">
            <span class="pipeline-number">01</span>
            Upload Image
        </div>

        <div class="pipeline-step">
            <span class="pipeline-number">02</span>
            Process Image
        </div>

        <div class="pipeline-step">
            <span class="pipeline-number">03</span>
            BLIP Analysis
        </div>

        <div class="pipeline-step">
            <span class="pipeline-number">04</span>
            Understand Content
        </div>

        <div class="pipeline-step">
            <span class="pipeline-number">05</span>
            Generate Caption
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

st.write("")

with st.expander(
    "View technical details"
):

    st.markdown(
        """
        ### Model

        **Salesforce/blip-image-captioning-base**

        BLIP (Bootstrapping Language-Image Pre-training)
        is a vision-language model designed for tasks
        involving both images and natural language.

        ### Processing Pipeline

        **Image → Pillow → BLIP Processor → Vision Encoder
        → Language Decoder → Caption**

        ### Technologies

        - Python
        - Streamlit
        - Hugging Face Transformers
        - PyTorch
        - Pillow
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        VisionAI · AI Image Caption Generator

        <br><br>

        Built with Python • Streamlit • BLIP • Hugging Face

        <br>

        Made by <strong>Mehul Gupta</strong>

    </div>
    """,
    unsafe_allow_html=True
)