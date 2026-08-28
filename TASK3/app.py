import streamlit as st
from PIL import Image, UnidentifiedImageError

from caption_generator import generate_caption


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f172a;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }

    .info-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .caption-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 25px;
        margin-top: 25px;
        text-align: center;
    }

    .caption-text {
        font-size: 1.25rem;
        font-weight: 500;
        line-height: 1.6;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 40px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🖼️ Image Caption AI")

    st.write(
        "Generate meaningful image descriptions "
        "using a pre-trained vision-language model."
    )

    st.divider()

    st.subheader("🤖 AI Model")

    st.markdown(
        """
        **Model:** BLIP

        **Model:** Salesforce/blip-image-captioning-base

        **Framework:** Hugging Face Transformers

        **Task:** Image Captioning
        """
    )

    st.divider()

    st.subheader("⚙️ Pipeline")

    st.markdown(
        """
        1. Upload image
        2. Process image
        3. BLIP analyzes visual content
        4. Generate caption
        5. Display result
        """
    )

    st.divider()

    st.markdown(
        """
        <div style="
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            padding: 10px;
        ">
            Made by <strong style="color:#e2e8f0;">
            Mehul Gupta
            </strong>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🖼️ AI Image Caption Generator</h1>
        <p class="subtitle">
            Upload an image and let AI describe what it sees.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="info-card">
        <h3>📤 Upload an Image</h3>
        <p style="color:#94a3b8;">
            Upload a JPG, JPEG, or PNG image and generate
            an AI-powered caption describing its contents.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
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
            "❌ The uploaded file is not a valid image."
        )

        st.stop()

    except Exception as error:

        st.error(
            f"❌ Unable to read the image: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    st.write("")


    # --------------------------------------------------------
    # GENERATE BUTTON
    # --------------------------------------------------------

    generate = st.button(
        "✨ Generate Caption",
        use_container_width=True
    )


    if generate:

        try:

            with st.spinner(
                "🤖 AI is analyzing your image..."
            ):

                caption = generate_caption(
                    image
                )


            if caption:

                st.markdown(
                    f"""
                    <div class="caption-card">
                        <h3>🤖 Generated Caption</h3>
                        <div class="caption-text">
                            {caption}
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
                "❌ Something went wrong while generating "
                "the caption."
            )

            st.exception(error)


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

with st.expander(
    "🧠 How does the AI generate captions?"
):

    st.markdown(
        """
        ### AI Caption Generation Pipeline

        **Uploaded Image**

        ↓

        **Image Processing with Pillow**

        ↓

        **BLIP Vision-Language Model**

        ↓

        **Visual Feature Understanding**

        ↓

        **Natural Language Generation**

        ↓

        **Generated Caption**

        BLIP is a pre-trained vision-language model that
        combines image understanding with language generation
        to produce a natural-language description of an image.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • BLIP • Hugging Face
    </div>
    """,
    unsafe_allow_html=True
)