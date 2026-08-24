import io

import pymupdf
from docx import Document


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF file.
    """

    text = []

    pdf = pymupdf.open(
        stream=file_bytes,
        filetype="pdf"
    )

    for page in pdf:
        page_text = page.get_text()

        if page_text:
            text.append(page_text)

    pdf.close()

    return "\n".join(text).strip()


# ============================================================
# DOCX TEXT EXTRACTION
# ============================================================

def extract_text_from_docx(file_bytes):
    """
    Extract text from a DOCX file.
    """

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():
            paragraphs.append(
                paragraph.text.strip()
            )

    return "\n".join(paragraphs).strip()


# ============================================================
# GENERIC RESUME TEXT EXTRACTION
# ============================================================

def extract_resume_text(uploaded_file):
    """
    Extract text from an uploaded PDF or DOCX file.

    Returns:
        filename
        extracted text
    """

    filename = uploaded_file.name.lower()

    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".pdf"):

        text = extract_text_from_pdf(
            file_bytes
        )

    elif filename.endswith(".docx"):

        text = extract_text_from_docx(
            file_bytes
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please upload a PDF or DOCX resume."
        )

    if not text:

        raise ValueError(
            f"No text could be extracted from {uploaded_file.name}."
        )

    return uploaded_file.name, text