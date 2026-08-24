from pathlib import Path

from resume_parser import extract_text_from_pdf


pdf_path = Path(
    "test_resumes/resume.pdf"
)


with open(
    pdf_path,
    "rb"
) as file:

    file_bytes = file.read()


text = extract_text_from_pdf(
    file_bytes
)


print("\nExtracted Resume Text:\n")
print(text[:3000])