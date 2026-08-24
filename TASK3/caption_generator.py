import streamlit as st
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_NAME = "Salesforce/blip-image-captioning-base"


@st.cache_resource
def load_model():
    """Load and cache the pre-trained BLIP model."""

    processor = BlipProcessor.from_pretrained(
        MODEL_NAME
    )

    model = BlipForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    return processor, model


def generate_caption(image):
    """Generate a more descriptive caption for a PIL image."""

    if image is None:
        raise ValueError("No image was provided.")

    processor, model = load_model()

    prompt = "A detailed description of this image is"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        min_new_tokens=10,
        num_beams=8,
        length_penalty=1.2,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption.strip()