import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration


@st.cache_resource
def load_model():
    """
    Load the pre-trained BLIP image captioning model.
    """

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(
        model_name
    )

    model = BlipForConditionalGeneration.from_pretrained(
        model_name
    )

    return processor, model


def generate_caption(image):
    """
    Generate a caption for the uploaded image.
    """

    processor, model = load_model()

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=50
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption