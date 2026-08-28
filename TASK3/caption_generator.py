from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


MODEL_NAME = "Salesforce/blip-image-captioning-base"


# Load model once
processor = BlipProcessor.from_pretrained(MODEL_NAME)

model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME
)


def generate_caption(image):

    # Make sure image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Process image
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Generate caption
    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=30,
            min_new_tokens=5,
            num_beams=5,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True
        )

    # Convert output to text
    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption