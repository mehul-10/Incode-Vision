from PIL import Image
from caption_generator import generate_caption


image = Image.open(
    "test_images/test.jpg"
).convert("RGB")


caption = generate_caption(image)


print("\nGenerated Caption:")
print(caption)