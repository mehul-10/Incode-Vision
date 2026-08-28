# 🖼️ AI Image Caption Generator

An AI-powered image captioning application that analyzes uploaded images and automatically generates natural-language descriptions using a pre-trained **BLIP (Bootstrapping Language-Image Pre-training)** model.

Built as **Project 3** during my Artificial Intelligence Internship at **Incode Vision**.

---

## ✨ Features

- 🖼️ Upload JPG, JPEG, and PNG images
- 🤖 AI-powered image understanding
- 📝 Automatic caption generation
- ⚡ Simple and interactive Streamlit interface
- 🔍 Uses a pre-trained vision-language model
- 📱 Clean and responsive UI
- 🧠 No model training required

---

## 🧠 How It Works

The application follows a simple AI pipeline:

```text
        Upload Image
             ↓
      Image Processing
          (Pillow)
             ↓
      BLIP Processor
             ↓
    Vision-Language Model
             ↓
      Caption Generation
             ↓
      Natural Language
          Description