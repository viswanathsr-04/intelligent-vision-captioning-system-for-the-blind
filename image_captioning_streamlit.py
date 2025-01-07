import os
import torch
from PIL import Image
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from gtts import gTTS
from groq import Groq
import io

# Initialize Groq client
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Cache the model, processor, and tokenizer to optimize performance
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

model, processor, tokenizer = load_model()

# Configure the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the captioning function
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def generate_initial_caption(image):
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption

def refine_caption_with_groq(initial_caption):
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Refine the following image caption to be more precise without the loss of context and not more than 40 words in total, so that it would be converted into an audio description of the image being uploaded: '{initial_caption}'"}
        ],
        max_tokens=100,
        model="llama-3.3-70b-versatile"
    )
    refined_caption = response.choices[0].message.content.strip()
    return refined_caption

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Streamlit Application
st.title("Enhanced Image Captioning Application")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    st.write("")
    with st.spinner("Generating caption..."):
        initial_caption = generate_initial_caption(image)
        print(initial_caption)
        refined_caption = refine_caption_with_groq(initial_caption)
    st.success(f"Refined Caption: {refined_caption}")

    # Convert the refined caption to speech
    audio_data = text_to_speech(refined_caption)

    # Play the audio in Streamlit
    st.audio(audio_data, format="audio/mp3")
