from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np

torch.cuda.empty_cache()
# Load the Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

st.title("GenAI app.")

# Display a header
st.header("text-to-image generation using stable diffusion.")

# Add user input
text_prompt = st.text_input("Enter text prompt.")

if st.button("Submit"):
    if text_prompt.strip():
        st.write(f"Generating image for the prompt: '{text_prompt}'")
        
        try:
            # Generate the image
            generated_image = pipe(text_prompt).images[0]
            
            # Display the image
            st.image(generated_image, caption="Generated Image")
        except Exception as e:
            st.error(f"Failed to generate image: {e}")
    else:
        st.warning("Please enter a valid text prompt.")