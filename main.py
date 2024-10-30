import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("Gerador de Imagens com IA")
st.write("✨ Descreva uma imagem e a IA irá gerá-la para você! ✨")

prompt = st.text_input("Digite a descrição da imagem:")

@st.cache_resource
def load_pipeline():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  
    return pipe

pipe = load_pipeline()

if st.button("Gerar Imagem") and prompt:
    with st.spinner("Gerando a imagem..."):
        image = pipe(prompt, guidance_scale=7.5).images[0]  
        st.image(image, caption=f"Imagem gerada para o prompt: {prompt}")
