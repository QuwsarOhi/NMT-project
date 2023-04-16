#installing the gradio transformer
#!pip install -q gradio git+https://github.com/huggingface/transformers gradio torch

import gradio as gr
#from transformers import AutoModelForSeq2SeqLM, pipeline
import torch
import json
from transformers import T5ForConditionalGeneration
from model.T5 import T5

# this model was loaded from https://hf.co/models
model = T5('t5-small').to('cuda')
LANGS = ["English", "German", "Italian", "Dutch", "Romanian", "French"]

# Load the weights
with open("config.json", "r") as f:
    config = json.load(f)
    
model.load_state_dict(torch.load(config["weight"]))

def translate(text, src_lang, tgt_lang):
    """
    Translate the text from source lang to target lang
    """

    inputs = ["translate "+src_lang+" to "+tgt_lang+": "+text]
    
    with torch.inference_mode():
        outputs = model.predict(inputs)

    return outputs[0]

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.components.Textbox(label="Text"),
        gr.components.Dropdown(label="Source Language", choices=LANGS),
        gr.components.Dropdown(label="Target Language", choices=LANGS),
    ],
    outputs=["text"],
    #examples=[["Building a translation demo with Gradio is so easy!", "eng_Latn", "spa_Latn"]],
    cache_examples=False,
    title="Language Translator",
    description="This is a GUI for the Language Translation System"
)

demo.launch(share=True)