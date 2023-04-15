#installing the gradio transformer
#!pip install transformers ipywidgets gradio --upgrade

#importing the libraries
import gradio as gr                   # UI library
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline     # Transformers pipeline


def main():
    # Installing the Pre Trained HuggingFace Pipeline and setting up for En To Fr
    #translation_pipeline = pipeline('translation_en_to_fr')
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    device = 0 if torch.cuda.is_available() else -1

    # Providing the input text for translation
    #translate('I love to travel the world')

    # Creating the User Interface Space
    interface = gr.Interface(fn=translate, 
                             inputs=gr.inputs.Textbox(lines=2, 
                                                      placeholder='Text to translate'), 
                             outputs='text')

    # Launching the interface
    interface.launch()


# Creating a fuction called translate
def translate(text, src_lang, tgt_lang):
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=400, device=device)
    results = translation_pipeline(text)
    return results[0]['translation_text']


if __name__ == "__main__":
    main()