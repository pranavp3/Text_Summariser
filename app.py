import streamlit as st
import os
import gdown
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch


model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

st.sidebar.title("Text Summarizer App")
inputtext = st.text_area(
    "Text to summarize")

try:
    if inputtext:
        input_text = "summarize: " + inputtext
        tokenized_text = tokenizer.encode(input_text, return_tensors='pt', max_length=512).to(device)
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        st.write(summary)

except Exception as error:
    st.write('Give a proper input')
    print(str(error))

