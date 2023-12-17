from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartTokenizer, MBartForConditionalGeneration
import streamlit as st


pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")

#summirizer
model_name = "IlyaGusev/mbart_ru_sum_gazeta"
Tokenizer = MBartTokenizer.from_pretrained(model_name)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

st.title("Перевод текста с английского на русский")
button = st.button('Перевести')

input_for_st = st.text_input("Введите ваш текст")

if button:
    st.write("Качество требует времени, поэтому подождите пожалуйста пару мгновений :D")
    inputs = tokenizer(input_for_st, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=100)
    out_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    st.write(out_text[0])

model = MBartForConditionalGeneration.from_pretrained(model_name)
model.to("cuda")

input_ids = Tokenizer(
    [out_text[0]],
    max_length=600,
    truncation=True,
    return_tensors="pt",
)["input_ids"].to("cuda")
st.write(input_ids)