import streamlit as st
from sentence_transformers import SentenceTransformer
import spacy
from engine import process_text


@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_lg")



model = load_model()
spacy_model = load_spacy_model()
sample_text = st.text_area("Text Input:", value='''In today's meeting, we discussed a variety of issues affecting our department. The weather was unusually sunny, a pleasant backdrop to our serious discussions. We came to the consensus that we need to do better in terms of performance. Sally brought doughnuts, which lightened the mood. It's important to make good use of what we have at our disposal. During the coffee break, we talked about the upcoming company picnic. We should aim to be more efficient and look for ways to be more creative in our daily tasks. Growth is essential for our future, but equally important is building strong relationships with our team members. As a reminder, the annual staff survey is due next Friday. Lastly, we agreed that we must take time to look over our plans carefully and consider all angles before moving forward. On a side note, David mentioned that his cat is recovering well from surgery.''',
                           height=400)
if st.button("Suggest Phrases"):

    df = process_text(sample_text=sample_text, model=model, spacy_model=spacy_model)
    st.table(df)