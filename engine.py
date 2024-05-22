from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en import stop_words
import pandas as pd

from utils import load_terms


model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
spacy_model = spacy.load("en_core_web_lg")


def get_phrases_from_input(input_text : str) -> list:
    phrases = []
    for token in input_text:
        if token.pos_ in {"VERB"} and token.text.lower() not in stop_words.STOP_WORDS:
            phrase = input_text[token.i:token.right_edge.i + 1]
            if len(phrase) > 2 and not all(word.text.lower() in stop_words.STOP_WORDS for word in phrase):
                phrases.append(phrase.text.strip())
    return list(set(phrases))


def compute_similarity(phrases, terms, term_embeds, threshold):
    

    
    results = []
    for phrase in phrases:
        phrase_embedding = model.encode(phrase).reshape(1, -1)
        similarities = np.array([cosine_similarity(phrase_embedding, term_embedding.reshape(1, -1))[0][0] for term_embedding in term_embeds])
        idx = np.argmax(similarities)
        similarity = similarities[idx]
        if similarity > threshold:
            results.append((phrase, terms[idx], similarity))
    return results

def generate_suggestions(results):
    df = pd.DataFrame({"Original Phrase": [i[0] for i in results],
                        "Suggested Phrase": [i[1] for i in results],
                        "Similarity Score": [i[2] for i in results]})
    return df

def process_text(sample_text, threshold: float = 0.45):
    
    terms = load_terms()
    term_embeds = [model.encode(term) for term in terms]
    doc = spacy_model(sample_text)
    phrases_from_input = get_phrases_from_input(doc)
    results = compute_similarity(phrases_from_input, terms, term_embeds, threshold)
    return generate_suggestions(results)

