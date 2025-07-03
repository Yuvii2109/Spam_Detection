import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import os

# Tell nltk to use bundled data
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))
st.write("NLTK paths:", nltk.data.path)

#── 1) Download NLTK data (only needs to happen once)

#── 2) Prepare text‐processing tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
punctuations = set(string.punctuation)

def transform_text(text: str) -> str:
    """
    1) lowercase
    2) tokenize
    3) keep only alphanumeric
    4) remove stopwords & punctuation
    5) stem
    """
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stop_words and t not in punctuations]
    return " ".join(ps.stem(t) for t in tokens)

#── 3) Load your pre‐fitted vectorizer & model
with open("vectorizer.pkl", "rb") as vf:
    tfidf = pickle.load(vf)

with open("model.pkl", "rb") as mf:
    model = pickle.load(mf)

#── 4) Build the Streamlit UI
st.title("📨 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter your message here -")

if st.button("Predict"):
    if not input_sms:
        st.warning("Please type a message before clicking Predict.")
    else:
        # Pre‐process & vectorize
        transformed = transform_text(input_sms)
        vector_input = tfidf.transform([transformed])

        # Predict
        try:
            pred = model.predict(vector_input)[0]
            st.success("🚫 Spam" if pred == 1 else "✅ Ham")
        except Exception as e:
            st.error(f"Prediction failed - \n{e}")