import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text Preprocessing Function (same as used during training)
def preprocess_text(text):
    # Remove special characters, numbers, and punctuations
    text = re.sub(r'\W', ' ', text)  # Replace non-word characters with space
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\d', '', text)    # Remove digits

    # Convert text to lowercase
    text = text.lower()

    # Tokenize, remove stopwords, and lemmatize
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

# Load the saved model and vectorizer

with open("C:/Users/Abhinay/Downloads/rf_model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)

with open("C:/Users/Abhinay/Downloads/tfidf_vectorizer.pkl", 'rb') as vectorizer_file:
    loaded_tfidf = pickle.load(vectorizer_file)

# Streamlit title and input area
st.title("Real vs Fake News - Using  Random Forest Classifier")

st.write("""
This app classifies whether a given news article is Real or Fake.
""")

# User input
user_input = st.text_area("Enter a news article:")

# Prediction button
if st.button("Classify"):
    if user_input:
        # Preprocess the user input
        cleaned_input = preprocess_text(user_input)
        
        # Vectorize the preprocessed input
        user_input_tfidf = loaded_tfidf.transform([cleaned_input])
        
        # Make prediction
        prediction = loaded_model.predict(user_input_tfidf)
        
        # Output result
        label_map = {0: 'Fake', 1: 'Real'}
        st.write(f"Prediction: **{label_map[prediction[0]]}**")
    else:
        st.write("Please enter some text.")
