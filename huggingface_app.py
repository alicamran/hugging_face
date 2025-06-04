import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import the required libraries, show clear error message if not available
try:
    from transformers import pipeline
    import tqdm as notebook_tqdm
    from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False

# Get token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# Add title and description
st.title("Sentiment Analysis with Hugging Face")
st.markdown("This app uses DistilBERT model fine-tuned on SST-2 dataset to analyze sentiment of text.")

# Create sidebar with information
with st.sidebar:
    st.header("About")
    st.info("This app demonstrates how to use Hugging Face transformers with Streamlit for sentiment analysis.")
    
    if transformers_available:
        st.header("Authentication")
        if HF_TOKEN:
            st.success("Using Hugging Face token from environment variables")
        else:
            st.warning("""
            No Hugging Face token found. To use private models or avoid rate limits:
            1. Create a .env file in the project root
            2. Add your token as: HF_TOKEN=your_token_here
            3. Restart the app
            """)
    else:
        st.error("⚠️ Required libraries not found")
        st.markdown("""
        Please install the required libraries:
        ```
        pip install transformers huggingface-hub pandas tqdm
        ```
        """)
        st.stop()

# Create the sentiment analysis pipeline
@st.cache_resource
def load_model():
    # If token is available, use it
    if HF_TOKEN:
        return pipeline(
            task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            token=HF_TOKEN
        )
    else:
        # Try without token (works for public models)
        return pipeline(
            task="text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

# Load the classifier
try:
    with st.spinner("Loading model (this may take a minute the first time)..."):
        classifier = load_model()
        model_load_state = st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("""
    Unable to connect to Hugging Face Hub. This could be due to:
    1. Invalid or missing token - Make sure to replace 'YOUR_TOKEN_HERE' with an actual token
    2. Internet connection issues
    3. The model being unavailable
    """)
    st.stop()

# Input section
st.header("Enter Text for Sentiment Analysis")
text_input = st.text_area("Type or paste text here", height=150)

# Process the input when the user clicks the button
if st.button("Analyze Sentiment"):
    if text_input:
        with st.spinner("Analyzing sentiment..."):
            result = classifier(text_input)
        
        # Display results in a nice format
        st.header("Results")
        
        label = result[0]['label']
        score = result[0]['score']
        
        # Create a color based on sentiment
        color = "green" if label == "POSITIVE" else "red"
        
        st.markdown(f"**Sentiment:** <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {score:.4f}")
        
        # Display as gauge chart
        st.progress(score)
        
        # Create a dataframe for the result to show in a table
        df = pd.DataFrame({
            'Label': [label],
            'Confidence': [f"{score:.4f}"]
        })
        st.table(df)
    else:
        st.warning("Please enter some text to analyze.")

