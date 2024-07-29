import streamlit as st
from transformers import pipeline
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import pyttsx3
import PyPDF2
import docx
import speech_recognition as sr
import requests


# Load summarization model
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

extractive_summarizer = load_summarizer()

# Load question-answering model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering")

qa_pipeline = load_qa_model()

def summarize_chunk(chunk):
    summary = extractive_summarizer(chunk, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def extractive_summarize(text):
    max_chunk_size = 1024  # max input size for the model
    overlap = 200  # Overlap between chunks to maintain context
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size - overlap)]

    summaries = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(summarize_chunk, chunk): chunk for chunk in text_chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            summaries.append(future.result())
    
    return ' '.join(summaries)

def highlight_keywords(text: str, keywords: List[str]) -> str:
    for keyword in keywords:
        text = text.replace(keyword, f"<mark style='background-color: yellow; color: green;'>{keyword}</mark>")
    return text

def extract_keywords_tfidf(text: str, num_keywords: int = 5) -> List[str]:
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()
    top_indices = tfidf_scores.argsort()[-num_keywords:][::-1]
    return [feature_names[i] for i in top_indices]

def read_text_aloud(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_audio(file):
    recognizer = sr.Recognizer()
    try:
        # Ensure the file is a WAV file
        if file.type != "audio/wav":
            st.error("Only PCM WAV audio files are supported.")
            return ""

        # Use speech_recognition directly with PCM WAV
        audio_file = sr.AudioFile(file)
        with audio_file as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except (sr.UnknownValueError, sr.RequestError, ValueError) as e:
        st.error(f"Error processing audio file: {e}")
        return ""
        
def submit_feedback(feedback):
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLScAy_JyClKuDT4cxtsV6tToEaYRohv5tVPDdi61wpuwDIZDEA/formResponse"
    form_data = {
        "entry.78796369": feedback  # Use the correct name attribute of the feedback field
    }
    response = requests.post(form_url, data=form_data)
    return response.status_code, response.text


# Streamlit App
st.title("SumAny - Summarize Anything")

# File uploader
uploaded_file = st.file_uploader("Upload a text, PDF, Word, or WAV audio file", type=["txt", "pdf", "docx", "wav"])

text = ""
max_chars = 10000  # Define max_chars

if uploaded_file:
    if uploaded_file.type == "text/plain":
        text = str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_word(uploaded_file)
    elif uploaded_file.type == "audio/wav":
        text = extract_text_from_audio(uploaded_file)

# Text area for pasting text with character counter
text = st.text_area("Or paste your text here", value=text, height=200, key="text_area")
text_length = len(text)
counter_text = f"{text_length}/{max_chars}"

# Custom CSS for the text counter inside the text area
st.markdown(f"""
    <style>
    .stTextArea textarea {{
        position: relative;
    }}
    .stTextArea:after {{
        content: '{counter_text}';
        position: absolute;
        bottom: 10px;
        right: 10px;
        font-size: 12px;
        color: grey;
    }}
    </style>
    """, unsafe_allow_html=True)

st.session_state.context = text

# Sidebar settings
st.sidebar.header("Settings")
highlight_keywords_checkbox = st.sidebar.checkbox("Highlight keywords in summary")
read_aloud_checkbox = st.sidebar.checkbox("Read out the summary")

# Chatbot settings
st.sidebar.header("AskAny-Mini Bot")
with st.sidebar.expander("Chat with our Q/A bot about the text"):
    user_input = st.text_input("You:", "")
    if st.button("Send", key="chat"):
        if user_input.strip():
            with st.spinner('Generating response...'):
                response = qa_pipeline(question=user_input, context=st.session_state.context )
                st.write(f"Bot: {response['answer']}")
        else:
            st.warning("Please enter a query to ask about the text.")

if st.button("Summarize"):
    if text_length > max_chars:
        st.warning(f"Text exceeds the maximum allowed character count of {max_chars}. Please reduce the text length.")
    elif text.strip():  # Check if text is not empty
        with st.spinner('Summarizing...'):
            summary = extractive_summarize(text)
        if highlight_keywords_checkbox:
            keywords = extract_keywords_tfidf(summary)
            summary = highlight_keywords(summary, keywords)
            st.markdown(f"<div>{summary}</div>", unsafe_allow_html=True)
        else:
            st.subheader("Summary")
            st.write(summary)
        
        if read_aloud_checkbox:
            read_text_aloud(summary)
        
        st.session_state['context'] = text
    else:
        st.warning("Please enter some text to summarize.")

st.markdown('___')
st.write(':point_left: Use the menu at left to access settings and bot (click on > if closed).')
st.write(':point_left: Your valuable feedback would be greatly appreciated (click on > if closed).')
st.markdown('___')
        
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Your feedback is valuable !")

if st.sidebar.button("Submit Feedback"):
    if feedback.strip():
        status_code, response_text = submit_feedback(feedback)
        if status_code == 200:
            st.sidebar.success("Feedback submitted successfully!")
        else:
            st.sidebar.error(f"Failed to submit feedback. Status code: {status_code}, Response: {response_text}")
    else:
        st.sidebar.warning("Please enter your feedback before submitting.")
