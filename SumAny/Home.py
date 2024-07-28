import streamlit as st

st.set_page_config(
    page_title="SumAny",
    page_icon="📝",
)

st.title("Welcome to SumAny")

# Add project description
st.markdown("""
    ## About SumAny

    **SumAny** is a powerful text summarization tool designed to help you quickly and efficiently summarize text from various sources. Whether you need to condense a lengthy article, a PDF document, a Word file, or even an audio recording, SumAny provides a seamless and intuitive interface to get the job done.

    ### Key Features:
    - **Text Summarization**: Paste your text directly or upload files (PDF, Word, text, audio) to generate concise summaries.
    - **Keyword Highlighting**: Automatically highlights important keywords and entities within the summarized text.
    - **Text-to-Speech**: Listen to your summaries with our integrated text-to-speech functionality.
    - **Interactive Chatbot**: Ask questions about the summarized content using our chatbot powered by advanced NLP models.
    
    ### How to Use:
    1. **Paste or Upload**: Paste your text directly into the provided text area or upload a supported file.
    2. **Customize**: Choose whether to highlight keywords and enable text-to-speech from the settings in the sidebar.
    3. **Summarize**: Click the "Summarize" button to generate your summary.
    4. **Interact**: Use the Short Q/A chatbot to ask questions about the summarized text for deeper insights.

    Explore the features by selecting a page from the sidebar!
""")

st.sidebar.success("Select a page above.")
